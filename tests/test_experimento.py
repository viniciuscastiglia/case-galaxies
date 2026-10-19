"""Testes do runner de experimentos (aula 2.5) e do GBM da Opção A (M4; aula 3.4).

Base sintética pequena no formato da base mínima: 6 canais, 2021 a 2025, curtidas com efeito do
canal e da duração. O LightGBM roda com poucas árvores para o teste ser rápido.
"""

import numpy as np
import pandas as pd
import pytest

from previa.experimento import (
    aplicar_pca,
    avaliar,
    folds,
    ganho_entre,
    hash_base,
    ler_config,
    linhas_de_treino,
    preparar,
    rodar,
    treinar_gbm,
)
from previa.splits import ler_avaliacao

AVALIACAO = """\
teste_inicio: "2026-04-01"
treino_inicio: "2021-07-01"
gap_dias: 30
cal_inicio: "2026-02-01"
teste_fim: "2026-09-15"
walk_forward:
  n_folds: 4
  inicio: "2025-01-01"
canal_novo:
  n_grupos: 2
  seed: 0
"""

CONFIG = """\
nome: teste
hipotese: H1
descricao: GBM pequeno
features: [hist_n, hist_mediana, tendencia, posts_30d, hora_local, duracao_s, setor]
lacuna_dias: 30
val_dias: 180
seed: 0
max_arvores: 60
paciencia: 10
lgbm: {objective: regression, learning_rate: 0.1, num_leaves: 7, min_data_in_leaf: 20}
"""


@pytest.fixture
def cfg_av(tmp_path):
    (tmp_path / "avaliacao.yaml").write_text(AVALIACAO)
    return ler_avaliacao(tmp_path / "avaliacao.yaml")


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "exp.yaml").write_text(CONFIG)
    return ler_config(tmp_path / "exp.yaml")


def _base(seed=0):
    """6 canais, ~1 post a cada 4 dias de 2021-05 a 2025-12; duração explica parte das curtidas."""
    rng = np.random.default_rng(seed)
    linhas = []
    for c in range(6):
        datas = pd.date_range("2021-05-01", "2025-12-31", freq="4D", tz="UTC")
        datas = datas + pd.to_timedelta(rng.integers(0, 24, len(datas)), "h")
        dur = rng.integers(10, 60, len(datas))
        log_curtidas = 5 + 0.3 * c + 0.02 * dur + rng.normal(0, 0.5, len(datas))
        for k, (t, d, lc) in enumerate(zip(datas, dur, log_curtidas, strict=True)):
            pid = f"c{c}p{k:05d}".ljust(11, "x")
            setor = "beleza" if c < 3 else "automotivo"
            linhas.append((pid, f"UC{c}", t, int(np.expm1(lc)), setor, pid, d))
    df = pd.DataFrame(
        linhas,
        columns=[
            "post_id",
            "channel_id",
            "published_at",
            "metric_target",
            "sector",
            "duplicate_group",
            "duration_s",
        ],
    )
    return df.drop(columns="duration_s"), df.set_index("post_id")["duration_s"]


def test_config_recusa_chave_desconhecida(tmp_path):
    (tmp_path / "erro.yaml").write_text(CONFIG + "gap_dia: 30\n")
    with pytest.raises(TypeError):
        ler_config(tmp_path / "erro.yaml")


def test_hash_da_base_muda_com_um_valor():
    base, _ = _base()
    outra = base.copy()
    outra.loc[5, "metric_target"] += 1
    assert hash_base(base) == hash_base(base.copy())
    assert hash_base(base) != hash_base(outra)


def test_preparar_marca_as_linhas_e_mantem_o_historico(cfg_av):
    base, dur = _base()
    base.loc[200, "duplicate_group"] = base.loc[199, "post_id"]  # cópia
    base.loc[201, "metric_target"] = np.nan  # curtidas ocultas
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    assert not d.loc[200, "linha"] and not d.loc[201, "linha"]
    assert not d.loc[0, "linha"]  # antes de 01/07/2021: só histórico
    assert d.loc[202, "linha"]
    assert set(d["setor"].unique()) == {0.0, 1.0}  # automotivo, beleza
    # o post de 2021-05 entra no histórico de quem vem 30+ dias depois, no mesmo canal
    assert d.loc[20, "hist_n"] >= 1
    assert d["duracao_s"].notna().all()


def test_gbm_early_stopping_refit_e_seed(cfg_av, cfg):
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    idx = d.index[d["linha"] & (d["published_at"] < pd.Timestamp("2024-12-01", tz="UTC"))]
    m1, n1 = treinar_gbm(d, idx, cfg)
    m2, n2 = treinar_gbm(d, idx, cfg)
    assert 1 <= n1 <= cfg.max_arvores
    assert m1.num_trees() == n1  # o refit usa o nº de árvores do early stopping
    x = d.loc[idx[:50], cfg.features]
    np.testing.assert_array_equal(m1.predict(x), m2.predict(x))  # a seed reproduz


def test_folds_respeitam_o_tempo_e_os_canais(cfg_av):
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    f = folds(d, cfg_av)
    assert len(f["walk_forward"]) == 4 and len(f["canal_novo"]) == 2
    for tr, te in f["walk_forward"]:
        assert d.loc[tr, "published_at"].max() < d.loc[te, "published_at"].min() - pd.Timedelta(
            days=30
        )
        assert d.loc[tr, "linha"].all() and d.loc[te, "linha"].all()
    for tr, te in f["canal_novo"]:
        assert set(d.loc[tr, "channel_id"]).isdisjoint(d.loc[te, "channel_id"])
        assert d.loc[tr, "published_at"].max() < pd.Timestamp("2024-12-02", tz="UTC")
        assert d.loc[te, "published_at"].min() >= pd.Timestamp("2025-01-01", tz="UTC")


def test_avaliar_modelo_igual_ao_baseline_da_ganho_zero(cfg_av):
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    te = d.index[d["linha"] & (d["published_at"] >= pd.Timestamp("2025-01-01", tz="UTC"))]
    r = avaliar(d, te, d.loc[te, "p_base"], n_boot=200, seed=0)
    assert r["ganho"] == 0.0 and r["ic_ganho_lo"] == 0.0 and r["ic_ganho_hi"] == 0.0
    assert r["linhas"] == d.loc[te, "p_base"].notna().sum()


def test_rodar_ponta_a_ponta(cfg_av, cfg):
    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=200, seed=0)
    for avaliacao in ("walk_forward", "canal_novo"):
        assert {"manchete_modelo", "manchete_base", "ganho", "ic_ganho_lo"} <= set(r[avaliacao])
    # a duração explica parte das curtidas na base sintética: o GBM ganha da mediana do canal
    assert r["walk_forward"]["ic_ganho_lo"] > 0
    # a âncora (+ ref) não pode se perder: sem ela, o MAE em log explode (revisão de 19/10)
    assert abs(r["walk_forward"]["mae_log_modelo"] - r["walk_forward"]["mae_log_base"]) < 0.5


def test_pca_ajustado_so_no_treino():
    rng = np.random.default_rng(0)
    emb = pd.DataFrame(rng.normal(size=(40, 6)), index=[f"v{k:02d}" for k in range(40)])
    emb.iloc[30:] += 50  # o "teste" é muito diferente: se entrasse no ajuste, a média mudaria
    treino, todos = list(emb.index[:30]), list(emb.index) + ["sem_emb"]
    cols, pca = aplicar_pca(emb, treino, todos, n=2, prefixo="titulo", seed=0)
    np.testing.assert_allclose(pca.mean_, emb.iloc[:30].mean().to_numpy())
    assert list(cols.columns) == ["titulo_pc00", "titulo_pc01"]
    assert cols.loc["sem_emb"].isna().all() and cols.index.tolist() == todos


def test_rodar_com_bloco_de_embedding(cfg_av, cfg):
    base, dur = _base()
    rng = np.random.default_rng(1)
    emb = pd.DataFrame(rng.normal(size=(len(base), 5)), index=base["post_id"])
    cfg.pca = {"titulo": 2}
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0, embeddings={"titulo": emb})
    assert r["walk_forward"]["n_features"] == len(cfg.features) + 2


def test_rodar_devolve_as_previsoes_por_post(cfg_av, cfg):
    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0)
    prev = r["walk_forward"]["previsoes"]
    assert prev.index.isin(base["post_id"]).all() and prev.notna().all()
    assert len(prev) >= r["walk_forward"]["linhas"]


def test_ganho_entre_dois_modelos_iguais_e_zero(cfg_av):
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    te = d.index[d["linha"] & (d["published_at"] >= pd.Timestamp("2025-01-01", tz="UTC"))]
    p = d.loc[te, "hist_media"]
    r = ganho_entre(d, te, p, p, n_boot=100, seed=0)
    assert r["ganho"] == 0.0 and r["ic_ganho_lo"] == 0.0 and r["ic_ganho_hi"] == 0.0


def test_alvo_relativo_ao_canal(cfg_av):
    # decisão de 05/10 (README, aula 1.6): log da razão entre as curtidas e a mediana madura dos
    # posts anteriores, definido com 10+ posts maduros (#20)
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    ok = d["n_ref"] >= 10
    np.testing.assert_allclose(d.loc[ok, "y_rel"], d.loc[ok, "y"] - d.loc[ok, "ref"])
    assert d.loc[~ok, "y_rel"].isna().all() and (~ok).sum() > 0
    assert (d.loc[d["p_base"].notna(), "p_base"] == d.loc[d["p_base"].notna(), "ref"]).all()


def test_treino_relativo_so_com_alvo_definido(cfg_av, cfg):
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    tr = d.index[d["linha"]]
    assert (d.loc[linhas_de_treino(d, tr, cfg), "n_ref"] >= 10).all()
    cfg.alvo = "absoluto"
    assert len(linhas_de_treino(d, tr, cfg)) == len(tr)


def test_teto_por_canal_fica_com_os_mais_recentes(cfg_av, cfg):
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    tr = d.index[d["linha"]]
    cfg.teto_por_canal = 50
    sel = linhas_de_treino(d, tr, cfg)
    assert d.loc[sel].groupby("channel_id").size().max() == 50
    ultimo = d.loc[
        linhas_de_treino(d, tr, cfg.__class__(**{**cfg.__dict__, "teto_por_canal": None}))
    ]
    mais_recente = ultimo.groupby("channel_id")["published_at"].max()
    assert (d.loc[sel].groupby("channel_id")["published_at"].max() == mais_recente).all()


def test_idade_do_rotulo_filtra_e_fixa_na_previsao(cfg_av, cfg):
    from previa.experimento import fixar_na_previsao

    base, dur = _base()
    base["target_age_days"] = np.where(np.arange(len(base)) % 2 == 0, 40.0, 400.0)
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    cfg.idade_rotulo_max = 90
    sel = linhas_de_treino(d, d.index[d["linha"]], cfg)
    assert (d.loc[sel, "target_age_days"] <= 90).all() and len(sel) > 0
    cfg.fixar = {"target_age_days": 30}
    x = fixar_na_previsao(d.loc[sel[:5], ["target_age_days", "hora_local"]], cfg)
    assert (x["target_age_days"] == 30).all()


def test_extras_entram_por_join_com_nan_e_prefixo(cfg_av, cfg):
    from previa.experimento import expandir_features

    base, dur = _base()
    ids = base["post_id"].iloc[:10]
    vlm = pd.DataFrame({"vlm_gancho": range(10), "vlm_rosto": [1.0] * 10}, index=ids)
    d = preparar(base, dur, cfg_av, lacuna_dias=30, extras={"vlm": vlm})
    assert d["vlm_gancho"].notna().sum() == 10 and d["vlm_gancho"].isna().sum() == len(d) - 10
    feats = expandir_features(["hist_n", "vlm_*"], d.columns)
    assert feats == ["hist_n", "vlm_gancho", "vlm_rosto"]


def test_avaliar_so_na_amostra(cfg_av, cfg):
    base, dur = _base()
    amostra = set(base["post_id"].iloc[::3])
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0, so_avaliar=amostra)
    assert r["walk_forward"]["previsoes"].index.isin(amostra).all()


def test_pares_so_dentro_do_mesmo_fold(cfg_av):
    # dois Shorts do mesmo canal em folds diferentes tiveram modelos diferentes: o par não conta
    # (na Lumina, A e B sempre têm o mesmo modelo; revisão de 19/10)
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    te = d.index[d["linha"] & (d["published_at"] >= pd.Timestamp("2025-01-01", tz="UTC"))]
    meio = pd.Timestamp("2025-07-01", tz="UTC")
    dobra = pd.Series((d.loc[te, "published_at"] >= meio).astype(int).to_numpy(), index=te)
    junto = avaliar(d, te, d.loc[te, "p_base"], n_boot=50, seed=0)
    separado = avaliar(d, te, d.loc[te, "p_base"], n_boot=50, seed=0, dobras=dobra)
    assert separado["pares"] < junto["pares"]


def test_curinga_sem_colunas_levanta():
    from previa.experimento import expandir_features

    with pytest.raises(ValueError, match="vlm_"):
        expandir_features(["hist_n", "vlm_*"], ["hist_n", "hist_media"])


def test_setor_desconhecido_levanta():
    from previa.experimento import codigo_setor

    with pytest.raises(ValueError, match="Beleza"):
        codigo_setor(pd.Series(["beleza", "Beleza"]))
