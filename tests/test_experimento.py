"""Testes do runner de experimentos (aula 2.5) e do GBM da Opção A (M4; aula 3.4).

Base sintética pequena no formato da base mínima: 6 canais, 2021 a 2025, curtidas com efeito do
canal e da duração. O LightGBM roda com poucas árvores para o teste ser rápido.
"""

import json
import warnings

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from previa.caminhos import CONFIGS
from previa.experimento import (
    EXPERIMENTO,
    LIMITE_IMPORTANCIA,
    aplicar_pca,
    avaliar,
    avisar_importancia_dominante,
    comparar,
    conferir_cortes,
    folds,
    ganho_entre,
    hash_base,
    ler_config,
    linhas_de_treino,
    painel,
    painel_das_previsoes,
    preparar,
    registrar,
    rodar,
    tabela_de_previsoes,
    treinar_gbm,
)
from previa.features import FEATURES
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


def test_preparar_junta_os_quantis_do_canal_das_faixas(cfg_av):
    # decisão #7 (aula 1.6): p25/p75/p95 do histórico maduro, sempre com 30 dias e min_posts_alvo
    base, dur = _base()
    d = preparar(base, dur, cfg_av, lacuna_dias=30)
    q = ["faixa_p25", "faixa_p75", "faixa_p95"]
    assert (d["n_ref"] >= 10).sum() > 1000  # a base sintética tem posts de sobra
    assert (
        d[q].notna().all(axis=1).tolist() == (d["n_ref"] >= 10).tolist()
    )  # a mesma régua do y_rel
    ok = d[q].dropna()
    assert (ok["faixa_p25"] <= d.loc[ok.index, "ref"]).all()  # p25 <= mediana <= p75
    assert (d.loc[ok.index, "ref"] <= ok["faixa_p75"]).all()
    assert (ok["faixa_p75"] <= ok["faixa_p95"]).all()
    assert not {*q} & {*FEATURES}  # a régua das faixas não é feature do modelo


def test_preparar_quantis_usam_sempre_30_dias_e_nao_mudam_o_resto(cfg_av):
    base, dur = _base()
    a = preparar(base, dur, cfg_av, lacuna_dias=30)
    b = preparar(base, dur, cfg_av, lacuna_dias=7)
    q = ["faixa_p25", "faixa_p75", "faixa_p95"]
    pd.testing.assert_frame_equal(a[q], b[q])  # como a `ref`: a régua é de 30 dias
    pd.testing.assert_series_equal(a["ref"], b["ref"])
    assert not a["hist_n"].equals(b["hist_n"])  # e a feature do modelo segue a lacuna pedida


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
    assert r["dif_mae_log"] == 0.0 and r["ic_dif_mae_lo"] == 0.0 and r["ic_dif_mae_hi"] == 0.0


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


def test_bloco_extra_sem_nenhuma_feature_levanta(cfg_av, cfg):
    # 20/10: a H7 juntou a cor e o brilho da capa (`thumb_*`), mas a lista de features só tinha
    # `sonda_*`, e o tratamento rodou sem eles (4.8, erro comum 4). Falha nunca silenciosa.
    base, dur = _base()
    cor = pd.DataFrame({"thumb_brilho": 0.5}, index=base["post_id"])
    with pytest.raises(ValueError, match="aparencia_thumb"):
        rodar(cfg, base, dur, cfg_av, n_boot=50, seed=0, extras={"aparencia_thumb": cor})


# --- painel de métricas (TAPI 6.1; aulas 2.1, 2.3, 2.4) ---


def _caso_a_mao():
    """2 canais, 3 posts cada; o modelo erra 0,5 em todos e o baseline erra 1 em todos.

    Faixas do canal X: p25 5,8, p75 6,2, p95 6,8 (y 5, 6, 7 -> abaixo, na média, viral); as
    previsões 5,5, 6,5 e 7,5 caem em abaixo, acima e viral: F1 = (1 + 0 + 0 + 1) / 4 = 0,5.
    O canal Y é o X somado de 3, as mesmas faixas deslocadas."""
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    return pd.DataFrame(
        {
            "channel_id": ["X"] * 3 + ["Y"] * 3,
            "y": y,
            "p": y + 0.5,
            "p_base": y + 1.0,
            "faixa_p25": [5.8] * 3 + [8.8] * 3,
            "faixa_p75": [6.2] * 3 + [9.2] * 3,
            "faixa_p95": [6.8] * 3 + [9.8] * 3,
            "dobra": [0, 0, 0, 1, 1, 1],
        }
    )


def test_painel_caso_a_mao():
    a = _caso_a_mao()
    r = painel(a, n_boot=200, seed=0)
    assert r["mae_log"] == pytest.approx(0.5)
    assert r["e_mae"] == pytest.approx(np.exp(0.5))  # erra por um fator de 1,65, para cima ou baixo
    assert r["rmse_log"] == pytest.approx(0.5)
    assert r["mape"] == pytest.approx(100 * np.mean(0.5 / a["y"]))
    assert r["spearman_global"] == pytest.approx(1.0)
    assert r["f1_faixas"] == pytest.approx(0.5)
    assert r["skill_mae"] == pytest.approx(0.5)  # 1 - 0,5 / 1 (2.1, regra 3)
    # o erro é o mesmo em todo post: qualquer reamostra de canais dá os mesmos valores
    for k in ("mae_log", "rmse_log", "skill_mae", "f1_faixas"):
        assert r[f"{k}_lo"] == pytest.approx(r[k]) and r[f"{k}_hi"] == pytest.approx(r[k])
    assert r["e_mae_lo"] == pytest.approx(np.exp(r["mae_log_lo"]))  # e^IC: monótona
    assert r["e_mae_hi"] == pytest.approx(np.exp(r["mae_log_hi"]))
    # um fold por canal: cada um com 3 posts em ordem perfeita, o Spearman intra é 1
    assert r["dobra0_mae_log"] == pytest.approx(0.5) and r["dobra1_mae_log"] == pytest.approx(0.5)
    assert r["dobra0_spearman_intra"] == pytest.approx(1.0)
    assert r["dobra1_spearman_intra"] == pytest.approx(1.0)
    assert "dobra0_mae_log_lo" not in r  # por fold, só o valor
    assert not any("cobertura" in k for k in r)  # sem intervalo ainda (M6)


def test_painel_ic_cobre_o_valor_e_varia_com_os_canais():
    # X erra 0,2 em todo post e Y erra 1,0: o MAE do conjunto é 0,6 e a reamostra de canais vai de
    # 0,2 (XX) a 1,0 (YY); o IC tem que mostrar essa incerteza, e o ponto fica dentro
    a = _caso_a_mao()
    a["p"] = a["y"] + np.r_[np.full(3, 0.2), np.full(3, 1.0)]
    r = painel(a, n_boot=500, seed=0)
    assert r["mae_log"] == pytest.approx(0.6)
    assert r["mae_log_lo"] == pytest.approx(0.2) and r["mae_log_hi"] == pytest.approx(1.0)


def test_painel_recusa_nan():
    for col in ("p", "y", "p_base"):
        a = _caso_a_mao()
        a.loc[2, col] = np.nan
        with pytest.raises(ValueError, match=col):
            painel(a, n_boot=20, seed=0)


def test_painel_skill_zero_se_o_modelo_for_o_baseline_e_nan_sem_erro_no_baseline():
    a = _caso_a_mao()
    a["p_base"] = a["p"]
    assert painel(a, n_boot=20, seed=0)["skill_mae"] == 0.0
    a["p_base"], a["p"] = a["y"], a["y"] + 0.5  # baseline perfeito: MAE 0, a razão não existe
    r = painel(a, n_boot=20, seed=0)
    assert np.isnan(r["skill_mae"]) and np.isnan(r["skill_mae_lo"])


def test_painel_f1_sem_historico_e_nan_com_aviso():
    a = _caso_a_mao()
    a[["faixa_p25", "faixa_p75", "faixa_p95"]] = np.nan  # nenhum canal com 10+ posts maduros
    with pytest.warns(UserWarning, match="histórico"):
        r = painel(a, n_boot=20, seed=0)
    assert np.isnan(r["f1_faixas"]) and np.isnan(r["f1_faixas_lo"])
    assert r["mae_log"] == pytest.approx(0.5)  # o resto do painel não depende das faixas


def test_rodar_registra_o_painel(cfg_av, cfg):
    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0)
    metricas = ["mae_log", "e_mae", "rmse_log", "mape", "spearman_global", "f1_faixas", "skill_mae"]
    for avaliacao in ("walk_forward", "canal_novo"):
        res = r[avaliacao]
        for m in metricas:
            for sufixo in ("", "_lo", "_hi"):
                assert np.isfinite(res[f"painel_{m}{sufixo}"]), f"{avaliacao} painel_{m}{sufixo}"
        for k in range(len(res["arvores"])):  # uma entrada por dobra
            assert np.isfinite(res[f"painel_dobra{k}_mae_log"])
            assert np.isfinite(res[f"painel_dobra{k}_spearman_intra"])
        # o painel mede as mesmas linhas das chaves antigas, que continuam aí
        assert {"manchete_modelo", "ganho", "ic_ganho_lo", "mae_log_modelo"} <= set(res)
        assert res["painel_mae_log"] == pytest.approx(res["mae_log_modelo"])
        skill = 1 - res["mae_log_modelo"] / res["mae_log_base"]
        assert res["painel_skill_mae"] == pytest.approx(skill)
        assert res["painel_mae_log_lo"] <= res["painel_mae_log"] <= res["painel_mae_log_hi"]
        assert 0 <= res["painel_f1_faixas"] <= 1


def test_painel_das_previsoes_igual_ao_do_rodar(cfg_av, cfg):
    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0)
    d = preparar(base, dur, cfg_av, cfg.lacuna_dias, min_posts_alvo=cfg.min_posts_alvo)
    tabela = painel_das_previsoes(d, tabela_de_previsoes(r), n_boot=100, seed=0)
    assert set(tabela.columns) == {"walk_forward", "canal_novo"}
    for avaliacao in tabela.columns:
        do_rodar = {
            k.removeprefix("painel_"): v for k, v in r[avaliacao].items() if k.startswith("painel_")
        }
        for k, v in do_rodar.items():
            assert tabela.loc[k, avaliacao] == pytest.approx(v, rel=1e-9, nan_ok=True), k
        # a tabela une as chaves das duas avaliações: o que o rodar não tem (dobra a mais) é NaN
        assert tabela.loc[list(set(tabela.index) - set(do_rodar)), avaliacao].isna().all()


def test_painel_do_run_le_as_previsoes_salvas_sem_treinar(cfg_av, cfg, monkeypatch):
    # a CLI `painel RUN_ID` para os runs lacrados: aqui com o MLflow e a base trocados por fakes
    import previa.experimento as exp

    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0)
    d = preparar(base, dur, cfg_av, cfg.lacuna_dias, min_posts_alvo=cfg.min_posts_alvo)
    esperado = painel_das_previsoes(d, tabela_de_previsoes(r), n_boot=100, seed=0)

    class FakeRun:
        class data:  # noqa: N801
            tags = {"hash_base": hash_base(base)}

    monkeypatch.setattr(exp.mlflow, "set_experiment", lambda nome: None)
    monkeypatch.setattr(exp.mlflow, "get_run", lambda run_id: FakeRun)
    monkeypatch.setattr(exp.mlflow, "load_table", lambda nome, run_ids: tabela_de_previsoes(r))
    monkeypatch.setattr(exp, "carregar_base", lambda: base)
    monkeypatch.setattr(exp, "ler_duracoes", lambda: dur)
    monkeypatch.setattr(exp, "ler_avaliacao", lambda: cfg_av)
    obtido = exp.painel_do_run("RUN", n_boot=100, seed=0, min_posts=3)
    pd.testing.assert_frame_equal(obtido, esperado)
    FakeRun.data.tags = {"hash_base": "outra"}  # run de outra base: falha alto, como o `comparar`
    with pytest.raises(ValueError, match="base atual"):
        exp.painel_do_run("RUN", n_boot=100, seed=0, min_posts=3)


# --- data de corte no run e importância por ganho (F22, F35; aulas 2.2 e 2.5) ---


@pytest.fixture
def mlflow_temporario(tmp_path):
    """O MLflow num sqlite de tmp_path, com os artefatos também lá (o padrão seria `./mlruns`):
    nenhum teste toca o `mlflow.db` do projeto. Restaura o endereço e o experimento ativo."""
    antes = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment(EXPERIMENTO, artifact_location=str(tmp_path / "artefatos"))
    yield tmp_path
    mlflow.set_tracking_uri(antes)
    mlflow.tracking.fluent._active_experiment_id = None


def test_conferir_cortes_avisa_run_sem_tag_e_recusa_cortes_diferentes():
    # runs lacrados antes da F22 não têm `data_corte`: avisa e segue (2.5, erro comum 3)
    with pytest.warns(UserWarning, match="a sem data_corte"):
        conferir_cortes({"a": {}, "b": {"data_corte": "2025-01-01"}}, "2025-01-01")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # todos com o mesmo corte: nem um aviso
        conferir_cortes({"a": {"data_corte": "2025-01-01"}}, "2025-01-01")
    # um run com outro corte que o do outro run, ou que o atual: a comparação não vale
    with pytest.raises(ValueError, match="data_corte"):
        conferir_cortes(
            {"a": {"data_corte": "2025-01-01"}, "b": {"data_corte": "2024-07-01"}}, "2025-01-01"
        )
    with pytest.raises(ValueError, match="2024-07-01"):
        conferir_cortes({"a": {"data_corte": "2024-07-01"}}, "2025-01-01")


def test_importancia_soma_1_e_pega_a_feature_vazada(cfg_av, cfg):
    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0)
    for avaliacao in ("walk_forward", "canal_novo"):
        imp = r[avaliacao]["importancia"]
        assert imp.sum() == pytest.approx(1.0)
        assert set(imp.index) == set(cfg.features)  # o nome vem do modelo, não da posição
        assert imp.is_monotonic_decreasing
        assert r[avaliacao]["importancia_maior_fatia"] == pytest.approx(imp.iloc[0])
    # uma feature vazada (o alvo mais um ruído pequeno) tem que dominar o ganho e disparar o alarme
    d = preparar(base, dur, cfg_av, cfg.lacuna_dias, min_posts_alvo=cfg.min_posts_alvo)
    ruido = np.random.default_rng(1).normal(0, 0.05, len(d))
    vazada = pd.DataFrame({"vazada": d["y_rel"].to_numpy() + ruido}, index=d["post_id"])
    cfg.features = [*cfg.features, "vazada"]
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0, extras={"vazada": vazada})
    for avaliacao in ("walk_forward", "canal_novo"):
        assert r[avaliacao]["importancia"].index[0] == "vazada"
        assert r[avaliacao]["importancia_maior_fatia"] > LIMITE_IMPORTANCIA
    with pytest.warns(UserWarning, match="vazada"):
        avisar_importancia_dominante(r)


def test_alarme_de_importancia_so_acima_do_limite():
    def res(fatia):
        imp = pd.Series({"vazada": fatia, "outra": 1 - fatia})
        return {"walk_forward": {"importancia": imp, "importancia_maior_fatia": fatia}}

    assert LIMITE_IMPORTANCIA == 0.5  # 2.2: "importância dominante"
    with pytest.warns(UserWarning, match=r"walk_forward.*vazada.*0\.80"):
        avisar_importancia_dominante(res(0.8))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        avisar_importancia_dominante(res(0.4))


def test_registrar_grava_corte_e_avaliacao(cfg_av, cfg, mlflow_temporario):
    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0)
    client = mlflow.MlflowClient()

    run_id = registrar(cfg, "exp.yaml", r, base, cfg_av=cfg_av)
    run = client.get_run(run_id)
    assert run.data.tags["data_corte"] == "2025-01-01"  # o início do walk-forward do cfg_av
    assert run.data.tags["gap_dias"] == "30"
    arquivos = {f.path for f in client.list_artifacts(run_id)}
    assert {"avaliacao.json", "importancia_gain.json", "config.json"} <= arquivos
    cru = mlflow.artifacts.load_dict(f"runs:/{run_id}/avaliacao.json")
    assert cru == json.loads(
        json.dumps(yaml.safe_load((CONFIGS / "avaliacao.yaml").read_text(encoding="utf-8")))
    )  # o arquivo cru, como estava na hora do run
    gain = mlflow.artifacts.load_dict(f"runs:/{run_id}/importancia_gain.json")
    assert set(gain) == {"walk_forward", "canal_novo"}
    for fatias in gain.values():
        assert set(fatias) == set(cfg.features)
        assert sum(fatias.values()) == pytest.approx(1.0)
    # a fatia dominante é métrica; a série da importância não (o `log_metrics` pede float)
    assert "walk_forward_importancia_maior_fatia" in run.data.metrics
    assert not any(k.endswith("_importancia") for k in run.data.metrics)

    # sem `cfg_av` (como a ablação chama), o corte sai do avaliacao.yaml do projeto
    sem = client.get_run(registrar(cfg, "exp.yaml", r, base))
    esperado = str(ler_avaliacao()["walk_forward"]["inicio"].date())
    assert sem.data.tags["data_corte"] == esperado


def test_comparar_recusa_cortes_diferentes_antes_do_hash_e_avisa_run_antigo(
    cfg_av, cfg, monkeypatch
):
    import previa.experimento as exp

    base, dur = _base()
    r = rodar(cfg, base, dur, cfg_av, n_boot=100, seed=0)
    tags = {}

    class FakeRun:
        def __init__(self, run_id):
            self.data = type("Dados", (), {"tags": tags[run_id]})

    monkeypatch.setattr(exp.mlflow, "set_experiment", lambda nome: None)
    monkeypatch.setattr(exp.mlflow, "get_run", FakeRun)
    monkeypatch.setattr(exp.mlflow, "load_table", lambda nome, run_ids: tabela_de_previsoes(r))
    monkeypatch.setattr(exp, "carregar_base", lambda: base)
    monkeypatch.setattr(exp, "ler_duracoes", lambda: dur)
    monkeypatch.setattr(exp, "ler_avaliacao", lambda: cfg_av)

    # cortes diferentes E hash errado: o corte é conferido primeiro, e é ele que a mensagem cita
    tags |= {
        "A": {"hash_base": "outra", "data_corte": "2024-07-01"},
        "B": {"hash_base": "outra", "data_corte": "2025-01-01"},
    }
    with pytest.raises(ValueError, match="data_corte"):
        comparar("A", "B", n_boot=50)
    # run lacrado sem a tag: avisa e compara (o hash da base já o confere)
    tags |= {"A": {"hash_base": hash_base(base)}, "B": {"hash_base": hash_base(base)}}
    with pytest.warns(UserWarning, match="sem data_corte"):
        g = comparar("A", "B", n_boot=50)
    assert g.loc["ganho", "walk_forward"] == 0.0  # os mesmos dois modelos
