"""Testes do modelo de uso (M4 final; aulas 6.1 e 6.3; princípio 4 do CLAUDE.md).

O teste central é a paridade: as features que `features_de_um_short` monta para um Short "novo",
só com os posts anteriores do canal, são as mesmas que o treino calculou para ele.
"""

import numpy as np
import pandas as pd
import pytest

from previa.experimento import ler_config, preparar
from previa.modelo import (
    carregar,
    features_de_um_short,
    prever_log,
    referencia,
    salvar,
    treinar_bundle,
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
features: [hist_n, hist_mediana, tendencia, dias_desde_ultimo, posts_30d, hora_local,
           feriado, idade_canal_dias, duracao_s, setor, txt_caracteres, txt_cta]
max_arvores: 60
paciencia: 10
lgbm: {objective: regression, learning_rate: 0.1, num_leaves: 7, min_data_in_leaf: 20}
pca: {titulo: 2}
"""


@pytest.fixture
def cfg_av(tmp_path):
    (tmp_path / "avaliacao.yaml").write_text(AVALIACAO)
    return ler_avaliacao(tmp_path / "avaliacao.yaml")


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "exp.yaml").write_text(CONFIG)
    return ler_config(tmp_path / "exp.yaml")


@pytest.fixture
def base():
    """4 canais, um post a cada 5 dias de 2021-05 a 2026-01, com título, duração e embedding."""
    rng = np.random.default_rng(0)
    linhas = []
    for c in range(4):
        datas = pd.date_range("2021-05-01", "2026-01-31", freq="5D", tz="UTC")
        datas = datas + pd.to_timedelta(rng.integers(0, 24, len(datas)), "h")
        for k, t in enumerate(datas):
            pid = f"c{c}p{k:05d}".ljust(11, "x")
            dur = int(rng.integers(10, 60))
            curt = int(np.expm1(5 + 0.3 * c + 0.02 * dur + rng.normal(0, 0.5)))
            titulo = "Comenta aqui! " * int(rng.integers(0, 2)) + f"Short {k} do canal {c}"
            linhas.append((pid, f"UC{c}", t, curt, "beleza" if c < 2 else "automotivo", pid, dur))
            linhas[-1] += (titulo + "\ndescrição padrão",)
    cols = ["post_id", "channel_id", "published_at", "metric_target", "sector"]
    cols += ["duplicate_group", "duration_s", "caption"]
    return pd.DataFrame(linhas, columns=cols)


def _partes(base):
    dur = base.set_index("post_id")["duration_s"]
    emb = pd.DataFrame(np.random.default_rng(1).normal(size=(len(base), 5)), index=base["post_id"])
    return base.drop(columns="duration_s"), dur, {"titulo": emb}


def test_paridade_treino_uso(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    d = preparar(b, dur, cfg_av, cfg.lacuna_dias)
    alvo = d.index[d["linha"] & (d["channel_id"] == "UC1")][-40]  # um post do fim de 2025
    linha = d.loc[alvo]
    anteriores = d[(d["channel_id"] == "UC1") & (d["published_at"] < linha["published_at"])]
    novo = {
        "published_at": linha["published_at"],
        "caption": linha["caption"],
        "sector": linha["sector"],
        "duration_s": float(dur[linha["post_id"]]),
        "embeddings": {"titulo": emb["titulo"].loc[linha["post_id"]].to_numpy()},
    }
    x = features_de_um_short(bundle, anteriores[["channel_id", "published_at", "y"]], novo)
    sem_pca = [f for f in cfg.features if f != "setor"]
    np.testing.assert_allclose(
        x[sem_pca].to_numpy(float)[0], d.loc[alvo, sem_pca].to_numpy(float), equal_nan=True
    )
    assert x["setor"].iloc[0] == linha["setor"]
    esperado = bundle.pcas["titulo"].transform(novo["embeddings"]["titulo"][None, :])[0]
    np.testing.assert_allclose(x[["titulo_pc00", "titulo_pc01"]].to_numpy()[0], esperado)


def test_salva_carrega_e_preve_igual(base, cfg, cfg_av, tmp_path):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    salvar(bundle, tmp_path / "artefatos")
    carregado = carregar(tmp_path / "artefatos")
    novo = {
        "published_at": pd.Timestamp("2026-01-20T22:00Z"),
        "caption": "Comenta aqui! Título novo",
        "sector": "beleza",
        "duration_s": 30.0,
        "embeddings": {"titulo": np.zeros(5)},
    }
    posts = preparar(b, dur, cfg_av)[["channel_id", "published_at", "y"]]
    posts = posts[posts["channel_id"] == "UC0"]
    x = features_de_um_short(carregado, posts, novo)
    ref, n_ref = referencia(posts, novo["published_at"])
    assert n_ref >= 10
    assert prever_log(carregado, x, ref, n_ref) == pytest.approx(prever_log(bundle, x, ref, n_ref))
    assert prever_log(bundle, x, ref, n_ref) == pytest.approx(bundle.modelo.predict(x)[0] + ref)
    with pytest.raises(ValueError, match="ancorar"):
        prever_log(bundle, x, float("nan"), 0)
    with pytest.raises(ValueError, match="3 posts"):  # regime nunca avaliado (#20; revisão)
        prever_log(bundle, x, ref, 2)
    assert carregado.versao == bundle.versao and bundle.versao.startswith("gbm-teste-")


def test_bundle_treina_so_ate_o_fim_do_treino(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    assert pd.Timestamp(bundle.ultimo_treino) < cfg_av["cal_inicio"] - pd.Timedelta(days=30)


def test_referencia_bate_com_a_do_treino(base, cfg, cfg_av):
    b, dur, _ = _partes(base)
    d = preparar(b, dur, cfg_av)
    alvo = d.index[d["linha"] & (d["channel_id"] == "UC2")][-10]
    linha = d.loc[alvo]
    anteriores = d[(d["channel_id"] == "UC2") & (d["published_at"] < linha["published_at"])]
    ref, n_ref = referencia(anteriores[["published_at", "y"]], linha["published_at"])
    assert ref == pytest.approx(linha["ref"]) and n_ref == linha["n_ref"]


FEATURES_PRODUCAO = """\
nome: producao
hipotese: H1
descricao: a lista da opcao_a_texto
features: [hist_n, hist_mediana, hist_media, hist_desvio, hist_med5, hist_med20, tendencia,
           dias_desde_ultimo, posts_30d, hora_local, dia_semana_local, feriado, idade_canal_dias,
           duracao_s, setor, txt_caracteres, txt_palavras, txt_hashtags, txt_emojis, txt_pergunta,
           txt_cta, txt_numeros, txt_frac_maiusculas, txt_ingles, txt_termos_setor]
max_arvores: 30
paciencia: 5
lgbm: {objective: regression, learning_rate: 0.1, num_leaves: 7, min_data_in_leaf: 20}
pca: {titulo: 2}
"""


def test_paridade_com_a_lista_de_producao_e_historico_completo(base, cfg_av, tmp_path):
    (tmp_path / "prod.yaml").write_text(FEATURES_PRODUCAO)
    cfg = ler_config(tmp_path / "prod.yaml")
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    d = preparar(b, dur, cfg_av)
    alvo = d.index[d["linha"] & (d["channel_id"] == "UC3")][-60]
    linha = d.loc[alvo]
    novo = {
        "published_at": linha["published_at"],
        "caption": linha["caption"],
        "sector": linha["sector"],
        "duration_s": float(dur[linha["post_id"]]),
        "embeddings": {"titulo": emb["titulo"].loc[linha["post_id"]].to_numpy()},
    }
    canal = d[d["channel_id"] == "UC3"][["channel_id", "published_at", "y"]]
    so_antes = canal[canal["published_at"] < linha["published_at"]]
    x_antes = features_de_um_short(bundle, so_antes, novo)
    x_tudo = features_de_um_short(bundle, canal, novo)  # inclui o próprio post e os seguintes
    nao_pca = [f for f in bundle.features if not f.startswith("titulo_pc")]
    np.testing.assert_allclose(
        x_antes[nao_pca].to_numpy(float)[0], d.loc[alvo, nao_pca].to_numpy(float), equal_nan=True
    )
    pd.testing.assert_frame_equal(x_antes, x_tudo)


def test_bundle_recusa_extras(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    cfg.extras = ["vlm"]
    with pytest.raises(ValueError, match="extras"):
        treinar_bundle(cfg, b, dur, cfg_av, emb)
