"""Testes dos baselines da leitura da M2 (aulas 2.1, 2.2 e 2.4; princípio 2 do CLAUDE.md).

A mediana madura é a régua honesta: só entram posts publicados ao menos 30 dias antes. As
datas e as medianas esperadas são literais conferidos à mão.
"""

import numpy as np
import pandas as pd
import pytest

from previa.baselines import (
    alvo,
    k_escolhido,
    leitura_janela,
    ler_config,
    linhas_de_avaliacao,
    media_ultimos_k,
    mediana_com_imaturos,
    mediana_janela,
    mediana_madura,
)
from previa.splits import ler_avaliacao

DIA_0 = pd.Timestamp("2025-01-01", tz="UTC")


def _canal(dias, ys, canal="UCa"):
    """Um post por dia da lista, todos à meia-noite UTC."""
    return pd.DataFrame(
        {
            "channel_id": canal,
            "published_at": DIA_0 + pd.to_timedelta(dias, "D"),
            "y": np.asarray(ys, float),
        }
    )


def test_ler_config():
    cfg = ler_config()
    assert cfg["maturidade_dias"] == 30
    assert cfg["janela_dias"] == 14
    assert cfg["n_boot"] == 2000
    assert cfg["ks"] == [3, 5, 10, 20]
    assert cfg["janela_baseline_dias"] == 90


def test_alvo_e_log1p_das_curtidas():
    df = pd.DataFrame({"metric_target": pd.array([0, 99, None], dtype="Int64")})
    assert alvo(df).tolist()[:2] == pytest.approx([0.0, np.log(100)])
    assert np.isnan(alvo(df).iloc[2])


def test_mediana_madura_so_usa_posts_com_30_dias():
    df = _canal([0, 10, 40, 45], [1, 3, 10, 20]).iloc[[3, 1, 0, 2]]  # fora de ordem de data
    m = mediana_madura(df, dias=30, min_posts=1)
    # dia 40: entram os dias 0 e 10 (30 dias antes, inclusive); dia 45: os mesmos
    assert m.loc[[0, 1, 2, 3]].tolist() == pytest.approx([np.nan, np.nan, 2.0, 2.0], nan_ok=True)


def test_mediana_madura_perturbacao():
    # para o post do dia 41: o do dia 11 tem 30 dias (entra), o do dia 12 tem 29 (não entra)
    df = _canal([0, 11, 12, 41], [1, 3, 5, 7])
    base = mediana_madura(df, dias=30, min_posts=1).loc[3]
    assert base == 2.0
    df29 = df.copy()
    df29.loc[2, "y"] = 100.0
    assert mediana_madura(df29, dias=30, min_posts=1).loc[3] == base
    df30 = df.copy()
    df30.loc[1, "y"] = 100.0
    assert mediana_madura(df30, dias=30, min_posts=1).loc[3] != base


def test_mediana_madura_ignora_y_ausente():
    df = _canal([0, 5, 40], [1, np.nan, 9])
    assert mediana_madura(df, dias=30, min_posts=1).loc[2] == 1.0
    assert np.isnan(mediana_madura(df, dias=30, min_posts=2).loc[2])  # o NaN não conta


def test_mediana_madura_min_posts():
    df = _canal([0, 5, 40], [1, 2, 9])
    assert np.isnan(mediana_madura(df, dias=30, min_posts=3).loc[2])
    assert mediana_madura(df, dias=30, min_posts=2).loc[2] == 1.5


def test_mediana_madura_por_canal():
    df = pd.concat([_canal([0, 40], [1, 0]), _canal([1, 41], [50, 0], canal="UCb")])
    df = df.reset_index(drop=True)
    m = mediana_madura(df, dias=30, min_posts=1)
    assert m.loc[1] == 1.0  # UCa, dia 40: só o post de UCa
    assert m.loc[3] == 50.0  # UCb, dia 41: só o post de UCb


def test_mediana_com_imaturos_usa_o_post_de_ontem():
    df = _canal([0, 1, 2], [1, 3, 5])
    m = mediana_com_imaturos(df)
    assert m.tolist() == pytest.approx([np.nan, 1.0, 2.0], nan_ok=True)


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
  n_grupos: 4
  seed: 0
"""


def test_linhas_de_avaliacao(tmp_path):
    caminho = tmp_path / "avaliacao.yaml"
    caminho.write_text(AVALIACAO)
    cfg = ler_avaliacao(caminho)
    posts = [
        # post_id, data, duplicate_group, y
        ("antigo_2024", "2024-06-01", "antigo_2024", 1.0),  # antes dos folds: fora
        ("tri1_ok____", "2025-02-01", "tri1_ok____", 1.0),  # entra
        ("tri1_copia_", "2025-02-02", "tri1_ok____", 1.0),  # cópia: fora
        ("tri2_sem_y_", "2025-05-01", "tri2_sem_y_", np.nan),  # sem y: fora
        ("tri2_ok____", "2025-05-02", "tri2_ok____", 1.0),  # entra
        ("tri3_ok____", "2025-08-01", "tri3_ok____", 1.0),  # entra
        ("tri4_ok____", "2025-11-01", "tri4_ok____", 1.0),  # entra
        ("calibracao_", "2026-02-15", "calibracao_", 1.0),  # calibração: fora
    ]
    df = pd.DataFrame(posts, columns=["post_id", "published_at", "duplicate_group", "y"])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["channel_id"] = "UCa"
    linhas = linhas_de_avaliacao(df, cfg)
    assert sorted(df.loc[linhas, "post_id"]) == [
        "tri1_ok____",
        "tri2_ok____",
        "tri3_ok____",
        "tri4_ok____",
    ]


# --- baselines mais fortes (aula 2.1: compare com o baseline mais forte que você conseguir) ---


def test_media_ultimos_k_caso_a_mao():
    # seis posts maduros com y = 1..6 (dias 0 a 5); o post do dia 40 enxerga os seis
    df = _canal([0, 1, 2, 3, 4, 5, 40], [1, 2, 3, 4, 5, 6, np.nan])
    assert media_ultimos_k(df, k=3).loc[6] == 5.0  # (4 + 5 + 6) / 3
    assert media_ultimos_k(df, k=10).loc[6] == 3.5  # só há 6: usa os que houver
    # com 2 maduros (dias 0 e 1), o post do dia 40 não tem baseline: NaN, nunca um zero
    dois = _canal([0, 1, 40], [1, 2, np.nan])
    assert np.isnan(media_ultimos_k(dois, k=3).loc[2])
    assert media_ultimos_k(dois, k=3, min_posts=2).loc[2] == 1.5


def test_ultimos_k_so_maduros():
    # post do dia 60: o do dia 30 tem 30 dias (entra), o do dia 31 tem 29 (fica de fora)
    df = _canal([0, 10, 20, 30, 31, 60], [1, 2, 3, 4, 100, np.nan])
    base = media_ultimos_k(df, k=3).loc[5]
    assert base == 3.0  # (2 + 3 + 4) / 3
    df29 = df.copy()
    df29.loc[4, "y"] = -50.0
    assert media_ultimos_k(df29, k=3).loc[5] == base
    df30 = df.copy()
    df30.loc[3, "y"] = 40.0
    assert media_ultimos_k(df30, k=3).loc[5] != base


def test_media_ultimos_k_ignora_y_ausente_e_separa_canais():
    a = _canal([0, 1, 2, 3, 50], [1, np.nan, 3, 5, np.nan], canal="UCa")
    b = _canal([0, 1, 2, 3, 50], [100, 100, 100, 100, np.nan], canal="UCb")
    m = media_ultimos_k(pd.concat([a, b], ignore_index=True), k=3)
    assert m.loc[4] == 3.0  # UCa: (1 + 3 + 5) / 3, o NaN não conta como post
    assert m.loc[9] == 100.0  # UCb: o outro canal não entra


def test_media_ultimos_k_pede_k_positivo():
    with pytest.raises(ValueError, match="k"):
        media_ultimos_k(_canal([0, 40], [1, np.nan]), k=0)


def test_media_ultimos_k_fora_de_ordem_de_data():
    df = _canal([0, 1, 2, 40], [1, 2, 3, np.nan]).iloc[[3, 1, 0, 2]]
    m = media_ultimos_k(df, k=2)
    assert m.loc[3] == 2.5  # (2 + 3) / 2
    assert list(m.index) == [3, 1, 0, 2]  # no índice de entrada


def test_mediana_janela_caso_a_mao():
    # post do dia 320: maduros até o dia 290; janela de 90 dias = dias 200 a 290
    dias = list(range(0, 300, 20)) + [320]  # 15 posts de y = 1..15, mais o do dia 320
    df = _canal(dias, list(range(1, 16)) + [np.nan])
    assert mediana_janela(df).loc[15] == 13.0  # dias 200..280: y = 11..15


def test_mediana_janela_bordas_inclusivas():
    # post do dia 150: janela = dias 30 a 120. Com 3 posts na janela, vale a mediana deles (20);
    # se a borda saísse, sobrariam 2 e o fallback (os 10 últimos maduros) daria 15.
    baixo = _canal([0, 30, 60, 90, 150], [0, 10, 20, 30, np.nan])  # dia 30: t - 120, inclusive
    assert mediana_janela(baixo).loc[4] == 20.0
    alto = _canal([0, 40, 80, 120, 150], [0, 10, 20, 30, np.nan])  # dia 120: t - 30, inclusive
    assert mediana_janela(alto).loc[4] == 20.0
    # dia 29 (t - 121, antes da janela) e dia 121 (t - 29, imaturo) ficam de fora
    fora = _canal([0, 29, 60, 90, 120, 121, 150], [0, 999, 10, 20, 30, 999, np.nan])
    assert mediana_janela(fora).loc[6] == 20.0


def test_mediana_janela_cai_para_os_10_ultimos():
    # post do dia 400: maduros até o dia 370, mas a janela (dias 280 a 370) só tem o do dia 280
    dias = list(range(0, 300, 20)) + [400]  # 15 posts de y = 1..15
    df = _canal(dias, list(range(1, 16)) + [np.nan])
    assert mediana_janela(df).loc[15] == 10.5  # os 10 últimos maduros: y = 6..15
    # com 5 maduros só, o fallback usa os que houver
    poucos = _canal([0, 20, 40, 60, 80, 400], [1, 2, 3, 4, 5, np.nan])
    assert mediana_janela(poucos).loc[5] == 3.0


def test_mediana_janela_min_posts_e_y_ausente():
    df = _canal([0, 5, 60], [1, 2, np.nan])
    assert np.isnan(mediana_janela(df).loc[2])  # 2 maduros < min_posts = 3
    assert mediana_janela(df, min_posts=2).loc[2] == 1.5
    com_nan = _canal([0, 5, 6, 60], [1, np.nan, 3, np.nan])
    assert np.isnan(mediana_janela(com_nan, min_posts=3).loc[3])  # o NaN não conta como post


def test_baselines_fortes_perturbar_o_futuro_nao_muda():
    # post de interesse: o do dia 100. Imaturos (dias 71 a 100) e futuros (dias 130 e 200) não
    # podem mexer em nenhum dos dois baselines (princípio 2).
    dias = [0, 20, 40, 60, 70, 71, 85, 100, 130, 200]
    df = _canal(dias, [1, 2, 3, 4, 5, 6, 7, np.nan, 9, 10])
    alvo_i = 7
    base_k = media_ultimos_k(df, k=3).loc[alvo_i]
    base_j = mediana_janela(df).loc[alvo_i]
    assert base_k == 4.0 and base_j == 3.0  # maduros: dias 0..70 (y = 1..5)
    for i in (5, 6, 7, 8, 9):  # dias 71, 85, 100 (o próprio), 130 e 200
        d = df.copy()
        d.loc[i, "y"] = 1e6
        assert media_ultimos_k(d, k=3).loc[alvo_i] == base_k
        assert mediana_janela(d).loc[alvo_i] == base_j
    # já os maduros mexem: os 3 últimos (y = 3, 4, 5) na média, os 5 da janela na mediana (que só
    # muda se o valor trocar de lado dela, por isso o sinal)
    for i in (2, 3, 4):
        d = df.copy()
        d.loc[i, "y"] = 1e6
        assert media_ultimos_k(d, k=3).loc[alvo_i] != base_k
    for i in (0, 1, 2, 3, 4):
        d = df.copy()
        d.loc[i, "y"] = 1e6 if df.loc[i, "y"] <= 3 else -1e6
        assert mediana_janela(d).loc[alvo_i] != base_j


def _base_para_leitura():
    """Três canais com um post a cada 5 dias, de out/2024 a dez/2025: dentro do treino."""
    rng = np.random.default_rng(0)
    partes = []
    for c in range(3):
        datas = pd.date_range("2024-10-01", "2025-12-20", freq="5D", tz="UTC")
        partes.append(
            pd.DataFrame(
                {
                    "channel_id": f"UC{c}",
                    "post_id": [f"UC{c}_{i:03d}" for i in range(len(datas))],
                    "published_at": datas,
                    "metric_target": rng.integers(10, 5000, len(datas)),
                }
            )
        )
    df = pd.concat(partes, ignore_index=True)
    df["duplicate_group"] = df["post_id"]
    return df


def test_leitura_janela_mesmas_linhas(tmp_path):
    caminho = tmp_path / "avaliacao.yaml"
    caminho.write_text(AVALIACAO)
    cfg_av = ler_avaliacao(caminho)
    cfg = {**ler_config(), "n_boot": 50, "ks": [3, 5]}
    df = _base_para_leitura()
    painel = leitura_janela(df, cfg, cfg_av)
    assert list(painel.columns) == [
        "p_madura",
        "p_com_imaturos",
        "p_ultimos_3",
        "p_ultimos_5",
        "p_mediana_90d",
    ]
    assert painel.loc["linhas"].nunique() == 1  # as mesmas linhas para todos (2.1, regra 3)
    assert painel.loc["pares_janela"].nunique() == 1
    # as linhas dos dois baselines antigos não encolhem com as colunas novas
    d = df.copy()
    d["y"] = alvo(d)
    ok = d.loc[linhas_de_avaliacao(d, cfg_av)].index
    antigas = (
        mediana_madura(d, cfg["maturidade_dias"], cfg["min_posts"]).loc[ok].notna()
        & mediana_com_imaturos(d).loc[ok].notna()
    )
    assert painel.loc["linhas", "p_madura"] == antigas.sum() > 0


def test_k_escolhido_e_o_de_maior_manchete():
    painel = pd.DataFrame(
        {
            "p_madura": {"manchete_janela_peso_posts": 0.50},
            "p_ultimos_3": {"manchete_janela_peso_posts": 0.51},
            "p_ultimos_10": {"manchete_janela_peso_posts": 0.55},
            "p_ultimos_20": {"manchete_janela_peso_posts": 0.53},
            "p_mediana_90d": {"manchete_janela_peso_posts": 0.90},  # não é um k
        }
    )
    assert k_escolhido(painel) == (10, 0.55)
    with pytest.raises(ValueError, match="p_ultimos"):
        k_escolhido(painel[["p_madura"]])
