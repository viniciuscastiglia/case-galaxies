"""Testes dos baselines da leitura da M2 (aulas 2.1, 2.2 e 2.4; princípio 2 do CLAUDE.md).

A mediana madura é a régua honesta: só entram posts publicados ao menos 30 dias antes. As
datas e as medianas esperadas são literais conferidos à mão.
"""

import numpy as np
import pandas as pd
import pytest

from previa.baselines import (
    alvo,
    ler_config,
    linhas_de_avaliacao,
    mediana_com_imaturos,
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
