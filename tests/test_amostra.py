"""Testes da amostra multimodal (decisão #12 de 18/10; aulas 4.1, 4.2 e 4.8).

A regra é escrita antes e é cega às curtidas: 20 por canal, seed 0, só primeiras cópias,
proporcional aos blocos de tempo do canal.
"""

import pandas as pd
import pytest

from previa.amostra import alocar, sortear_amostra
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
  n_grupos: 4
  seed: 0
"""


@pytest.fixture
def cfg(tmp_path):
    caminho = tmp_path / "avaliacao.yaml"
    caminho.write_text(AVALIACAO)
    return ler_avaliacao(caminho)


def _base(canal, n_treino, n_cal, n_teste, n_gap=0, n_antigo=0):
    """Posts de um canal em cada bloco, sem coluna de curtidas (a amostra não pode lê-las)."""
    blocos = [
        ("2024-01-01", n_treino),
        ("2026-02-02", n_cal),
        ("2026-04-02", n_teste),
        ("2026-01-10", n_gap),  # gap (jan/2026): fica fora
        ("2021-01-01", n_antigo),  # antes do Shorts no Brasil: fica fora
    ]
    linhas = []
    for b, (inicio, n) in enumerate(blocos):
        for k in range(n):
            pid = f"{canal}{b}{k:04d}".ljust(11, "x")
            data = pd.Timestamp(inicio, tz="UTC") + pd.Timedelta(hours=k)
            linhas.append((pid, canal, data, pid, "beleza"))
    return pd.DataFrame(
        linhas, columns=["post_id", "channel_id", "published_at", "duplicate_group", "sector"]
    )


def test_alocar_proporcional():
    assert alocar({"treino": 100, "cal": 50, "teste": 50}, 20) == {
        "treino": 10,
        "cal": 5,
        "teste": 5,
    }


def test_alocar_maiores_restos():
    # 20 × (70, 20, 10)/100 = 14, 4, 2
    assert alocar({"treino": 70, "cal": 20, "teste": 10}, 20) == {
        "treino": 14,
        "cal": 4,
        "teste": 2,
    }
    # 20 × (1, 1, 1)/3 = 6,67 cada: os restos empatam e a soma fecha em 20
    assert sum(alocar({"treino": 1000, "cal": 1000, "teste": 1000}, 20).values()) == 20


def test_alocar_canal_pequeno():
    assert alocar({"treino": 4, "cal": 2, "teste": 1}, 20) == {"treino": 4, "cal": 2, "teste": 1}


def test_amostra_so_primeiras_copias(cfg):
    df = _base("UCa", 30, 0, 0)
    df.loc[1:, "duplicate_group"] = df.loc[0, "post_id"]  # todas copiam a primeira
    amostra = sortear_amostra(df, cfg, por_canal=20, seed=0)
    assert amostra["post_id"].tolist() == [df.loc[0, "post_id"]]


def test_amostra_fora_do_gap_e_antes_de_2021(cfg):
    df = _base("UCa", 5, 0, 0, n_gap=50, n_antigo=50)
    amostra = sortear_amostra(df, cfg, por_canal=20, seed=0)
    assert len(amostra) == 5
    assert set(amostra["bloco"]) == {"treino"}


def test_amostra_proporcional_por_canal(cfg):
    df = pd.concat([_base("UCa", 100, 50, 50), _base("UCb", 10, 0, 30)], ignore_index=True)
    amostra = sortear_amostra(df, cfg, por_canal=20, seed=0)
    contagem = amostra.groupby(["channel_id", "bloco"]).size().to_dict()
    assert contagem == {
        ("UCa", "treino"): 10,
        ("UCa", "cal"): 5,
        ("UCa", "teste"): 5,
        ("UCb", "treino"): 5,
        ("UCb", "teste"): 15,
    }


def test_amostra_reprodutivel(cfg):
    df = pd.concat([_base("UCa", 100, 50, 50), _base("UCb", 10, 0, 30)], ignore_index=True)
    a = sortear_amostra(df, cfg, por_canal=20, seed=0)
    b = sortear_amostra(df, cfg, por_canal=20, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert sorted(a["ordem"]) == list(range(len(a)))  # a ordem de processamento é uma permutação
