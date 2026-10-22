"""A regra de seleção da demo (#18): escrita antes de rodar, para não parecer escolhida a dedo."""

import pandas as pd

from previa.demo import selecionar


def test_o_mais_recente_de_cada_canal_e_o_par_dos_dois_mais_recentes():
    g = pd.DataFrame({
        "handle": ["@a", "@a", "@a", "@b", "@b"],
        "video_id": ["a1", "a2", "a3", "b1", "b2"],
        "published_at": pd.to_datetime(
            ["2026-08-01", "2026-08-10", "2026-08-05", "2026-07-01", "2026-07-20"], utc=True
        ),
    })  # fmt: skip
    recentes, pares = selecionar(g)
    assert list(recentes["video_id"]) == ["a2", "b2"]
    assert [(a["video_id"], b["video_id"]) for a, b in pares] == [("a3", "a2"), ("b1", "b2")]


def test_canal_com_um_short_so_nao_forma_par():
    g = pd.DataFrame({
        "handle": ["@a"],
        "video_id": ["a1"],
        "published_at": pd.to_datetime(["2026-08-01"], utc=True),
    })  # fmt: skip
    recentes, pares = selecionar(g)
    assert list(recentes["video_id"]) == ["a1"] and pares == []
