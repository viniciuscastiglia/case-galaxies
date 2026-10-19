"""Teste da medição da maturação (decisões #11 e #21; aulas 1.7 e 3.5, exercício "quanto custa a
lacuna"): quanto as curtidas ainda crescem depois de cada idade, com os snapshots diários."""

import pandas as pd

from previa.maturacao import crescimento_por_idade


def test_crescimento_entre_o_primeiro_e_o_ultimo_snapshot():
    snap = pd.DataFrame(
        {
            "video_id": ["a", "a", "a", "b", "b", "c"],
            "age_days": [2.0, 5.0, 16.0, 31.0, 45.0, 10.0],
            "likes": [100, 150, 200, 1000, 1010, 50],
        }
    )
    t = crescimento_por_idade(snap, faixas=[0, 7, 30, 60], min_intervalo=10)
    # "a": de 2 a 16 dias, +100%; "b": de 31 a 45 dias, +1%; "c" tem um snapshot só
    assert t.loc["0-7", "videos"] == 1 and t.loc["0-7", "mediana"] == 1.0
    assert t.loc["30-60", "mediana"] == 0.01
    assert "7-30" not in t.index
