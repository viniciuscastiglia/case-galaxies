"""Teste da medição da maturação (decisões #11 e #21; aulas 1.7 e 3.5, exercício "quanto custa a
lacuna"): quanto as curtidas ainda crescem depois de cada idade, com os snapshots diários."""

import pandas as pd

from previa.maturacao import FAIXAS_CAUDA, crescimento_por_idade


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


def test_a_cauda_vai_de_meses_a_anos():
    # #21: os Shorts com mais de 180 dias tinham um snapshot só; a recoleta de 20/10 dá o segundo
    snap = pd.DataFrame(
        {
            "video_id": ["velho", "velho", "antigo", "antigo"],
            "age_days": [400.0, 412.0, 1500.0, 1512.0],
            "likes": [1000, 1020, 5000, 5000],
        }
    )
    t = crescimento_por_idade(snap, faixas=FAIXAS_CAUDA, min_intervalo=10)
    assert t.loc["365-730", "mediana"] == 0.02
    assert t.loc["730+", "mediana"] == 0.0  # sem teto: um Short de 2018 cai na última faixa
