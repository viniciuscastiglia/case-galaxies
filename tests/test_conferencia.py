"""Testes da conferência de 50 classificações de Short (decisão #19; aula 1.5).

A lista sai da tabela `videos` do coletor (sem curtidas) e mistura Shorts da base com verticais
que a regra recusou por poucos segundos: são as bordas onde a regra pode errar.
"""

import pandas as pd
import pytest

from previa.conferencia import concordancia, estrato, sortear_lista

COTAS = {
    "short_antes_ate_60s": 2,
    "short_depois_ate_60s": 3,
    "short_depois_61_a_180s": 3,
    "nao_short_vertical_na_borda": 2,
}


def _videos(linhas):
    """(video_id, published_at, duration_s, vertical, is_short) -> tabela `videos` mínima."""
    v = pd.DataFrame(
        linhas, columns=["video_id", "published_at", "duration_s", "vertical", "is_short"]
    )
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True)
    v["channel_id"] = "UCa"
    return v


def _muitos():
    """8 vídeos de cada estrato e 8 que não entram em nenhum (horizontais)."""
    linhas = []
    for k in range(8):
        linhas += [
            (f"antes{k:06d}", "2023-05-01", 30, True, True),
            (f"dep60{k:06d}", "2025-05-01", 40, True, True),
            (f"dep3m{k:06d}", "2025-05-01", 120, True, True),
            (f"borda{k:06d}", "2023-05-01", 75, True, False),
            (f"horiz{k:06d}", "2025-05-01", 40, False, False),
        ]
    return _videos(linhas)


def test_estratos_da_regra():
    v = _videos(
        [
            ("a0000000000", "2023-05-01", 59, True, True),
            ("b0000000000", "2025-05-01", 61, True, True),  # 60 s + 1 s de folga
            ("c0000000000", "2025-05-01", 120, True, True),
            ("d0000000000", "2023-05-01", 75, True, False),  # antes da mudança, 61 a 91 s
            ("e0000000000", "2025-05-01", 200, True, False),  # depois, 181 a 211 s
            ("f0000000000", "2025-05-01", 250, True, False),  # longe da borda
            ("g0000000000", "2025-05-01", 40, False, False),  # horizontal
        ]
    )
    assert estrato(v).tolist() == [
        "short_antes_ate_60s",
        "short_depois_ate_60s",
        "short_depois_61_a_180s",
        "nao_short_vertical_na_borda",
        "nao_short_vertical_na_borda",
        pd.NA,
        pd.NA,
    ]


def test_lista_respeita_cotas_e_nao_repete():
    lista = sortear_lista(_muitos(), COTAS, seed=0)
    assert lista["estrato"].value_counts().to_dict() == COTAS
    assert lista["post_id"].is_unique
    assert sorted(lista["n"]) == list(range(1, 11))  # ordem sorteada, numerada a partir de 1
    assert (lista["regra"] == "Short").sum() == 8
    assert lista["url"].str.startswith("https://www.youtube.com/shorts/").all()
    assert lista["no_youtube_e_short"].eq("").all()  # coluna para o Vinicius preencher


def test_lista_reprodutivel():
    a = sortear_lista(_muitos(), COTAS, seed=0)
    b = sortear_lista(_muitos(), COTAS, seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_estrato_pequeno_leva_o_que_tem():
    v = _muitos().iloc[:5]  # um vídeo de cada tipo
    lista = sortear_lista(v, COTAS, seed=0)
    assert lista["estrato"].value_counts().to_dict() == {e: 1 for e in COTAS}


def test_concordancia():
    lista = pd.DataFrame(
        {
            "estrato": ["short_antes_ate_60s"] * 3 + ["nao_short_vertical_na_borda"] * 2,
            "regra": ["Short"] * 3 + ["não Short"] * 2,
            "no_youtube_e_short": ["Sim", "não", "", "nao", "s"],
        }
    )
    tab = concordancia(lista)
    assert tab.loc["short_antes_ate_60s", "conferidos"] == 2  # a resposta em branco fica fora
    assert tab.loc["short_antes_ate_60s", "concordam"] == 1
    assert tab.loc["nao_short_vertical_na_borda", "concordam"] == 1
    assert tab.loc["total", "conferidos"] == 4
    assert tab.loc["total", "taxa"] == 0.5


def test_resposta_invalida_levanta():
    lista = pd.DataFrame(
        {"estrato": ["short_antes_ate_60s"], "regra": ["Short"], "no_youtube_e_short": ["talvez"]}
    )
    with pytest.raises(ValueError, match="talvez"):
        concordancia(lista)
