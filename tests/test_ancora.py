"""A âncora da cadeia de cold start (#34; aula 2.1): canal, inscritos, setor."""

import numpy as np
import pandas as pd
import pytest

from previa.ancora import ajustar_ancora, ancora

FIM = pd.Timestamp("2025-06-01", tz="UTC")
QUANDO = pd.Timestamp("2026-06-01T12:00Z")


def _posts(ys, dia0="2026-01-01"):
    datas = pd.date_range(dia0, periods=len(ys), freq="7D", tz="UTC")
    return pd.DataFrame({"published_at": datas, "y": ys})


@pytest.fixture
def treino():
    # 3 canais; y = log1p(curtidas); inscritos tais que log1p(medianas) = 1 + 2 * log1p(insc)
    # pontos exatos: x = 1, 2, 3 -> medianas 3, 5, 7
    linhas = []
    for canal, setor, mediana in [("UC_a", "beleza", 3.0), ("UC_b", "beleza", 5.0),
                                  ("UC_c", "automotivo", 7.0)]:  # fmt: skip
        for i, y in enumerate([mediana - 1, mediana, mediana + 1]):
            linhas.append(
                (canal, setor, pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i), y)
            )
    base = pd.DataFrame(linhas, columns=["channel_id", "sector", "published_at", "y"])
    insc = pd.Series(
        {c: np.expm1(x) for c, x in zip(["UC_a", "UC_b", "UC_c"], [1, 2, 3], strict=True)}
    )
    return base, insc


def test_regressao_com_resposta_conhecida(treino):
    p = ajustar_ancora(*treino, FIM)
    assert p["coef"] == pytest.approx([2.0, 1.0])  # inclinação, intercepto
    assert p["mediana_setor"]["beleza"] == pytest.approx(4.0)  # y: 2,3,4,4,5,6
    assert p["mediana_setor"]["automotivo"] == pytest.approx(7.0)


def test_ajuste_ignora_posts_sem_curtidas(treino):
    base, insc = treino
    oculto = base.iloc[[0]].assign(y=np.nan)
    p = ajustar_ancora(pd.concat([base, oculto]), insc, FIM)
    assert p["coef"] == pytest.approx([2.0, 1.0])


def test_ajuste_sem_canais_suficientes_levanta(treino):
    base, insc = treino
    with pytest.raises(ValueError, match="canais"):
        ajustar_ancora(base[base["channel_id"] == "UC_a"], insc, FIM)


def test_degrau_do_canal(treino):
    p = ajustar_ancora(*treino, FIM)
    posts = _posts([4.0, 6.0, 5.0, 7.0, 3.0])  # 5 maduros em QUANDO; mediana 5
    valor, degrau, n = ancora(posts, 1000, "beleza", QUANDO, p)
    assert (valor, degrau, n) == (pytest.approx(5.0), "mediana_do_canal", 5)


def test_degrau_dos_inscritos_com_dois_posts(treino):
    p = ajustar_ancora(*treino, FIM)
    posts = _posts([4.0, 6.0])
    valor, degrau, n = ancora(posts, int(np.expm1(2.5)), "beleza", QUANDO, p)
    assert degrau == "inscritos" and n == 2
    assert valor == pytest.approx(2.0 * np.log1p(int(np.expm1(2.5))) + 1.0)


def test_degrau_do_setor_sem_inscritos(treino):
    p = ajustar_ancora(*treino, FIM)
    valor, degrau, n = ancora(_posts([]), None, "automotivo", QUANDO, p)
    assert (valor, degrau, n) == (pytest.approx(7.0), "mediana_do_setor", 0)


def test_posts_imaturos_nao_contam(treino):
    p = ajustar_ancora(*treino, FIM)
    posts = _posts([4.0, 6.0, 5.0, 7.0], dia0="2026-04-20")  # só os 2 primeiros têm 30 dias
    _, degrau, n = ancora(posts, 100, "beleza", QUANDO, p)
    assert (degrau, n) == ("inscritos", 2)


def test_curtidas_ocultas_nao_contam_f10(treino):
    p = ajustar_ancora(*treino, FIM)
    posts = _posts([np.nan, None, np.nan, 5.0])  # F10: 3 ocultos, 1 com curtida
    _, degrau, n = ancora(posts, 100, "beleza", QUANDO, p)
    assert (degrau, n) == ("inscritos", 1)
    _, degrau, n = ancora(posts, None, "beleza", QUANDO, p)
    assert (degrau, n) == ("mediana_do_setor", 1)


def test_setor_sem_mediana_levanta(treino):
    p = ajustar_ancora(*treino, FIM)
    with pytest.raises(ValueError, match="setor"):
        ancora(_posts([]), None, "moda", QUANDO, p)


@pytest.mark.parametrize("quando", ["2025-05-10", "2025-07-01"])  # imaturo; depois do fim do treino
def test_ajuste_ignora_imaturos_e_posteriores(treino, quando):
    base, insc = treino
    fora = pd.DataFrame(
        {"channel_id": ["UC_a"] * 5, "sector": "beleza", "y": 20.0,
         "published_at": pd.Timestamp(quando, tz="UTC")}
    )  # fmt: skip
    p = ajustar_ancora(pd.concat([base, fora]), insc, FIM)
    assert p["coef"] == pytest.approx([2.0, 1.0])
    assert p["mediana_setor"]["beleza"] == pytest.approx(4.0)


def test_inscritos_invalidos_caem_no_setor_e_avisam(treino):
    base, insc = treino
    p = ajustar_ancora(base, insc, FIM)
    for ruim in (float("nan"), pd.NA, -5):
        assert ancora(_posts([]), ruim, "beleza", QUANDO, p)[1] == "mediana_do_setor"
    with pytest.warns(UserWarning, match="1 de 3 canais"):
        ajustar_ancora(base, insc.drop("UC_c"), FIM)
