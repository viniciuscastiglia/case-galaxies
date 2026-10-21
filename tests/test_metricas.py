"""Testes do metricas.py (aula 2.4).

Os 6 primeiros são os da aula, literais. Os de IC e peso igual cobrem os auxiliares da leitura
por canal (bootstrap de canais, erro comum 4; peso dos canais dominantes, decisão #6).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, spearmanr

from previa.metricas import (
    cobertura,
    f1_faixas,
    f1_faixas_canal,
    faixa_do_canal,
    ic_bootstrap_canal,
    ic_dif_mae_pareado,
    ic_ganho_pareado,
    ic_par_a_par,
    ic_par_a_par_peso_posts,
    ic_por_canal,
    mae_log,
    mape,
    pairwise_accuracy,
    pairwise_janela,
    par_a_par_peso_igual,
    par_a_par_peso_posts,
    por_canal,
    spearman_global,
    spearman_intra_canal,
)

rng = np.random.default_rng(0)


def test_mae_e_mape():
    assert mae_log([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)
    assert mape([0, 10], [5, 12]) == pytest.approx(20.0)  # o y = 0 é ignorado


def test_spearman_bate_com_scipy_e_constante_vale_zero():
    y, p = rng.normal(size=50), rng.normal(size=50)
    assert spearman_global(y, p) == pytest.approx(spearmanr(y, p).statistic)
    assert spearman_global(y, np.ones(50)) == 0.0


def test_modelo_que_so_conhece_o_canal():
    canal = np.repeat(["grande", "pequeno"], 20)
    nivel = np.where(canal == "grande", 9.0, 4.0)
    y = nivel + rng.normal(0, 0.5, 40)
    assert spearman_global(y, nivel) > 0.8  # parece ótimo...
    assert spearman_intra_canal(y, nivel, canal) == 0.0  # ...mas não ordena nada
    assert pairwise_accuracy(y, nivel, canal) == 0.5


def test_pairwise_e_kendall_sem_empates():
    y, p = rng.normal(size=30), rng.normal(size=30)
    tau = kendalltau(y, p).statistic
    assert pairwise_accuracy(y, p) == pytest.approx((tau + 1) / 2)


def test_pairwise_intra_ignora_pares_entre_canais():
    y = np.array([1.0, 2.0, 10.0, 11.0])
    p = np.array([5.0, 6.0, 0.0, 1.0])  # erra a ordem ENTRE canais, acerta dentro
    canal = np.array(["a", "a", "b", "b"])
    assert pairwise_accuracy(y, p, canal) == 1.0
    assert pairwise_accuracy(y, p) < 1.0


def test_f1_faixas_e_cobertura():
    r = np.array([-1.0, 0.0, 1.0, 2.0])
    assert f1_faixas(r, r) == 1.0
    assert cobertura([1, 2, 3, 4], [0, 0, 0, 5], [5, 5, 5, 6]) == 0.75


# --- faixas pelos quantis do canal (decisão #7, aula 1.6): p25, p75 e p95 do histórico maduro ---

Q = (3.25, 7.75, 9.55)  # o caso à mão: y = 1..10, interpolação linear


def test_faixa_do_canal_bordas():
    v = [3.0, 3.25, 7.75, 8.0, 9.55, 9.6]
    faixas = faixa_do_canal(v, *Q)
    assert faixas.tolist() == ["abaixo", "na média", "na média", "acima", "acima", "viral"]
    assert isinstance(faixas, np.ndarray)


def test_faixa_do_canal_cada_linha_com_os_quantis_do_seu_canal():
    # o mesmo y = 5 é "abaixo" num canal que costuma passar de 6 e "viral" num que fica em 2
    faixas = faixa_do_canal([5.0, 5.0], [6.0, 1.0], [8.0, 2.0], [9.0, 3.0])
    assert faixas.tolist() == ["abaixo", "viral"]


def test_faixa_do_canal_sem_historico_e_nunca_abaixo():
    nan = np.nan
    v = [1.0, 5.0, 9.9, 5.0]
    p25, p75, p95 = [nan, 3.25, 3.25, 3.25], [nan, 7.75, nan, 7.75], [nan, 9.55, 9.55, nan]
    faixas = faixa_do_canal(v, p25, p75, p95)  # qualquer quantil NaN basta
    assert faixas.tolist() == ["sem histórico", "na média", "sem histórico", "sem histórico"]


def test_faixa_do_canal_recusa_valor_nan_com_historico():
    # NaN comparado a qualquer corte vira "viral" em silêncio: tem que falhar alto
    with pytest.raises(ValueError, match="NaN"):
        faixa_do_canal([np.nan], *Q)
    assert faixa_do_canal([np.nan], np.nan, np.nan, np.nan).tolist() == ["sem histórico"]


def test_f1_faixas_canal_previsao_perfeita_e_erro_de_uma_faixa():
    y = np.array([3.0, 5.0, 8.0, 9.6])  # uma de cada faixa, com os quantis Q
    q = [np.full(4, c) for c in Q]
    assert f1_faixas_canal(y, y, *q) == 1.0
    # erra só o viral (previsto "acima"): abaixo e na média perfeitas; acima 2/3 (precisão 1/2,
    # revocação 1); viral 0 -> (1 + 1 + 2/3 + 0) / 4
    p = np.array([3.0, 5.0, 8.0, 9.0])
    assert f1_faixas_canal(y, p, *q) == pytest.approx((1 + 1 + 2 / 3 + 0) / 4)


def test_faixa_sem_historico_fica_fora_do_f1():
    y = np.array([3.0, 5.0, 8.0, 9.6])
    p = np.array([3.0, 5.0, 8.0, 9.0])
    q = [np.full(4, c) for c in Q]
    base = f1_faixas_canal(y, p, *q)
    # duas linhas sem histórico (quantis NaN), com y e p que dariam "abaixo" se contassem
    y2, p2 = np.append(y, [0.0, 0.0]), np.append(p, [0.0, 1.0])
    q2 = [np.append(c, [np.nan, np.nan]) for c in q]
    assert f1_faixas_canal(y2, p2, *q2) == base
    # e o F1 só com as linhas sem histórico não existe
    assert np.isnan(f1_faixas_canal([0.0, 1.0], [0.0, 1.0], *[[np.nan, np.nan]] * 3))


def test_f1_faixas_canal_recusa_nan_nas_linhas_com_historico():
    q = [np.full(2, c) for c in Q]
    with pytest.raises(ValueError, match="NaN"):
        f1_faixas_canal([1.0, np.nan], [1.0, 1.0], *q)
    with pytest.raises(ValueError, match="NaN"):
        f1_faixas_canal([1.0, 2.0], [1.0, np.nan], *q)


# --- auxiliares da leitura por canal (Claude) ---


def test_ic_de_modelo_que_empata_tudo_e_meio():
    canal = np.repeat(["a", "b", "c"], 10)
    y = rng.normal(size=30)
    tab = por_canal(y, np.zeros(30), canal)
    assert list(ic_par_a_par(tab)) == [0.5, 0.5]
    _, ic_pw = ic_bootstrap_canal(tab)
    assert list(ic_pw) == [0.5, 0.5]


def test_ic_par_a_par_um_canal():
    tab = pd.DataFrame({"acertos": [3.0], "pares": [4]}, index=["a"])
    assert list(ic_par_a_par(tab)) == [0.75, 0.75]


def test_peso_igual_contra_ponderado():
    tab = pd.DataFrame({"acertos": [9.0, 0.0, 0.0], "pares": [10, 2, 0]}, index=["a", "b", "c"])
    assert par_a_par_peso_igual(tab) == pytest.approx(0.45)
    assert tab["acertos"].sum() / tab["pares"].sum() == pytest.approx(0.75)


def test_peso_posts_a_manchete_da_6():
    # "a" acerta 90% em 100 pares, "b" erra tudo em 10: por par dá 90/110; por canal, 0,45
    tab = pd.DataFrame({"acertos": [90.0, 0.0], "pares": [100, 10]}, index=["a", "b"])
    assert par_a_par_peso_posts(tab, pd.Series({"a": 20, "b": 20})) == pytest.approx(0.45)
    assert par_a_par_peso_posts(tab, pd.Series({"a": 30, "b": 10})) == pytest.approx(0.675)


def test_peso_posts_ignora_canal_sem_par():
    tab = pd.DataFrame({"acertos": [3.0, 0.0], "pares": [4, 0]}, index=["a", "b"])
    assert par_a_par_peso_posts(tab, pd.Series({"a": 5, "b": 500})) == pytest.approx(0.75)


def test_ic_peso_posts():
    meio = pd.DataFrame({"acertos": [5.0, 1.0, 20.0], "pares": [10, 2, 40]}, index=["a", "b", "c"])
    n = pd.Series({"a": 8, "b": 3, "c": 30})
    assert list(ic_par_a_par_peso_posts(meio, n)) == [0.5, 0.5]  # empata tudo: IC sem largura
    um = pd.DataFrame({"acertos": [3.0], "pares": [4]}, index=["a"])
    assert list(ic_par_a_par_peso_posts(um, pd.Series({"a": 7}))) == [0.75, 0.75]


def test_ganho_pareado_da_5():
    base = pd.DataFrame({"acertos": [5.0, 1.0, 20.0], "pares": [10, 2, 40]}, index=["a", "b", "c"])
    n = pd.Series({"a": 8, "b": 3, "c": 30})
    assert list(ic_ganho_pareado(base, base, n)) == [0.0, 0.0]  # modelo igual ao baseline
    modelo = base.assign(acertos=[6.0, 1.0, 24.0])  # +10 pontos em "a" e em "c", 0 em "b"
    ganho = par_a_par_peso_posts(modelo, n) - par_a_par_peso_posts(base, n)
    assert ganho == pytest.approx((8 * 0.1 + 30 * 0.1) / 41)
    lo, hi = ic_ganho_pareado(modelo, base, n)
    assert 0 <= lo <= ganho <= hi <= 0.1 + 1e-12  # 3,7% das reamostras só sorteiam "b" (ganho 0)


# --- M2: casos pequenos com a resposta conferida à mão, par a par, no comentário ao lado ---


def _posts(linhas):
    """(canal, data UTC, y, p) -> DataFrame com as colunas da base (published_at com fuso)."""
    d = pd.DataFrame(linhas, columns=["channel_id", "published_at", "y", "p"])
    d["published_at"] = pd.to_datetime(d["published_at"], utc=True)
    return d


def test_janela_caso_a_mao_de_19_10():
    # A-B a 9 dias (o modelo erra), B-C a 10 dias (acerta), A-C a 19 dias (fora): 1 em 2
    d = _posts(
        [
            ("UCa", "2025-10-01", np.log1p(500), np.log1p(400)),
            ("UCa", "2025-10-10", np.log1p(300), np.log1p(450)),
            ("UCa", "2025-10-20", np.log1p(800), np.log1p(600)),
        ]
    )
    tab = pairwise_janela(d, "p")
    assert tab.loc["UCa", "pares"] == 2
    assert tab.loc["UCa", "acertos"] == 1.0


def test_janela_ignora_pares_entre_canais():
    d = _posts(
        [
            ("UCa", "2025-03-01", 1.0, 5.0),
            ("UCa", "2025-03-02", 2.0, 6.0),  # par de UCa: acerta
            ("UCb", "2025-03-01", 10.0, 0.0),
            ("UCb", "2025-03-03", 11.0, 1.0),  # par de UCb: acerta
        ]
    )  # entre canais, a previsão erra tudo; esses pares não contam
    tab = pairwise_janela(d, "p")
    assert list(tab.index) == ["UCa", "UCb"]
    assert tab["pares"].sum() == 2
    assert tab["acertos"].sum() == 2.0


def test_janela_empate_na_previsao_vale_meio_e_empate_em_y_fica_fora():
    d = _posts(
        [
            ("UCa", "2025-03-01", 1.0, 3.0),
            ("UCa", "2025-03-02", 2.0, 3.0),  # com o 1º: y diferente, previsão empatada -> 0,5
            ("UCa", "2025-03-03", 2.0, 9.0),  # com o 1º: acerta -> 1; com o 2º: y empatado -> fora
        ]
    )
    tab = pairwise_janela(d, "p")
    assert tab.loc["UCa", "pares"] == 2
    assert tab.loc["UCa", "acertos"] == 1.5


def test_janela_14_dias_entra_15_nao():
    d = _posts(
        [
            ("UCa", "2025-03-01T10:00:00Z", 1.0, 1.0),
            ("UCa", "2025-03-15T10:00:00Z", 2.0, 2.0),  # 14 dias do 1º: entra, e acerta
            ("UCa", "2025-03-30T10:00:00Z", 3.0, 0.0),  # 15 dias do 2º e 29 do 1º: fora
        ]
    )
    tab = pairwise_janela(d, "p")
    assert tab.loc["UCa", "pares"] == 1
    assert tab.loc["UCa", "acertos"] == 1.0


def test_janela_enorme_bate_com_pairwise_accuracy():
    gerador = np.random.default_rng(1)
    canal = np.repeat(["UCa", "UCb", "UCc"], 15)
    datas = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        gerador.integers(0, 300, 45), "D"
    )
    d = pd.DataFrame(
        {
            "channel_id": canal,
            "published_at": datas,
            "y": gerador.normal(size=45).round(1),  # arredondado: há empates em y
            "p": gerador.normal(size=45).round(1),  # e na previsão
        }
    )
    tab = pairwise_janela(d, "p", dias=100_000)  # janela maior que a base: todos os pares
    assert tab["acertos"].sum() / tab["pares"].sum() == pytest.approx(
        pairwise_accuracy(d["y"], d["p"], d["channel_id"])
    )
    esperado = por_canal(d["y"], d["p"], d["channel_id"])["pares"]
    assert tab["pares"].to_dict() == esperado.to_dict()


def test_janela_canal_sem_par():
    d = _posts(
        [
            ("UCa", "2025-03-01", 1.0, 1.0),  # sozinho: nenhum par
            ("UCb", "2025-03-01", 1.0, 1.0),
            ("UCb", "2025-03-02", 2.0, 2.0),
        ]
    )
    tab = pairwise_janela(d, "p")
    assert tab.loc["UCa", "pares"] == 0
    assert tab.loc["UCb", "pares"] == 1


def test_janela_recusa_nan():
    d = _posts([("UCa", "2025-03-01", 1.0, 1.0), ("UCa", "2025-03-02", 2.0, np.nan)])
    with pytest.raises(ValueError, match="NaN"):
        pairwise_janela(d, "p")


def test_ic_com_reamostra_sem_pares_nao_vira_nan():
    tab = pd.DataFrame({"acertos": [3.0] + [0.0] * 29, "pares": [4] + [0] * 29})
    n = pd.Series(1, index=tab.index)
    assert np.isfinite(ic_par_a_par(tab)).all()
    assert np.isfinite(ic_par_a_par_peso_posts(tab, n)).all()


# --- régua dos blocos (#27): o erro em curtidas, pareado por canal ---


def test_dif_mae_pareada_caso_pequeno():
    # canal X: A erra 1 e 0, B acerta os dois; canal Y: A erra 1, B erra 0,5.
    # MAE de A = 2/3, de B = 0,5/3: B - A = -0,5. Qualquer reamostra de canais dá -0,5 também
    # (X sozinho: -2/4; Y sozinho: -0,5/1), então o IC é o ponto.
    y = [1.0, 2.0, 0.0]
    pa, pb = [2.0, 2.0, 1.0], [1.0, 2.0, 0.5]
    lo, hi = ic_dif_mae_pareado(y, pa, pb, ["X", "X", "Y"], n_boot=200, seed=0)
    assert lo == pytest.approx(-0.5) and hi == pytest.approx(-0.5)


def test_dif_mae_pareada_modelos_iguais_e_zero():
    y = rng.normal(size=60)
    p = y + rng.normal(size=60)
    canal = np.repeat(["a", "b", "c"], 20)
    lo, hi = ic_dif_mae_pareado(y, p, p, canal, n_boot=200, seed=0)
    assert lo == 0.0 and hi == 0.0


def test_dif_mae_pareada_reamostra_canais():
    # B melhora muito o canal a e piora o b: com 2 canais, o IC cobre os dois lados do zero
    y = np.zeros(20)
    pa = np.r_[np.full(10, 1.0), np.full(10, 0.1)]
    pb = np.r_[np.full(10, 0.1), np.full(10, 0.8)]
    lo, hi = ic_dif_mae_pareado(y, pa, pb, np.repeat(["a", "b"], 10), n_boot=500, seed=0)
    assert lo == pytest.approx(-0.9) and hi == pytest.approx(0.7) and lo < 0 < hi


# --- painel de métricas (TAPI 6.1; aula 2.4): o IC de qualquer métrica, reamostrando canais ---


def test_ic_por_canal_um_canal_so():
    # com um canal só, toda reamostra é a base inteira: o IC é o próprio valor nas duas pontas
    t = pd.DataFrame({"channel_id": "a", "y": [1.0, 2.0, 6.0]})
    lo, hi = ic_por_canal(lambda g: float(g["y"].mean()), t, n_boot=50, seed=0)
    assert lo == pytest.approx(3.0) and hi == pytest.approx(3.0)


def test_ic_por_canal_reamostra_canais_e_nao_posts():
    # a tem 1 post e b tem 3: sorteando canais, o nº de linhas da reamostra é 2 (aa), 4 (ab/ba) ou
    # 6 (bb); sorteando posts seria sempre 4. Com 2000 reamostras, os extremos aparecem.
    t = pd.DataFrame({"channel_id": ["a", "b", "b", "b"], "y": [0.0, 1.0, 1.0, 1.0]})
    lo, hi = ic_por_canal(lambda g: float(len(g)), t, n_boot=2000, seed=0)
    assert (lo, hi) == (2.0, 6.0)


def test_ic_por_canal_ignora_nan_e_nao_quebra_se_tudo_for_nan():
    t = pd.DataFrame({"channel_id": ["a", "a", "b", "b"], "y": [0.0, 1.0, 2.0, 3.0]})

    def media_se_tem_b(g):  # a reamostra só com o canal a não tem a métrica: NaN fica de fora
        return float(g["y"].mean()) if (g["channel_id"] == "b").any() else np.nan

    lo, hi = ic_por_canal(media_se_tem_b, t, n_boot=200, seed=0)
    assert lo == 1.5 and hi == 2.5  # ab dá 1,5 e bb dá 2,5; aa (NaN) não entra, não vira NaN
    lo, hi = ic_por_canal(lambda g: np.nan, t, n_boot=20, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_ic_por_canal_usa_os_mesmos_canais_dos_outros_ics():
    # a mesma matriz de sorteios do ic_bootstrap_canal: com a mesma seed, os mesmos canais
    canal = np.repeat(["a", "b", "c", "d"], 6)
    y, p = rng.normal(size=24), rng.normal(size=24)
    t = pd.DataFrame({"channel_id": canal, "y": y, "p": p})
    tab = por_canal(y, p, canal)
    esperado, _ = ic_bootstrap_canal(tab, n_boot=300, seed=7)
    lo, hi = ic_por_canal(
        lambda g: spearman_intra_canal(g["y"], g["p"], g["channel_id"]), t, n_boot=300, seed=7
    )
    np.testing.assert_allclose([lo, hi], esperado, rtol=1e-9)


def test_ic_por_canal_recusa_tabela_vazia_ou_canal_nan():
    with pytest.raises(ValueError, match="vazia"):
        ic_por_canal(len, pd.DataFrame({"channel_id": []}), n_boot=10)
    with pytest.raises(ValueError, match="channel_id"):
        ic_por_canal(len, pd.DataFrame({"channel_id": ["a", None]}), n_boot=10)
