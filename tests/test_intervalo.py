"""M6, o núcleo: quantis, CQR e cobertura (aulas 5.2 e 5.3). O caso à mão do Vinicius é o teste."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from previa.experimento import Config
from previa.intervalo import (
    ConfigIntervalo,
    cobertura,
    conter_o_numero,
    em_curtidas,
    escores_cqr,
    faixa_de_posts,
    intervalo,
    ler_config,
    prever_quantis,
    q_chapeu,
    treinar_quantis,
)

# Caso à mão: 9 Shorts maduros da Lumina (escala log1p), alpha = 0,2.
Q10 = np.array([6.0, 6.2, 5.8, 6.0, 6.5, 6.1, 5.9, 6.3, 6.0])
Q90 = np.array([7.0, 7.2, 6.8, 7.0, 7.5, 7.1, 6.9, 7.3, 7.0])
Y = np.array([6.5, 7.5, 5.6, 6.9, 8.1, 6.0, 6.3, 7.0, 7.4])


def test_cobertura_crua_do_caso_a_mao():
    lo_hi = cobertura(Y, Q10, Q90, None)
    assert lo_hi.loc["total", "n"] == 9
    assert lo_hi.loc["total", "cobertura"] == pytest.approx(4 / 9)  # passo 1


def test_os_nove_scores():
    esperado = [-0.5, 0.3, 0.2, -0.1, 0.6, 0.1, -0.4, -0.3, 0.4]  # passo 2
    assert escores_cqr(Q10, Q90, Y) == pytest.approx(esperado)


def test_q_chapeu_do_caso_a_mao():
    escores = escores_cqr(Q10, Q90, Y)
    assert q_chapeu(escores, 0.2) == pytest.approx(0.4)  # passo 3: k = ceil(10 * 0.8) = 8


def test_q_chapeu_com_k_maior_que_n_e_infinito():
    assert q_chapeu(np.array([0.1, 0.2, 0.3]), 0.1) == np.inf  # k = ceil(4 * 0.9) = 4 > 3


def test_intervalo_do_short_b_em_log_e_em_curtidas():
    lo, hi = intervalo(np.array([6.0]), np.array([7.0]), 0.4)  # passo 4
    assert (lo[0], hi[0]) == pytest.approx((5.6, 7.4))
    c_lo, c_hi = em_curtidas(lo, hi)
    assert c_lo[0] == pytest.approx(269, abs=1)
    assert c_hi[0] == pytest.approx(1635, abs=1)
    assert np.expm1(6.5) == pytest.approx(664.1, abs=0.1)  # o número da tool (q50)


def test_em_curtidas_corta_em_zero():
    lo, hi = em_curtidas(np.array([-0.5]), np.array([2.0]))
    assert lo[0] == 0.0 and hi[0] == pytest.approx(np.expm1(2.0))


def test_com_q_chapeu_oito_de_nove_dentro():
    lo, hi = intervalo(Q10, Q90, 0.4)
    assert cobertura(Y, lo, hi, None).loc["total", "cobertura"] == pytest.approx(8 / 9)


def test_cruzamento_ordena_e_conta():
    modelos = {a: _FalsoModelo(v) for a, v in {0.1: 6.6, 0.5: 6.5, 0.9: 7.0}.items()}
    cfg = Config(nome="t", hipotese="t", descricao="t", features=["a"], alvo="absoluto")
    x = pd.DataFrame({"a": [1.0]})
    q, n_cruz = prever_quantis(modelos, x, pd.Series([0.0]), cfg)
    assert q[0].tolist() == [6.5, 6.6, 7.0]
    assert n_cruz == 1


def test_prever_quantis_soma_a_referencia_no_alvo_relativo():
    modelos = {a: _FalsoModelo(v) for a, v in {0.1: -1.0, 0.5: 0.0, 0.9: 1.0}.items()}
    cfg = Config(nome="t", hipotese="t", descricao="t", features=["a"], alvo="relativo")
    q, n_cruz = prever_quantis(modelos, pd.DataFrame({"a": [1.0]}), pd.Series([5.0]), cfg)
    assert q[0].tolist() == [4.0, 5.0, 6.0]
    assert n_cruz == 0


class _FalsoModelo:
    def __init__(self, v):
        self.v = v

    def predict(self, x):
        return np.full(len(x), self.v)


def test_q_chapeu_so_depende_da_calibracao():
    escores_cal = escores_cqr(Q10, Q90, Y)
    qhat = q_chapeu(escores_cal, 0.2)
    # o teste não entra: outro conjunto de teste, mesmo q̂
    y_teste = Y + 3.0
    _ = escores_cqr(Q10, Q90, y_teste)
    assert q_chapeu(escores_cal, 0.2) == qhat


def test_cobertura_de_80_por_cento_em_dados_permutaveis():
    rng = np.random.default_rng(0)
    n = 2000
    mu = rng.normal(6, 1, 2 * n)
    y = mu + rng.normal(0, 1, 2 * n)
    q10, q90 = mu - 0.3, mu + 0.3  # intervalo mal calibrado de propósito
    qhat = q_chapeu(escores_cqr(q10[:n], q90[:n], y[:n]), 0.2)
    lo, hi = intervalo(q10[n:], q90[n:], qhat)
    cob = cobertura(y[n:], lo, hi, None).loc["total", "cobertura"]
    assert cob == pytest.approx(0.80, abs=0.03)


def test_conter_o_numero_so_alarga():
    lo, hi = conter_o_numero(
        np.array([100.0, 10.0]), np.array([200.0, 20.0]), np.array([150.0, 30.0])
    )
    assert lo.tolist() == [100.0, 10.0]
    assert hi.tolist() == [200.0, 30.0]
    lo, hi = conter_o_numero(np.array([100.0]), np.array([200.0]), np.array([50.0]))
    assert (lo[0], hi[0]) == (50.0, 200.0)


def test_cobertura_por_grupo_com_erro_padrao_e_largura():
    y = np.array([1.0, 5.0, 1.0, 1.0])
    lo, hi = np.zeros(4), np.full(4, 2.0)
    g = pd.Series(["a", "a", "b", "b"])
    t = cobertura(y, lo, hi, g)
    assert t.loc["total", "cobertura"] == pytest.approx(0.75)
    assert t.loc["a", "cobertura"] == pytest.approx(0.5)
    assert t.loc["b", "n"] == 2
    assert t.loc["a", "erro_padrao"] == pytest.approx(np.sqrt(0.8 * 0.2 / 2))
    assert t.loc["a", "largura"] == pytest.approx(2.0)


def test_faixa_de_posts():
    cfg_q = ConfigIntervalo(faixas_posts=[10, 50])
    s = faixa_de_posts(pd.Series([3, 10, 50, 51]), cfg_q)
    assert s.tolist() == ["< 10", "10 a 50", "10 a 50", "> 50"]


def test_ler_config_do_repositorio():
    from previa.caminhos import CONFIGS

    cfg_q = ler_config(CONFIGS / "intervalo.yaml")
    assert cfg_q.alphas == [0.1, 0.5, 0.9] and cfg_q.alpha_cqr == 0.2
    assert cfg_q.min_data_in_leaf[0.9] > cfg_q.min_data_in_leaf[0.5]


def _base_sintetica(n=600):
    rng = np.random.default_rng(1)
    d = pd.DataFrame(
        {
            "published_at": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "channel_id": rng.choice(["a", "b", "c"], n),
            "x": rng.normal(size=n),
        }
    )
    d["y_rel"] = d["x"] + rng.normal(0, 0.5, n)
    return d


def test_treinar_quantis_ordena_e_usa_metric_quantile(monkeypatch):
    d = _base_sintetica()
    cfg = Config(
        nome="t",
        hipotese="t",
        descricao="t",
        features=["x"],
        val_dias=100,
        max_arvores=60,
        paciencia=10,
        lgbm={"objective": "huber", "alpha": 1.0, "learning_rate": 0.1, "num_leaves": 7},
    )
    cfg_q = ConfigIntervalo(min_data_in_leaf={0.1: 40, 0.5: 20, 0.9: 40})
    vistos = []
    original = lgb.train

    def espiao(params, *a, **k):
        vistos.append(dict(params))
        return original(params, *a, **k)

    monkeypatch.setattr(lgb, "train", espiao)
    modelos = treinar_quantis(d, d.index, cfg, ["x"], cfg_q)
    assert sorted(modelos) == [0.1, 0.5, 0.9]
    assert all(p["objective"] == "quantile" and p["metric"] == "quantile" for p in vistos)
    assert {p["alpha"] for p in vistos} == {0.1, 0.5, 0.9}
    assert {p["min_data_in_leaf"] for p in vistos if p["alpha"] == 0.9} == {40}
    assert {p["min_data_in_leaf"] for p in vistos if p["alpha"] == 0.5} == {20}
    q, n_cruz = prever_quantis(modelos, d[["x"]], pd.Series(0.0, index=d.index), cfg)
    assert q.shape == (len(d), 3)
    assert (np.diff(q, axis=1) >= 0).all()
    assert n_cruz < len(d) * 0.5  # na maioria, q10 <= q50 <= q90 sem precisar ordenar
