"""Testes das contribuições SHAP (Fase 5, encanamento; aula 5.6): a conta fecha (base + soma =
previsão, erro comum 7) e os componentes de PCA viram um bloco (erro comum 5). O fundo (#16) e o
`top_drivers` (M8) ficam com o Vinicius."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from previa.contribuicoes import agrupar_blocos, contribuicoes


@pytest.fixture
def modelo_e_x():
    rng = np.random.default_rng(0)
    n = 600
    x = pd.DataFrame(
        {
            "hist_mediana": rng.normal(6, 1, n),
            "duracao_s": rng.integers(10, 60, n).astype(float),
            "setor": rng.choice([0.0, 1.0], n),  # código 0/1, como no runner
            "titulo_pc00": rng.normal(size=n),
            "titulo_pc01": rng.normal(size=n),
        }
    )
    y = x["hist_mediana"] + 0.02 * x["duracao_s"] + 0.3 * x["titulo_pc00"] + rng.normal(0, 0.3, n)
    params = {"objective": "regression", "verbosity": -1, "seed": 0, "num_leaves": 7}
    return lgb.train(params, lgb.Dataset(x, y), num_boost_round=50), x


def test_a_conta_fecha_sem_fundo(modelo_e_x):
    modelo, x = modelo_e_x
    valores, base = contribuicoes(modelo, x.iloc[:20])
    np.testing.assert_allclose(valores.sum(axis=1) + base, modelo.predict(x.iloc[:20]), atol=1e-6)
    assert list(valores.columns) == list(x.columns)


def test_a_conta_fecha_com_fundo_do_canal(modelo_e_x):
    modelo, x = modelo_e_x
    fundo = x.iloc[100:200]
    valores, base = contribuicoes(modelo, x.iloc[:5], fundo=fundo)
    np.testing.assert_allclose(valores.sum(axis=1) + base, modelo.predict(x.iloc[:5]), atol=1e-4)
    # uma feature igual em toda a referência e no Short recebe zero (jogador nulo, 5.6)
    fixo = x.iloc[:5].assign(hist_mediana=6.0)
    v, _ = contribuicoes(modelo, fixo, fundo=fundo.assign(hist_mediana=6.0))
    np.testing.assert_allclose(v["hist_mediana"], 0.0, atol=1e-9)


def test_blocos_somam_os_componentes(modelo_e_x):
    modelo, x = modelo_e_x
    valores, _ = contribuicoes(modelo, x.iloc[:3])
    agrupado = agrupar_blocos(valores, {"título (embedding)": ["titulo_pc00", "titulo_pc01"]})
    assert list(agrupado.columns) == ["título (embedding)", "hist_mediana", "duracao_s", "setor"]
    np.testing.assert_allclose(agrupado.sum(axis=1), valores.sum(axis=1))
