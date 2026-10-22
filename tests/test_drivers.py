"""Testes do `top_drivers` (M8; aulas 5.6 e 6.3, Tool 4): o multiplicador é e^φ, os PCs viram um
fator só, o contexto fica fora, só os blocos aprovados falam, e a soma completa fecha com a
previsão (erro comum 7)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from previa.contribuicoes import contribuicoes
from previa.drivers import carregar_cfg, top_drivers


@pytest.fixture
def modelo_e_x():
    rng = np.random.default_rng(0)
    n = 800
    x = pd.DataFrame(
        {
            "hist_mediana": rng.normal(6, 1, n),
            "setor": rng.choice([0.0, 1.0], n),
            "duracao_s": rng.integers(10, 60, n).astype(float),
            "hora_local": rng.integers(0, 24, n).astype(float),
            "sonda_rosto": rng.normal(size=n),
            "titulo_pc00": rng.normal(size=n),
            "titulo_pc01": rng.normal(size=n),
            "thumb_pc00": rng.normal(size=n),
            "sonda_roda": rng.normal(size=n),
        }
    )
    y = (
        x["hist_mediana"]
        + 0.03 * x["duracao_s"]
        - 0.05 * x["hora_local"]
        + 0.6 * x["sonda_rosto"]
        + 0.4 * x["titulo_pc00"]
        - 0.3 * x["titulo_pc01"]
        + 0.2 * x["thumb_pc00"]
        + 0.3 * x["setor"]
        + 0.5 * x["sonda_roda"]
        + rng.normal(0, 0.2, n)
    )
    params = {"objective": "regression", "verbosity": -1, "seed": 0, "num_leaves": 15}
    return lgb.train(params, lgb.Dataset(x, y), num_boost_round=80), x


CFG = {
    "k": 5,
    "min_abs": 0.0,
    "contexto": ["hist_*", "setor"],
    "blocos": {"significado do título": ["titulo_pc*"], "capa (SigLIP)": ["thumb_pc*"]},
}


def test_multiplicador_e_exp_da_contribuicao(modelo_e_x):
    modelo, x = modelo_e_x
    for d in top_drivers(modelo, x.iloc[[3]], None, CFG):
        assert d["multiplicador"] == pytest.approx(np.exp(d["contribuicao_log"]), abs=0.006)
        assert set(d) == {"feature", "valor", "contribuicao_log", "multiplicador", "direcao"}


def test_pcs_do_titulo_viram_um_driver(modelo_e_x):
    modelo, x = modelo_e_x
    drivers = top_drivers(modelo, x.iloc[[3]], None, CFG | {"k": 20})
    nomes = [d["feature"] for d in drivers]
    assert nomes.count("significado do título") == 1
    assert not any(n.startswith(("titulo_pc", "thumb_pc")) for n in nomes)
    bloco = next(d for d in drivers if d["feature"] == "significado do título")
    assert bloco["valor"] is None  # um bloco não tem valor único (5.6)
    valores, _ = contribuicoes(modelo, x.iloc[[3]])
    soma = valores[["titulo_pc00", "titulo_pc01"]].sum(axis=1).iloc[0]
    assert bloco["contribuicao_log"] == pytest.approx(soma, abs=1e-3)


def test_valor_de_feature_simples_e_o_de_x(modelo_e_x):
    modelo, x = modelo_e_x
    drivers = top_drivers(modelo, x.iloc[[3]], None, CFG | {"k": 20})
    d = next(d for d in drivers if d["feature"] == "duracao_s")
    assert d["valor"] == x["duracao_s"].iloc[3]


def test_contexto_fica_fora(modelo_e_x):
    modelo, x = modelo_e_x
    nomes = [d["feature"] for d in top_drivers(modelo, x.iloc[[3]], None, CFG | {"k": 20})]
    assert "hist_mediana" not in nomes and "setor" not in nomes
    assert "duracao_s" in nomes


def test_so_blocos_aprovados(modelo_e_x):
    modelo, x = modelo_e_x
    aprovados = {"significado do título", "sonda_rosto"}
    nomes = [
        d["feature"] for d in top_drivers(modelo, x.iloc[[3]], None, CFG | {"k": 20}, aprovados)
    ]
    assert set(nomes) <= aprovados and nomes  # duração e hora (não aprovadas) saem


def test_corta_em_k_e_ordena_por_valor_absoluto(modelo_e_x):
    modelo, x = modelo_e_x
    drivers = top_drivers(modelo, x.iloc[[3]], None, CFG | {"k": 2})
    assert len(drivers) == 2
    completo = top_drivers(modelo, x.iloc[[3]], None, CFG | {"k": 20})
    absolutos = [abs(d["contribuicao_log"]) for d in completo]
    assert absolutos == sorted(absolutos, reverse=True)
    assert drivers == completo[:2]


def test_direcao_pelo_sinal(modelo_e_x):
    modelo, x = modelo_e_x
    for i in range(5):
        for d in top_drivers(modelo, x.iloc[[i]], None, CFG | {"k": 20}):
            assert d["direcao"] == ("aumenta" if d["contribuicao_log"] > 0 else "reduz")


def test_soma_completa_fecha_com_a_previsao(modelo_e_x):
    modelo, x = modelo_e_x
    linha = x.iloc[[7]]
    # sem contexto e sem corte: todos os drivers + o contexto + a base = a previsão
    cfg = {"k": 99, "min_abs": 0.0, "contexto": [], "blocos": CFG["blocos"]}
    drivers = top_drivers(modelo, linha, None, cfg)
    _, base = contribuicoes(modelo, linha)
    soma = sum(d["contribuicao_log"] for d in drivers)
    assert soma + base == pytest.approx(modelo.predict(linha)[0], abs=0.01)


def test_contribuicao_desprezivel_sai_antes_do_corte(modelo_e_x):
    modelo, x = modelo_e_x
    linha = x.iloc[[3]]
    todos = top_drivers(modelo, linha, None, CFG | {"k": 20})
    absolutos = sorted(abs(d["contribuicao_log"]) for d in todos)
    corte = (absolutos[0] + absolutos[1]) / 2 if absolutos[0] != absolutos[1] else absolutos[0] / 2
    filtrado = top_drivers(modelo, linha, None, CFG | {"k": 20, "min_abs": corte})
    assert len(filtrado) == len(todos) - 1  # só o menor |φ| saiu
    assert all(abs(d["contribuicao_log"]) >= corte for d in filtrado)
    # a 0,005 de |φ| fica abaixo do corte de 0,01 da config real
    assert carregar_cfg()["min_abs"] == 0.01 and 0.005 < carregar_cfg()["min_abs"]
    # o corte vem antes do k: com k=2 e corte alto, sobram só os que passam
    alto = top_drivers(modelo, linha, None, CFG | {"k": 2, "min_abs": absolutos[-2] + 1e-6})
    assert [abs(d["contribuicao_log"]) for d in alto] == absolutos[-1:]


def test_exige_uma_linha(modelo_e_x):
    modelo, x = modelo_e_x
    with pytest.raises(ValueError, match="uma linha"):
        top_drivers(modelo, x.iloc[:2], None, CFG)


def test_config_real_carrega():
    cfg = carregar_cfg()
    assert cfg["k"] == 5
    assert "setor" in cfg["contexto"] and "titulo_pc*" in cfg["blocos"]["significado do título"]


# Só as sondas validadas contra a rubrica do VLM falam com o próprio nome, e só no setor em que
# passaram (aula 4.4, erro comum 5); as outras somam no bloco da capa, e a conta não muda.
CFG_SONDAS = CFG | {
    "k": 20,
    "sondas_validadas": {"beleza": ["sonda_rosto"], "automotivo": []},
    "bloco_das_sondas_nao_validadas": "capa (SigLIP)",
}


def test_sonda_nao_validada_soma_no_bloco_da_capa(modelo_e_x):
    modelo, x = modelo_e_x
    drivers = top_drivers(modelo, x.iloc[[3]], None, CFG_SONDAS, setor="automotivo")
    nomes = [d["feature"] for d in drivers]
    assert "sonda_rosto" not in nomes and "sonda_roda" not in nomes and "capa (SigLIP)" in nomes
    valores, _ = contribuicoes(modelo, x.iloc[[3]])
    soma = valores[["thumb_pc00", "sonda_rosto", "sonda_roda"]].sum(axis=1).iloc[0]
    capa = next(d for d in drivers if d["feature"] == "capa (SigLIP)")
    assert capa["contribuicao_log"] == pytest.approx(soma, abs=1e-3)


def test_sonda_validada_no_setor_fala_com_o_proprio_nome(modelo_e_x):
    modelo, x = modelo_e_x
    drivers = top_drivers(modelo, x.iloc[[3]], None, CFG_SONDAS, setor="beleza")
    nomes = [d["feature"] for d in drivers]
    assert "sonda_rosto" in nomes and "sonda_roda" not in nomes


def test_a_soma_nao_muda_ao_juntar_as_sondas(modelo_e_x):
    modelo, x = modelo_e_x
    cfg = CFG_SONDAS | {"min_abs": 0.0}
    antes = sum(d["contribuicao_log"] for d in top_drivers(modelo, x.iloc[[3]], None, cfg))
    depois = sum(
        d["contribuicao_log"]
        for d in top_drivers(modelo, x.iloc[[3]], None, cfg, setor="automotivo")
    )
    assert depois == pytest.approx(antes, abs=2e-3)


def test_sem_setor_as_sondas_seguem_individuais(modelo_e_x):
    modelo, x = modelo_e_x
    nomes = [d["feature"] for d in top_drivers(modelo, x.iloc[[3]], None, CFG_SONDAS)]
    assert "sonda_roda" in nomes


def test_pacote_sem_sondas_nao_quebra(modelo_e_x):
    modelo, x = modelo_e_x
    sem = x.drop(columns=["sonda_rosto", "sonda_roda"])
    params = {"objective": "regression", "verbosity": -1, "seed": 0}
    m = lgb.train(params, lgb.Dataset(sem, x["hora_local"]), num_boost_round=10)
    drivers = top_drivers(m, sem.iloc[[0]], None, CFG_SONDAS, setor="beleza")
    assert not any(d["feature"].startswith("sonda_") for d in drivers)
