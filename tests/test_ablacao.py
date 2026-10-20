"""Testes da ablação por blocos (aulas 2.6 e 4.8), na base sintética do runner."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from test_experimento import AVALIACAO, CONFIG, _base

from previa.ablacao import BASELINE, ablar, desenho, separar_blocos
from previa.experimento import ler_config, rodar
from previa.splits import ler_avaliacao

BLOCOS = {
    "historico": {"features": ["hist_*", "tendencia", "posts_30d"]},
    "calendario": {"features": ["hora_local"]},
    "metadados": {"features": ["duracao_s", "setor"]},
}
SO_O_BLOCO = ("nome", "descricao", "features", "pca", "extras", "ruido")


@pytest.fixture
def cfg_av(tmp_path):
    (tmp_path / "avaliacao.yaml").write_text(AVALIACAO)
    return ler_avaliacao(tmp_path / "avaliacao.yaml")


@pytest.fixture
def completo(tmp_path):
    """O GBM pequeno do runner, com as 7 features divididas nos 3 blocos de `BLOCOS`."""
    (tmp_path / "exp.yaml").write_text(CONFIG)
    return ler_config(tmp_path / "exp.yaml")


def test_variantes_mudam_so_os_blocos(completo):
    # 4.8, erro comum 2: mesmo split, hiperparâmetros e seed; só as colunas mudam
    configs, _ = desenho(completo, BLOCOS)
    fixo = {k: v for k, v in asdict(completo).items() if k not in SO_O_BLOCO}
    for variante in configs.values():
        assert {k: v for k, v in asdict(variante).items() if k not in SO_O_BLOCO} == fixo


def test_escada_acrescenta_um_bloco_por_vez_e_termina_no_completo(completo):
    configs, comparacoes = desenho(completo, BLOCOS)
    escada = [(a, b) for d, _, a, b in comparacoes if d == "cumulativa"]
    assert [bl for d, bl, _, _ in comparacoes if d == "cumulativa"] == list(BLOCOS)
    assert escada[0][0] == BASELINE  # o primeiro degrau é medido contra a mediana madura
    assert [b for _, b in escada][-1] == "completo"
    assert configs["completo"].features == completo.features
    assert configs[escada[0][1]].features == ["hist_n", "hist_mediana", "tendencia", "posts_30d"]
    for (a, b), (_, seguinte) in zip(escada[1:], escada[2:], strict=False):
        assert set(configs[a].features) < set(configs[b].features) < set(configs[seguinte].features)


def test_todos_menos_um_tira_so_o_bloco_e_compara_com_o_completo(completo):
    configs, comparacoes = desenho(completo, BLOCOS)
    menos_um = {a: b for d, _, a, b in comparacoes if d == "todos menos um"}
    assert set(menos_um.values()) == {"completo"}
    assert configs["sem-calendario"].features == [f for f in completo.features if f != "hora_local"]
    # sem o último bloco é o penúltimo degrau da escada: roda uma vez só
    assert "sem-metadados" not in configs and len(menos_um) == len(BLOCOS)


def test_controle_negativo_acrescenta_ruido_ao_completo(completo):
    configs, comparacoes = desenho(completo, BLOCOS, ruido=2)
    ((_, bloco, a, b),) = [c for c in comparacoes if c[0] == "controle negativo"]
    assert bloco == "ruido"
    assert a == "completo" and configs[b].ruido == 2
    assert configs[b].features == completo.features + ["ruido_1", "ruido_2"]


def test_blocos_mal_definidos_levantam(completo):
    with pytest.raises(ValueError, match="txt_"):  # padrão que não acha nada no completo
        separar_blocos(completo, {**BLOCOS, "texto": {"features": ["txt_*"]}})
    with pytest.raises(ValueError, match="hora_local"):  # entrada do completo fora dos blocos
        separar_blocos(completo, {k: v for k, v in BLOCOS.items() if k != "calendario"})
    with pytest.raises(ValueError, match="setor"):  # a mesma entrada em dois blocos
        separar_blocos(completo, {**BLOCOS, "outro": {"features": ["setor"]}})
    with pytest.raises(ValueError, match="vazio"):  # o degrau compararia uma variante com ela mesma
        separar_blocos(completo, {**BLOCOS, "nada": {}})
    with pytest.raises(ValueError, match="pca"):  # PCA que o completo não tem
        separar_blocos(completo, {**BLOCOS, "texto": {"pca": ["titulo"]}})


def test_ruido_entra_como_colunas_sem_relacao_com_o_alvo(cfg_av, completo):
    base, dur = _base()
    completo.features = completo.features + ["ruido_1", "ruido_2"]
    completo.ruido = 2
    r1 = rodar(completo, base, dur, cfg_av, n_boot=50, seed=0)
    r2 = rodar(completo, base, dur, cfg_av, n_boot=50, seed=0)
    assert r1["walk_forward"]["n_features"] == 9
    pd.testing.assert_series_equal(r1["walk_forward"]["previsoes"], r2["walk_forward"]["previsoes"])


def test_ablar_ponta_a_ponta(cfg_av, completo):
    base, dur = _base()
    tabela, configs, resultados = ablar(completo, BLOCOS, base, dur, cfg_av, n_boot=100, seed=0)
    assert set(resultados) == set(configs)
    assert set(tabela["avaliacao"]) == {"walk_forward", "canal_novo"}
    # 3 degraus, 3 "menos um" e 1 controle, em cada avaliação
    assert len(tabela) == 2 * 7
    assert {"bloco", "ganho", "ic_ganho_lo", "dif_mae_log", "ic_dif_mae_hi", "n_features_b"} <= set(
        tabela
    )
    # na base sintética, a duração explica parte das curtidas: tirar os metadados faz falta
    sem_meta = tabela[
        (tabela["a"] == "escada-historico+calendario")
        & (tabela["b"] == "completo")
        & (tabela["avaliacao"] == "walk_forward")
    ]
    assert len(sem_meta) == 2  # a mesma comparação na escada e no "todos menos um"
    assert (sem_meta["ic_ganho_lo"] > 0).all()
    assert np.isfinite(tabela[["ganho", "dif_mae_log"]].to_numpy()).all()
