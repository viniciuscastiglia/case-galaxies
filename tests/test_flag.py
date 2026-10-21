"""O flag de baixa confiança (5.4; decisão #14): uma regra por teste, limiares da calibração."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from previa.flag import avaliar_flag, carregar_cfg, confidence_flag, limiares_da_calibracao
from previa.schemas import ConfidenceFlag

CFG = carregar_cfg()
ULTIMO = "2026-06-01"
DENTRO = datetime(2026, 7, 1, tzinfo=UTC)


@pytest.fixture
def limiares():
    larguras = pd.Series(np.linspace(0.0, 1.0, 101))  # p80 = 0,8
    dists = pd.Series(np.linspace(0.0, 10.0, 101))  # p99 = 9,9
    return limiares_da_calibracao(larguras, dists, ULTIMO, CFG)


def _normal(limiares, **troca):
    args = dict(
        largura_log=0.5,
        posts_maduros=40,
        extracao_falhou=[],
        dist_knn=2.0,
        data_planejada=DENTRO,
    )
    args.update(troca)
    return confidence_flag(limiares=limiares, **args)


def _codigos(flag):
    return [m.codigo for m in flag.motivos]


def test_limiares_saem_da_calibracao(limiares):
    assert limiares["largura_log"] == pytest.approx(0.8)
    assert limiares["dist_knn"] == pytest.approx(9.9)
    assert limiares["min_posts"] == 10
    assert limiares["max_dias"] == 180
    assert limiares["max_dias_passado"] == 365
    assert limiares["ultimo_treino"] == ULTIMO
    json.dumps(limiares)  # vai no pacote: tem de ser JSON


def test_limiares_nao_dependem_do_caso(limiares):
    antes = dict(limiares)
    _normal(limiares, largura_log=5.0)
    assert limiares == antes


def test_limiares_sem_distancias():
    lim = limiares_da_calibracao(pd.Series([0.1, 0.2, 0.3]), None, ULTIMO, CFG)
    assert lim["dist_knn"] is None
    json.dumps(lim)


def test_caso_normal_sem_motivo(limiares):
    flag = _normal(limiares)
    assert isinstance(flag, ConfidenceFlag)
    assert flag.baixa_confianca is False and flag.motivos == []


def test_intervalo_largo_sozinho_com_numeros(limiares):
    flag = _normal(limiares, largura_log=1.28)
    assert _codigos(flag) == ["intervalo_largo"]
    assert flag.baixa_confianca is True
    txt = flag.motivos[0].texto
    assert "3,6 vezes" in txt  # exp(1,28) = 3,6
    assert "2,2 vezes" in txt  # exp(0,8) = 2,2, o limiar


def test_largura_no_limiar_nao_dispara(limiares):
    assert _normal(limiares, largura_log=0.8).motivos == []


def test_canal_novo_sozinho_com_resolucao(limiares):
    flag = _normal(limiares, posts_maduros=4)
    assert _codigos(flag) == ["canal_novo"]
    txt = flag.motivos[0].texto
    assert "4 posts" in txt and "mais 6 posts publicados" in txt


def test_canal_novo_no_limite(limiares):
    assert _normal(limiares, posts_maduros=10).motivos == []
    assert _codigos(_normal(limiares, posts_maduros=9)) == ["canal_novo"]


def test_extracao_falhou_lista_o_que_faltou(limiares):
    flag = _normal(limiares, extracao_falhou=["capa"])
    assert _codigos(flag) == ["extracao_falhou"]
    txt = flag.motivos[0].texto
    assert "capa" in txt and "reenvie" in txt


def test_extracao_falhou_com_dois_itens(limiares):
    txt = _normal(limiares, extracao_falhou=["capa", "vídeo"]).motivos[0].texto
    assert "capa" in txt and "vídeo" in txt


def test_fora_da_distribuicao_sozinho(limiares):
    flag = _normal(limiares, dist_knn=12.5)
    assert _codigos(flag) == ["fora_da_distribuicao"]
    assert "12,5" in flag.motivos[0].texto


def test_fora_da_distribuicao_so_com_indice(limiares):
    # sem distância (a tool sem índice de comparáveis) a regra não se aplica
    assert _normal(limiares, dist_knn=None).motivos == []
    sem = limiares_da_calibracao(pd.Series([0.1, 0.2]), None, ULTIMO, CFG)
    assert _normal(sem, largura_log=0.05, dist_knn=99.0).motivos == []


def test_fora_da_distribuicao_ignorada_se_extracao_falhou(limiares):
    # 5.4: o `elif` da aula; a falha de extração já explica a distância ausente ou torta
    flag = _normal(limiares, extracao_falhou=["capa"], dist_knn=50.0)
    assert _codigos(flag) == ["extracao_falhou"]


def test_data_no_futuro_dispara(limiares):
    flag = _normal(limiares, data_planejada=datetime(2027, 1, 15, tzinfo=UTC))
    assert _codigos(flag) == ["data_fora_do_periodo"]
    assert "228 dias" in flag.motivos[0].texto  # 01/06/2026 a 15/01/2027


def test_data_no_limite_do_futuro_nao_dispara(limiares):
    # 180 dias depois de 01/06/2026 = 28/11/2026
    assert _normal(limiares, data_planejada=datetime(2026, 11, 28, tzinfo=UTC)).motivos == []
    flag = _normal(limiares, data_planejada=datetime(2026, 11, 29, tzinfo=UTC))
    assert _codigos(flag) == ["data_fora_do_periodo"]


def test_data_muito_no_passado_dispara(limiares):
    flag = _normal(limiares, data_planejada=datetime(2025, 5, 1, tzinfo=UTC))
    assert _codigos(flag) == ["data_fora_do_periodo"]


def test_setor_desconhecido_nunca_e_usado(limiares):
    flag = _normal(limiares, posts_maduros=1, largura_log=9, extracao_falhou=["capa"], dist_knn=99)
    assert "setor_desconhecido" not in _codigos(flag)


def test_varios_motivos_juntos_e_numero_nao_escondido(limiares):
    flag = _normal(limiares, posts_maduros=4, extracao_falhou=["capa"], largura_log=1.0)
    assert set(_codigos(flag)) == {"canal_novo", "extracao_falhou", "intervalo_largo"}
    assert flag.baixa_confianca


def test_avaliar_flag_caso_pequeno():
    y = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    pred = pd.Series([10.0, 11.0, 7.0, 10.0, 14.0])  # erros 0, 1, 3, 0, 4
    motivos = [[], [], ["canal_novo"], [], ["canal_novo", "intervalo_largo"]]
    res = avaliar_flag(y, pred, motivos)
    geral = res.set_index("grupo")
    assert geral.loc["sinalizados", "n"] == 2
    assert geral.loc["sinalizados", "taxa"] == pytest.approx(0.4)
    assert geral.loc["sinalizados", "mae"] == pytest.approx(3.5)
    assert geral.loc["nao_sinalizados", "n"] == 3
    assert geral.loc["nao_sinalizados", "mae"] == pytest.approx(1 / 3)
    assert geral.loc["canal_novo", "n"] == 2
    assert geral.loc["canal_novo", "mae"] == pytest.approx(3.5)
    assert geral.loc["intervalo_largo", "n"] == 1
    assert geral.loc["intervalo_largo", "mae"] == pytest.approx(4.0)


def test_avaliar_flag_tamanhos_diferentes_e_erro():
    with pytest.raises(ValueError, match="tamanho"):
        avaliar_flag(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), [[]])
