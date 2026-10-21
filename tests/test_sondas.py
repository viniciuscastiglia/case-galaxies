"""Testes da validação das sondas da capa contra a rubrica do VLM (aula 4.4, exercício; F26)."""

import numpy as np
import pandas as pd
import pytest

from previa.sondas import ressalvas, validar_sondas


def _ids(n):
    return [f"v{i:010d}" for i in range(n)]


def _rotulo(n=40):
    """Metade positiva, metade negativa, intercaladas."""
    return pd.Series([i % 2 == 0 for i in range(n)], index=_ids(n))


def test_sonda_que_separa_tem_auc_1_e_passa():
    rotulo = _rotulo()
    rng = np.random.default_rng(0)
    # rótulo + ruído pequeno: todo positivo fica acima de todo negativo
    sonda = rotulo.astype(float) + rng.normal(0, 0.05, len(rotulo))
    r = validar_sondas(
        pd.DataFrame({"sonda_rosto_thumb": sonda}),
        pd.DataFrame({"tem_rosto": rotulo}),
        {"rosto": "tem_rosto"},
        auc_min=0.7,
    )
    assert list(r.columns) == ["sonda", "rotulo", "setor", "n", "positivos", "auc", "passa"]
    linha = r.iloc[0]
    assert (linha["sonda"], linha["rotulo"], linha["setor"]) == ("rosto", "tem_rosto", "todos")
    assert (linha["n"], linha["positivos"]) == (40, 20)
    assert linha["auc"] == pytest.approx(1.0)
    assert bool(linha["passa"]) is True


def test_sonda_ao_acaso_nao_passa():
    rotulo = _rotulo(200)
    rng = np.random.default_rng(42)
    ao_acaso = pd.Series(rng.normal(size=len(rotulo)), index=rotulo.index)
    r = validar_sondas(
        pd.DataFrame({"sonda_texto_thumb": ao_acaso}),
        pd.DataFrame({"texto_na_tela": rotulo}),
        {"texto": "texto_na_tela"},
        auc_min=0.7,
    )
    assert r.loc[0, "auc"] < 0.7
    assert bool(r.loc[0, "passa"]) is False


def test_sonda_invertida_nao_passa():
    """AUC 0 não é "separa bem": a sonda aponta para o lado errado e a direção importa."""
    rotulo = _rotulo()
    r = validar_sondas(
        pd.DataFrame({"sonda_rosto_thumb": -rotulo.astype(float)}),
        pd.DataFrame({"tem_rosto": rotulo}),
        {"rosto": "tem_rosto"},
        auc_min=0.7,
    )
    assert r.loc[0, "auc"] == pytest.approx(0.0)
    assert bool(r.loc[0, "passa"]) is False


def test_rotulo_de_uma_classe_so_vira_nan():
    ids = _ids(10)
    sondas = pd.DataFrame({"sonda_rosto_thumb": np.linspace(0, 1, 10)}, index=ids)
    rotulos = pd.DataFrame({"tem_rosto": [True] * 10}, index=ids)  # nenhum negativo
    r = validar_sondas(sondas, rotulos, {"rosto": "tem_rosto"}, auc_min=0.7)
    assert np.isnan(r.loc[0, "auc"])
    assert pd.isna(r.loc[0, "passa"])
    assert (r.loc[0, "n"], r.loc[0, "positivos"]) == (10, 10)


def test_alinha_pelos_post_ids_em_comum():
    # sondas: v0..v5; rótulos: v3..v8; interseção: v3, v4, v5 (só 3 capas)
    sondas = pd.DataFrame({"sonda_rosto_thumb": [0.1, 0.2, 0.3, 0.9, 0.2, 0.8]}, index=_ids(6))
    rotulos = pd.DataFrame(
        {"tem_rosto": [True, False, True, True, False, False]}, index=_ids(9)[3:]
    )
    r = validar_sondas(sondas, rotulos, {"rosto": "tem_rosto"}, auc_min=0.7)
    assert (r.loc[0, "n"], r.loc[0, "positivos"]) == (3, 2)  # v3 e v5 positivos, v4 negativo
    # os positivos (0.9 e 0.8) ficam acima do negativo (0.2): AUC 1,0
    assert r.loc[0, "auc"] == pytest.approx(1.0)


def test_um_rotulo_para_duas_sondas_vira_duas_linhas():
    rotulo = _rotulo()
    boa = rotulo.astype(float)
    sondas = pd.DataFrame({"sonda_rosto_thumb": boa, "sonda_falando_thumb": -boa})
    r = validar_sondas(
        sondas,
        pd.DataFrame({"tem_rosto": rotulo}),
        {"rosto": "tem_rosto", "falando": "tem_rosto"},
        auc_min=0.7,
    )
    assert r["sonda"].tolist() == ["rosto", "falando"]
    assert r["auc"].tolist() == pytest.approx([1.0, 0.0])
    assert r["passa"].tolist() == [True, False]


def test_sonda_ou_rotulo_inexistente_e_erro_explicito():
    rotulo = _rotulo(10)
    sondas = pd.DataFrame({"sonda_rosto_thumb": np.arange(10.0)}, index=rotulo.index)
    rotulos = pd.DataFrame({"tem_rosto": rotulo})
    with pytest.raises(ValueError, match="sonda_roda_thumb"):
        validar_sondas(sondas, rotulos, {"roda": "tem_rosto"}, auc_min=0.7)
    with pytest.raises(ValueError, match="close_up_produto"):
        validar_sondas(sondas, rotulos, {"rosto": "close_up_produto"}, auc_min=0.7)


def test_nan_na_sonda_e_erro_explicito():
    rotulo = _rotulo(10)
    valores = np.arange(10.0)
    valores[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validar_sondas(
            pd.DataFrame({"sonda_rosto_thumb": valores}, index=rotulo.index),
            pd.DataFrame({"tem_rosto": rotulo}),
            {"rosto": "tem_rosto"},
            auc_min=0.7,
        )


def test_sem_post_id_em_comum_e_erro_explicito():
    sondas = pd.DataFrame({"sonda_rosto_thumb": [0.1, 0.2]}, index=["a" * 11, "b" * 11])
    rotulos = pd.DataFrame({"tem_rosto": [True, False]}, index=["c" * 11, "d" * 11])
    with pytest.raises(ValueError, match="em comum"):
        validar_sondas(sondas, rotulos, {"rosto": "tem_rosto"}, auc_min=0.7)


# --- um par pode valer só num setor (a rubrica de `close_up_produto` muda com o setor) ---


def _dois_setores():
    """20 posts de beleza e 20 de automotivo, metade positiva em cada um, intercalados.

    A sonda separa o rótulo na beleza (positivo = score alto) e erra no automotivo (o positivo tem
    score baixo, como uma sonda "produto de beleza" diante de um farol): no conjunto, o AUC cai
    para perto de 0,5 e esconde que ela funciona no setor certo (decisão da revisão final)."""
    ids = _ids(40)
    setores = pd.Series(["beleza"] * 20 + ["automotivo"] * 20, index=ids)
    rotulo = pd.Series([i % 2 == 0 for i in range(40)], index=ids)
    rng = np.random.default_rng(0)
    certo = rotulo.astype(float)
    score = certo.where(setores == "beleza", 1.0 - certo) + rng.normal(0, 0.05, 40)
    return ids, setores, rotulo, score


def test_par_com_setor_mede_so_nos_posts_do_setor():
    ids, setores, rotulo, score = _dois_setores()
    sondas = pd.DataFrame({"sonda_produto_beleza_thumb": score, "sonda_roda_thumb": score})
    rotulos = pd.DataFrame({"close_up_produto": rotulo})
    juntos = validar_sondas(
        sondas, rotulos, {"produto_beleza": "close_up_produto"}, auc_min=0.7, setores=setores
    )
    assert juntos.loc[0, "setor"] == "todos" and juntos.loc[0, "n"] == 40
    assert 0.3 < juntos.loc[0, "auc"] < 0.7  # os dois setores misturados: ~0,5
    assert bool(juntos.loc[0, "passa"]) is False

    pares = {
        "produto_beleza": {"rotulo": "close_up_produto", "setor": "beleza"},
        "roda": {"rotulo": "close_up_produto", "setor": "automotivo"},
    }
    r = validar_sondas(sondas, rotulos, pares, auc_min=0.7, setores=setores)
    assert r["sonda"].tolist() == ["produto_beleza", "roda"]
    assert r["setor"].tolist() == ["beleza", "automotivo"]
    assert r["n"].tolist() == [20, 20]
    assert r["positivos"].tolist() == [10, 10]
    # na beleza a sonda separa (AUC 1); no automotivo o mesmo score é o avesso (AUC 0)
    assert r["auc"].tolist() == pytest.approx([1.0, 0.0])
    assert r["passa"].tolist() == [True, False]


def test_par_sem_setor_continua_valendo_nos_dois_e_o_rotulo_em_texto_tambem():
    ids, setores, rotulo, score = _dois_setores()
    sondas = pd.DataFrame({"sonda_rosto_thumb": score, "sonda_roda_thumb": score})
    rotulos = pd.DataFrame({"tem_rosto": rotulo, "close_up_produto": rotulo})
    pares = {"rosto": "tem_rosto", "roda": {"rotulo": "close_up_produto", "setor": "automotivo"}}
    r = validar_sondas(sondas, rotulos, pares, auc_min=0.7, setores=setores)
    assert r["setor"].tolist() == ["todos", "automotivo"]
    assert r["n"].tolist() == [40, 20]
    # sem `setores`, os pares sem filtro valem como antes
    sem = validar_sondas(sondas, rotulos, {"rosto": "tem_rosto"}, auc_min=0.7)
    assert sem.loc[0, "auc"] == pytest.approx(r.loc[0, "auc"])


def test_setor_sem_tabela_de_setores_ou_sem_posts_e_erro_explicito():
    ids, setores, rotulo, score = _dois_setores()
    sondas = pd.DataFrame({"sonda_roda_thumb": score})
    rotulos = pd.DataFrame({"close_up_produto": rotulo})
    pares = {"roda": {"rotulo": "close_up_produto", "setor": "automotivo"}}
    with pytest.raises(ValueError, match="setores"):
        validar_sondas(sondas, rotulos, pares, auc_min=0.7)
    so_beleza = setores.where(setores == "beleza", "beleza")  # nenhum post de automotivo
    with pytest.raises(ValueError, match="automotivo"):
        validar_sondas(sondas, rotulos, pares, auc_min=0.7, setores=so_beleza)
    erro = {"roda": {"rotulo": "close_up_produto", "sector": "automotivo"}}  # chave errada
    with pytest.raises(ValueError, match="sector"):
        validar_sondas(sondas, rotulos, erro, auc_min=0.7, setores=setores)


def test_ressalvas_da_saida_em_uma_linha_cada():
    ids = _ids(6)
    sondas = pd.DataFrame({"sonda_falando_thumb": np.arange(4.0)}, index=ids[:4])
    rotulos = pd.DataFrame({"tem_rosto": [True] * 5}, index=ids[1:])
    linhas = ressalvas(sondas, rotulos, {"falando": "tem_rosto"})
    assert linhas == [
        "AUC é piso (rótulo do vídeo, sonda da capa)",
        "falando é medida contra tem_rosto (proxy)",
        "5 rótulos, 4 capas, 3 em comum",
    ]
    # sem o par "falando", a segunda ressalva não se aplica
    assert len(ressalvas(sondas, rotulos, {"rosto": "tem_rosto"})) == 2
