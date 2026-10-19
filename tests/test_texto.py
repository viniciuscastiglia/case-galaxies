"""Testes das features de texto (decisão #9; aula 3.6). Só o título: a descrição padrão dos
canais vira identificador de canal (3.6, erro comum 4). O encoder dos embeddings é falso."""

import numpy as np
import pandas as pd

from previa.texto import embeddings_com_cache, features_de_texto, titulo_de


def test_features_manuais_do_exemplo_da_aula():
    f = features_de_texto("Tutorial de pneu aro 17 em 5 passos #shorts")
    assert f["txt_caracteres"] == 43 and f["txt_palavras"] == 9
    assert f["txt_hashtags"] == 1 and f["txt_numeros"] == 2
    assert f["txt_cta"] == 0 and f["txt_pergunta"] == 0 and f["txt_ingles"] == 0
    assert f["txt_termos_setor"] == 1  # "pneu"


def test_cta_pergunta_e_termos_de_beleza():
    f = features_de_texto("Qual BASE usar na pele oleosa? comenta aqui 💄")
    assert f["txt_pergunta"] == 1 and f["txt_cta"] == 1 and f["txt_emojis"] == 1
    assert f["txt_termos_setor"] == 2  # base e pele


def test_titulo_e_a_primeira_linha_da_caption():
    caption = pd.Series(["Título do Short\nDescrição padrão do canal\n#links", "Só título"])
    assert titulo_de(caption).tolist() == ["Título do Short", "Só título"]


class EncoderFalso:
    def __init__(self):
        self.recebidos = []

    def encode(self, textos, **kw):
        self.recebidos += list(textos)
        v = np.array([[len(t), 1.0, 2.0] for t in textos])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_cache_por_video_e_hash_do_texto(tmp_path):
    arquivo = tmp_path / "emb.parquet"
    enc = EncoderFalso()
    e1 = embeddings_com_cache(["v1", "v2"], ["a", "bb"], enc, arquivo)
    assert e1.shape == (2, 3) and enc.recebidos == ["a", "bb"]
    enc.recebidos.clear()
    e2 = embeddings_com_cache(["v1", "v2"], ["a", "bb editado"], enc, arquivo)
    assert enc.recebidos == ["bb editado"]  # só o título que mudou
    np.testing.assert_allclose(e2.iloc[0], e1.iloc[0])
    assert list(e2.index) == ["v1", "v2"]
