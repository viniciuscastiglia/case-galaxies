"""Testes dos embeddings das thumbnails (decisão #12; aula 4.4), com um encoder falso: o
SigLIP 2 de verdade (1,5 GB) não roda nos testes."""

import numpy as np
import pandas as pd
from PIL import Image

from previa.visual import carregar_embeddings, embedar_thumbs, sondas_relativas


class EncoderFalso:
    """Devolve vetores de norma 1 que dependem da largura da imagem; guarda o que recebeu."""

    def __init__(self, dim=8):
        self.dim, self.larguras, self.chamadas = dim, [], 0

    def embed_imagens(self, imagens):
        self.chamadas += 1
        self.larguras += [im.size[0] for im in imagens]
        v = np.array([[im.size[0] + k for k in range(self.dim)] for im in imagens], float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _thumbs(pasta, ids):
    pasta.mkdir(parents=True, exist_ok=True)
    for v in ids:
        Image.fromarray(np.full((360, 480, 3), 120, dtype=np.uint8)).save(pasta / f"{v}.jpg")


def test_embeda_recortado_em_lotes_e_usa_o_cache(tmp_path):
    ids = [f"v{k:010d}" for k in range(5)]
    _thumbs(tmp_path / "thumbs", ids)
    enc = EncoderFalso()
    r = embedar_thumbs(ids + ["semthumb000"], tmp_path / "thumbs", tmp_path / "emb", enc, lote=2)
    assert r == {"novos": 5, "pulados": 0, "sem_thumb": 1}
    assert enc.chamadas == 3 and set(enc.larguras) == {202}  # faixa central 9:16 de 480 × 360
    emb = carregar_embeddings(tmp_path / "emb")
    assert sorted(emb.index) == ids and emb.shape[1] == 8
    np.testing.assert_allclose(np.linalg.norm(emb.to_numpy(), axis=1), 1.0, rtol=1e-6)
    enc2 = EncoderFalso()
    r2 = embedar_thumbs(ids, tmp_path / "thumbs", tmp_path / "emb", enc2, lote=2)
    assert r2["pulados"] == 5 and enc2.chamadas == 0


def test_sondas_relativas_tiram_a_media_da_imagem():
    img = np.array([[1.0, 0.0], [0.6, 0.8]])
    txt = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    rel = sondas_relativas(img, txt)
    np.testing.assert_allclose(rel.sum(axis=1), 0, atol=1e-12)
    bruto = img @ txt.T
    assert (np.argsort(rel, axis=1) == np.argsort(bruto, axis=1)).all()  # a ordem não muda


def test_carregar_embeddings_vazio(tmp_path):
    assert carregar_embeddings(tmp_path / "nada").empty
    assert isinstance(carregar_embeddings(tmp_path / "nada"), pd.DataFrame)
