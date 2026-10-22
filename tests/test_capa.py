"""A capa do criativo novo (F17 na previsão, F11): as mesmas funções do treino, sem torch."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from previa.amostra import ler_config
from previa.capa import BLOCOS_DA_CAPA, extras_da_capa
from previa.visual import aparencia_thumbs, preparar_thumb, sondas_relativas


class EncoderFalso:
    """Vetores determinísticos de norma 1 a partir do conteúdo; sem torch."""

    dim = 8

    def embed_imagens(self, imagens):
        out = []
        for im in imagens:
            a = np.asarray(im, float)
            v = np.array([a[..., c % 3].mean() + c for c in range(self.dim)])
            v = v + np.arange(self.dim) * a.std()
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    def embed_textos(self, textos):
        rng = np.random.default_rng(len(textos))
        v = rng.normal(size=(len(textos), self.dim))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _capa_480x360(caminho):
    """Como na base: o quadro vertical no centro e as laterais com uma cópia desfocada."""
    rng = np.random.default_rng(3)
    centro = rng.integers(0, 255, size=(360, 203, 3), dtype=np.uint8)
    lateral = np.full((360, 138, 3), 40, dtype=np.uint8)
    Image.fromarray(np.concatenate([lateral, centro, lateral], axis=1)).save(caminho)


def _nomes_sondas():
    return [f"sonda_{n}_thumb" for n in ler_config()["siglip"]["sondas"]]


def _bundle():
    colunas = {
        "sondas_thumb": _nomes_sondas(),
        "aparencia_thumb": [
            f"thumb_{c}" for c in ("brilho", "contraste", "saturacao", "colorfulness")
        ],
    }
    return SimpleNamespace(extras=list(colunas), colunas_extras=colunas)


def test_paridade_com_o_caminho_do_treino(tmp_path):
    caminho = tmp_path / "capa.jpg"
    _capa_480x360(caminho)
    enc = EncoderFalso()
    valores, embs, ausentes = extras_da_capa(caminho, _bundle(), enc)

    # o caminho do treino: aparencia_thumbs e sondas_relativas sobre preparar_thumb
    app = aparencia_thumbs(["capa"], tmp_path).iloc[0]
    sondas = ler_config()["siglip"]["sondas"]
    emb = enc.embed_imagens([preparar_thumb(caminho)])
    rel = sondas_relativas(emb, enc.embed_textos(list(sondas.values())))[0]

    for c, v in app.items():
        assert valores[c] == pytest.approx(v)
    for n, v in zip(sondas, rel, strict=True):
        assert valores[f"sonda_{n}_thumb"] == pytest.approx(v)
    assert ausentes == []
    assert embs["thumb"] == pytest.approx(emb[0])
    assert set(valores) == {c for cs in _bundle().colunas_extras.values() for c in cs}


def test_sem_imagem_declara_os_blocos_ausentes():
    valores, embs, ausentes = extras_da_capa(None, _bundle(), EncoderFalso())
    assert valores == {}
    assert embs == {"thumb": None}
    assert ausentes == ["sondas_thumb", "aparencia_thumb"] == list(BLOCOS_DA_CAPA)


def test_imagem_inexistente_levanta(tmp_path):
    with pytest.raises(FileNotFoundError):
        extras_da_capa(tmp_path / "nao_existe.jpg", _bundle(), EncoderFalso())


def test_coluna_do_bundle_que_a_capa_nao_calcula_levanta(tmp_path):
    caminho = tmp_path / "capa.jpg"
    _capa_480x360(caminho)
    b = _bundle()
    b.colunas_extras["sondas_thumb"] = [*b.colunas_extras["sondas_thumb"], "sonda_nova_thumb"]
    with pytest.raises(ValueError, match="sonda_nova_thumb"):
        extras_da_capa(caminho, b, EncoderFalso())


def test_encaixa_no_features_de_um_short(tmp_path):
    """Os valores e os ausentes entram em `novo` do jeito que o modelo espera."""
    valores, _, ausentes = extras_da_capa(None, _bundle(), EncoderFalso())
    novo = {"extras": valores, "extras_ausentes": {b: "sem capa" for b in ausentes}}
    assert set(novo["extras_ausentes"]) == set(_bundle().extras)
    assert isinstance(pd.Series(valores), pd.Series)


class EncoderContador(EncoderFalso):
    """O encoder falso, contando quantas vezes a imagem passou por ele."""

    def __init__(self):
        self.chamadas = 0

    def embed_imagens(self, imagens):
        self.chamadas += 1
        return super().embed_imagens(imagens)


def test_mesma_imagem_duas_vezes_chama_o_encoder_uma_vez(tmp_path):
    caminho = tmp_path / "capa.jpg"
    _capa_480x360(caminho)
    enc = EncoderContador()
    primeira = extras_da_capa(caminho, _bundle(), enc)
    segunda = extras_da_capa(caminho, _bundle(), enc)
    assert enc.chamadas == 1
    assert primeira[0] == segunda[0] and segunda[1]["thumb"] == pytest.approx(primeira[1]["thumb"])


def test_mesmo_conteudo_em_outro_caminho_usa_o_cache(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _capa_480x360(a)
    b.write_bytes(a.read_bytes())
    enc = EncoderContador()
    extras_da_capa(a, _bundle(), enc)
    extras_da_capa(b, _bundle(), enc)
    assert enc.chamadas == 1


def test_mesmo_caminho_com_conteudo_novo_recalcula(tmp_path):
    caminho = tmp_path / "capa.jpg"
    _capa_480x360(caminho)
    enc = EncoderContador()
    antes = extras_da_capa(caminho, _bundle(), enc)[0]
    Image.fromarray(np.full((360, 480, 3), 200, dtype=np.uint8)).save(caminho)
    depois = extras_da_capa(caminho, _bundle(), enc)[0]
    assert enc.chamadas == 2 and depois != antes


def test_o_cache_devolve_copias(tmp_path):
    caminho = tmp_path / "capa.jpg"
    _capa_480x360(caminho)
    enc = EncoderContador()
    valores, embs, _ = extras_da_capa(caminho, _bundle(), enc)
    original = dict(valores)
    valores["thumb_brilho"] = -1.0
    embs["thumb"][0] = 99.0
    valores2, embs2, _ = extras_da_capa(caminho, _bundle(), enc)
    assert valores2 == original and embs2["thumb"][0] != 99.0
