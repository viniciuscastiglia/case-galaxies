"""Detectores de rosto e texto na capa (aula 4.6; decisão #31).

Detectores falsos: sem rede, sem modelo.
"""

import numpy as np
import pytest

from previa.detectores import (
    criar_yunet,
    ler_config_detectores,
    rostos_na_capa,
    texto_na_capa,
)


class DetectorFalso:
    """Guarda o que recebeu, como o `cv2.FaceDetectorYN`: `setInputSize((w, h))` e `detect(bgr)`."""

    def __init__(self, faces):
        self.faces = faces
        self.tamanho = None
        self.imagem = None

    def setInputSize(self, tamanho):
        self.tamanho = tamanho

    def detect(self, bgr):
        self.imagem = bgr
        return 1, self.faces


def _linha(w, h):
    # YuNet: x, y, w, h, 10 pontos, score (15 colunas)
    return [5, 5, w, h] + [0] * 10 + [0.99]


def test_rosto_area_e_contagem():
    rgb = np.zeros((200, 100, 3), np.uint8)
    det = DetectorFalso(np.array([_linha(10, 10), _linha(20, 30)], dtype=np.float32))
    r = rostos_na_capa(rgb, det)
    assert r["n_rostos"] == 2
    assert r["area_maior_rosto"] == pytest.approx(600 / 20000)


def test_rosto_recebe_bgr_e_tamanho_certo():
    rgb = np.zeros((200, 100, 3), np.uint8)
    rgb[:, :] = (255, 0, 0)
    det = DetectorFalso(None)
    rostos_na_capa(rgb, det)
    assert det.tamanho == (100, 200)  # largura, altura
    assert tuple(det.imagem[0, 0]) == (0, 0, 255)
    assert det.imagem.flags["C_CONTIGUOUS"]


def test_sem_rosto_faces_none():
    r = rostos_na_capa(np.zeros((200, 100, 3), np.uint8), DetectorFalso(None))
    assert r == {"n_rostos": 0, "area_maior_rosto": 0.0}


def test_capa_preta_sem_rosto():
    r = rostos_na_capa(np.zeros((200, 100, 3), np.uint8), DetectorFalso(None))
    assert r == {"n_rostos": 0, "area_maior_rosto": 0.0}


def test_sha256_errado_levanta(tmp_path, monkeypatch):
    import previa.detectores as d

    monkeypatch.setattr(d, "DADOS", tmp_path)
    cfg = dict(ler_config_detectores()["yunet"])
    (tmp_path / "modelos").mkdir()
    (tmp_path / "modelos" / cfg["arquivo"]).write_bytes(b"qualquer coisa")
    with pytest.raises(ValueError, match="sha256"):
        criar_yunet(cfg)


ENTRADA = (416, 736)


class DetectorTextoFalso:
    """Como o `TextDetectionModel_DB`: `detect(bgr) -> (caixas, scores)`."""

    def __init__(self, caixas=(), scores=()):
        self.caixas = caixas
        self.scores = scores
        self.imagem = None

    def detect(self, bgr):
        self.imagem = bgr
        return self.caixas, self.scores


def _caixa(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _capa():
    return np.zeros((360, 202, 3), np.uint8)


def test_texto_filtra_score_e_area_minima():
    det = DetectorTextoFalso(
        [_caixa(10, 10, 200, 50), _caixa(10, 100, 200, 50), _caixa(300, 300, 2, 2)],
        [0.9, 0.3, 0.9],
    )
    r = texto_na_capa(_capa(), det, ENTRADA, score_min=0.5, area_min=0.002)
    assert r["n_linhas"] == 1


def test_area_texto_uniao_sem_contar_duas_vezes():
    det = DetectorTextoFalso([_caixa(0, 0, 100, 100), _caixa(50, 0, 100, 100)], [0.9, 0.9])
    r = texto_na_capa(_capa(), det, ENTRADA, score_min=0.5, area_min=0.002)
    assert r["n_linhas"] == 2
    assert r["area_texto"] == pytest.approx(15000 / (416 * 736), rel=0.02)


def test_texto_recebe_bgr_no_tamanho_de_entrada():
    rgb = np.zeros((360, 202, 3), np.uint8)
    rgb[:, :] = (255, 0, 0)
    det = DetectorTextoFalso()
    texto_na_capa(rgb, det, ENTRADA, score_min=0.5, area_min=0.002)
    assert det.imagem.shape == (736, 416, 3)
    assert det.imagem.flags["C_CONTIGUOUS"]
    assert tuple(det.imagem[0, 0]) == (0, 0, 255)


def test_texto_vazio():
    r = texto_na_capa(_capa(), DetectorTextoFalso((), ()), ENTRADA, score_min=0.5, area_min=0.002)
    assert r == {"n_linhas": 0, "area_texto": 0.0}


def test_download_com_hash_errado_nao_fica_no_lugar(tmp_path, monkeypatch):
    import httpx

    import previa.detectores as d

    class Resp:
        content = b"bytes quaisquer"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(d, "DADOS", tmp_path)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
    cfg = dict(ler_config_detectores()["texto"])
    with pytest.raises(ValueError, match="sha256"):
        d.criar_detector_texto(cfg)
    assert list((tmp_path / "modelos").glob("*")) == []
