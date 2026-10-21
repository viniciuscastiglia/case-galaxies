"""Detectores de rosto e texto na capa (aula 4.6; decisão #31).

Detectores falsos: sem rede, sem modelo.
"""

import numpy as np
import pytest

from previa.detectores import criar_yunet, ler_config_detectores, rostos_na_capa


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
