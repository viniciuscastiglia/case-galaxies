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


# ---- Task 3: features da capa, lote, concordância -------------------------------------------

CFG_TEXTO = {"entrada": [416, 736], "score_min": 0.5, "area_min": 0.002}
SEIS = {
    "det_rosto_thumb",
    "det_n_rostos_thumb",
    "det_area_rosto_thumb",
    "det_texto_thumb",
    "det_n_caixas_texto_thumb",
    "det_area_texto_thumb",
}


def test_features_da_capa_colunas_e_tipos():
    from previa.detectores import features_da_capa

    rosto = DetectorFalso(np.array([_linha(10, 10)], dtype=np.float32))
    texto = DetectorTextoFalso([_caixa(10, 10, 200, 50)], [0.9])
    f = features_da_capa(_capa(), rosto, texto, CFG_TEXTO)
    assert set(f) == SEIS
    assert f["det_rosto_thumb"] is True and f["det_n_rostos_thumb"] == 1
    assert f["det_texto_thumb"] is True and f["det_n_caixas_texto_thumb"] == 1
    assert isinstance(f["det_area_rosto_thumb"], float) and f["det_area_rosto_thumb"] > 0
    assert f["det_area_texto_thumb"] > 0

    vazio = features_da_capa(_capa(), DetectorFalso(None), DetectorTextoFalso(), CFG_TEXTO)
    assert vazio["det_rosto_thumb"] is False and vazio["det_texto_thumb"] is False
    assert vazio["det_n_rostos_thumb"] == 0 and vazio["det_n_caixas_texto_thumb"] == 0


class Contador:
    """Detectores falsos que contam as chamadas."""

    def __init__(self):
        self.rosto = DetectorFalso(None)
        self.texto = DetectorTextoFalso()
        self.chamadas = 0
        det = self.rosto.detect
        self.rosto.detect = lambda bgr: (setattr(self, "chamadas", self.chamadas + 1), det(bgr))[1]


def _capas(pasta, n):
    from PIL import Image

    pasta.mkdir(parents=True, exist_ok=True)
    ids = [f"{chr(97 + i)}" * 11 for i in range(n)]
    for v in ids:
        Image.new("RGB", (48, 36), (10, 20, 30)).save(pasta / f"{v}.jpg")
    return ids


def test_lote_retoma_sem_refazer(tmp_path):
    from previa.detectores import carregar_deteccoes, detectar_capas

    ids = _capas(tmp_path / "thumbs", 3)
    c = Contador()
    args = (ids, tmp_path / "thumbs", tmp_path / "det", c.rosto, c.texto, CFG_TEXTO, 2)
    r1 = detectar_capas(*args, log=lambda *_: None)
    assert r1["novos"] == 3 and r1["erros"] == 0
    assert c.chamadas == 3
    r2 = detectar_capas(*args, log=lambda *_: None)
    assert r2["novos"] == 0 and r2["pulados"] == 3
    assert c.chamadas == 3
    df = carregar_deteccoes(tmp_path / "det")
    assert sorted(df.index) == sorted(ids) and df.index.name == "video_id"
    assert set(df.columns) == SEIS
    assert not list((tmp_path / "det").glob("*.tmp"))


def test_lote_capa_corrompida_conta_erro_e_segue(tmp_path):
    from previa.detectores import carregar_deteccoes, detectar_capas

    ids = _capas(tmp_path / "thumbs", 3)
    (tmp_path / "thumbs" / f"{ids[1]}.jpg").write_bytes(b"isto nao e um jpeg")
    msgs = []
    c = Contador()
    r = detectar_capas(
        ids, tmp_path / "thumbs", tmp_path / "det", c.rosto, c.texto, CFG_TEXTO, 2, log=msgs.append
    )
    assert r["erros"] == 1 and r["novos"] == 3
    assert sorted(carregar_deteccoes(tmp_path / "det").index) == sorted([ids[0], ids[2]])
    assert any(ids[1] in str(m) for m in msgs)


def test_lote_sem_thumb_conta(tmp_path):
    from previa.detectores import detectar_capas

    ids = _capas(tmp_path / "thumbs", 2)
    c = Contador()
    r = detectar_capas(
        [*ids, "z" * 11],
        tmp_path / "thumbs",
        tmp_path / "det",
        c.rosto,
        c.texto,
        CFG_TEXTO,
        2,
        log=lambda *_: None,
    )
    assert r["sem_thumb"] == 1 and r["novos"] == 2


def test_concordancia_kappa_e_divergentes():
    import pandas as pd

    from previa.detectores import concordancia

    ids = [f"{i}" * 11 for i in "abcd"]
    det = pd.DataFrame(
        {"det_rosto_thumb": [True, False, True, False], "det_texto_thumb": [True] * 4}, index=ids
    )
    det.index.name = "video_id"
    rot = pd.DataFrame(
        {"tem_rosto": [True, False, False, False], "texto_na_tela": [True] * 4}, index=ids
    )
    rot.index.name = "post_id"
    t = concordancia(det, rot)
    assert len(t) == 2
    rosto = t.iloc[0]
    assert rosto["n"] == 4 and rosto["concordancia"] == 0.75 and rosto["divergentes"] == 1
    assert not np.isnan(rosto["kappa"])
    assert np.isnan(t.iloc[1]["kappa"])
