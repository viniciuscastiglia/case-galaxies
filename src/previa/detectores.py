"""Detectores de rosto e de texto na capa (decisão #31; aula 4.6).

O texto usa o detector DB do opencv_zoo (só caixas, sem ler), sem o PaddleOCR.

Camada 1 do bloco multimodal: sinais baratos e determinísticos sobre as 30 mil capas, sem torch e
sem LightGBM neste módulo (OpenMP no macOS, hub, troubleshooting). A imagem de entrada é a mesma
que o SigLIP vê (`recortar_9_16` sobre a capa em RGB). Os dois detectores esperam BGR: a conversão é
`rgb[:, :, ::-1]` seguida de `np.ascontiguousarray` (4.6, erro comum 2).

O YuNet (`opencv_zoo`, licença MIT) devolve `faces = None` quando não há rosto, não um array vazio.
Todos os parâmetros moram em `configs/multimodal.yaml`, seção `detectores`.
"""

import hashlib

import httpx
import numpy as np

from previa.amostra import ler_config
from previa.caminhos import DADOS


def ler_config_detectores() -> dict:
    return ler_config()["detectores"]


def _sha256(caminho) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _garantir_modelo(cfg: dict):
    """Baixa o .onnx se faltar. O sha256 é conferido no conteúdo baixado, antes de gravar:
    um download errado nunca chega a `DADOS / "modelos"`. Um arquivo que já existe é
    conferido a cada chamada."""
    pasta = DADOS / "modelos"
    caminho = pasta / cfg["arquivo"]
    if not caminho.exists():
        resp = httpx.get(cfg["url"], follow_redirects=True, timeout=60)
        resp.raise_for_status()
        _conferir(cfg["arquivo"], hashlib.sha256(resp.content).hexdigest(), cfg["sha256"])
        pasta.mkdir(parents=True, exist_ok=True)
        tmp = caminho.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(caminho)  # nunca um modelo pela metade
    else:
        _conferir(caminho.name, _sha256(caminho), cfg["sha256"])
    return caminho


def _conferir(nome: str, obtido: str, esperado: str) -> None:
    if obtido != esperado:
        raise ValueError(f"sha256 de {nome} não bate: esperado {esperado}, obtido {obtido}")


def criar_yunet(cfg: dict):
    """O YuNet do `opencv_zoo`: baixa o .onnx se faltar e confere o sha256 da config."""
    import cv2

    caminho = _garantir_modelo(cfg)
    return cv2.FaceDetectorYN.create(
        str(caminho),
        "",
        (320, 320),  # o tamanho real entra a cada imagem, em rostos_na_capa
        cfg["score_threshold"],
        cfg["nms_threshold"],
        cfg["top_k"],
    )


def rostos_na_capa(rgb: np.ndarray, detector) -> dict:
    """Número de rostos e área do maior, como fração do quadro (4.6)."""
    altura, largura = rgb.shape[:2]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    detector.setInputSize((largura, altura))  # (largura, altura), a cada imagem
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return {"n_rostos": 0, "area_maior_rosto": 0.0}
    faces = np.asarray(faces, dtype=float)
    areas = faces[:, 2] * faces[:, 3] / (largura * altura)  # colunas 2 e 3: largura e altura
    return {"n_rostos": len(faces), "area_maior_rosto": float(areas.max())}


def criar_detector_texto(cfg: dict):
    """O detector de texto PP-OCRv3 do `opencv_zoo` (DB): só acha as caixas, não lê o texto.

    Os parâmetros são os da demo do zoo (`demo.py`, `ppocr_det.py`): média ImageNet e escala
    1/255/desvio, não a 1/255 com média do Paddle.
    """
    import cv2

    caminho = _garantir_modelo(cfg)
    modelo = cv2.dnn.TextDetectionModel_DB(str(caminho))
    modelo.setBinaryThreshold(cfg["binary_threshold"])
    modelo.setPolygonThreshold(cfg["polygon_threshold"])
    modelo.setUnclipRatio(cfg["unclip_ratio"])
    modelo.setMaxCandidates(cfg["max_candidates"])
    modelo.setInputSize(tuple(cfg["entrada"]))
    modelo.setInputMean(tuple(cfg["media"]))
    modelo.setInputScale(1.0 / 255.0 / np.array(cfg["desvio"]))
    return modelo


def texto_na_capa(
    rgb: np.ndarray, detector, entrada: tuple[int, int], score_min: float, area_min: float
) -> dict:
    """Número de caixas de texto e fração do quadro coberta por elas (união, 4.6)."""
    import cv2

    largura, altura = entrada
    redim = cv2.resize(rgb, (largura, altura), interpolation=cv2.INTER_AREA)
    bgr = np.ascontiguousarray(redim[:, :, ::-1])
    caixas, scores = detector.detect(bgr)
    mascara = np.zeros((altura, largura), np.uint8)
    n_linhas = 0
    for caixa, score in zip(caixas, scores, strict=True):
        pontos = np.asarray(caixa, dtype=float).reshape(-1, 2)  # pode vir em int
        x, y = pontos[:, 0], pontos[:, 1]
        # área do polígono pela fórmula do cadarço, em float
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if score < score_min or area / (largura * altura) < area_min:
            continue
        cv2.fillPoly(mascara, [np.round(pontos).astype(np.int32)], 255)
        n_linhas += 1
    return {"n_linhas": n_linhas, "area_texto": float((mascara > 0).mean())}
