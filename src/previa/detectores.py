"""Detectores de rosto e de texto na capa (decisão #31; aula 4.6).

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


def criar_yunet(cfg: dict):
    """O YuNet do `opencv_zoo`: baixa o .onnx se faltar e confere o sha256 da config."""
    import cv2

    pasta = DADOS / "modelos"
    caminho = pasta / cfg["arquivo"]
    if not caminho.exists():
        pasta.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(cfg["url"], follow_redirects=True, timeout=60)
        resp.raise_for_status()
        tmp = caminho.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(caminho)  # nunca um modelo pela metade
    obtido = _sha256(caminho)
    if obtido != cfg["sha256"]:
        raise ValueError(
            f"sha256 de {caminho.name} não bate: esperado {cfg['sha256']}, obtido {obtido}"
        )
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
