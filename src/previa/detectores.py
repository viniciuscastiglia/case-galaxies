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
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

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


def features_da_capa(
    rgb: np.ndarray, yunet, detector_texto, cfg_texto: dict, capa: tuple[int, int] | None = None
) -> dict:
    """As 6 colunas `det_*` de uma capa já recortada 9:16 em RGB. A tool chama esta mesma função
    na capa do criativo (paridade treino/uso).

    `capa` é (largura, altura) das capas da base (`detectores.capa`, 202x360): o recorte é
    redimensionado para lá antes dos dois detectores, então uma capa 1080x1920 da tool chega
    com a mesma forma das 30 mil da base (na base é um no-op).
    """
    import cv2

    capa = tuple(capa or ler_config_detectores()["capa"])
    if rgb.shape[1::-1] != capa:
        rgb = cv2.resize(rgb, capa, interpolation=cv2.INTER_AREA)
    rosto = rostos_na_capa(rgb, yunet)
    texto = texto_na_capa(
        rgb,
        detector_texto,
        tuple(cfg_texto["entrada"]),
        cfg_texto["score_min"],
        cfg_texto["area_min"],
    )
    return {
        "det_rosto_thumb": bool(rosto["n_rostos"] > 0),
        "det_n_rostos_thumb": int(rosto["n_rostos"]),
        "det_area_rosto_thumb": float(rosto["area_maior_rosto"]),
        "det_texto_thumb": bool(texto["n_linhas"] > 0),
        "det_n_caixas_texto_thumb": int(texto["n_linhas"]),
        "det_area_texto_thumb": float(texto["area_texto"]),
    }


COLUNAS_DET = [
    "det_rosto_thumb",
    "det_n_rostos_thumb",
    "det_area_rosto_thumb",
    "det_texto_thumb",
    "det_n_caixas_texto_thumb",
    "det_area_texto_thumb",
]


def carregar_deteccoes(pasta: Path) -> pd.DataFrame:
    """Todos os lotes salvos: uma linha por vídeo (índice video_id)."""
    lotes = sorted(Path(pasta).glob("lote_*.parquet"))
    if not lotes:
        vazio = pd.DataFrame(columns=COLUNAS_DET)
        vazio.index.name = "video_id"
        return vazio
    return pd.concat([pd.read_parquet(p) for p in lotes]).set_index("video_id")


def detectar_capas(
    ids,
    pasta_thumbs: Path,
    destino: Path,
    yunet,
    detector_texto,
    cfg_texto: dict,
    lote: int,
    log=print,
    capa: tuple[int, int] | None = None,
) -> dict:
    """Detecta rosto e texto nas capas de `ids` que ainda não estão em `destino`, em lotes.

    Uma capa ausente conta em `sem_thumb`; uma corrompida vai para o `log` com o motivo, conta em
    `erros` e o lote segue (o vídeo fica sem linha). Só a leitura da imagem é tolerada (`OSError`,
    que inclui `UnidentifiedImageError` e JPEG truncado); uma exceção de detector propaga.
    """
    from PIL import Image

    from previa.thumbs import recortar_9_16

    capa = tuple(capa or ler_config_detectores()["capa"])
    pasta_thumbs = Path(pasta_thumbs)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    feitos = set(carregar_deteccoes(destino).index)
    faltam = [v for v in ids if v not in feitos and (pasta_thumbs / f"{v}.jpg").exists()]
    resumo = {
        "novos": len(faltam),
        "pulados": sum(v in feitos for v in ids),
        "sem_thumb": sum(v not in feitos and not (pasta_thumbs / f"{v}.jpg").exists() for v in ids),
        "erros": 0,
    }
    n_lote = len(list(destino.glob("lote_*.parquet")))
    for i in range(0, len(faltam), lote):
        bloco = faltam[i : i + lote]
        linhas = []
        for v in bloco:
            try:
                with Image.open(pasta_thumbs / f"{v}.jpg") as img:
                    rgb = recortar_9_16(np.asarray(img.convert("RGB")))
            except OSError as e:  # capa corrompida: registra e segue (princípio 7)
                resumo["erros"] += 1
                log(f"capa {v} ignorada: {type(e).__name__}: {e}")
                continue
            feats = features_da_capa(rgb, yunet, detector_texto, cfg_texto, capa)
            linhas.append({"video_id": v, **feats})
        if linhas:
            arquivo = destino / f"lote_{n_lote:05d}.parquet"
            tmp = arquivo.with_suffix(".tmp")
            pd.DataFrame(linhas).to_parquet(tmp, index=False)
            os.replace(tmp, arquivo)  # atômico: nunca fica um lote pela metade
            n_lote += 1
        log(f"{i + len(bloco)}/{len(faltam)} capas processadas")
    return resumo


PARES = [("det_rosto_thumb", "tem_rosto"), ("det_texto_thumb", "texto_na_tela")]


def concordancia(det: pd.DataFrame, rotulos: pd.DataFrame) -> pd.DataFrame:
    """Detector clássico contra a rubrica do VLM, só nos vídeos que têm os dois (kappa de Cohen)."""
    from sklearn.metrics import cohen_kappa_score

    if "post_id" in rotulos.columns:
        rotulos = rotulos.set_index("post_id")
    comuns = det.index.intersection(rotulos.index)
    linhas = []
    for col_det, col_vlm in PARES:
        # a rubrica é bool obrigatório no Pydantic (vlm.py) e as falhas vão para erros_v1.jsonl:
        # não há NaN, então o astype(bool) não esconde nada
        a = det.loc[comuns, col_det].astype(bool)
        b = rotulos.loc[comuns, col_vlm].astype(bool)
        um_lado_so = a.nunique() < 2 or b.nunique() < 2
        linhas.append(
            {
                "detector": col_det,
                "vlm": col_vlm,
                "n": len(comuns),
                "concordancia": float((a == b).mean()) if len(comuns) else float("nan"),
                "kappa": float("nan") if um_lado_so else float(cohen_kappa_score(a, b)),
                "divergentes": int((a != b).sum()),
                # det=T e VLM=F: possível falso positivo do detector
                "so_detector": int((a & ~b).sum()),
                # det=F e VLM=T: esperado, o VLM rotula o vídeo inteiro e o detector vê só a capa
                "so_vlm": int((~a & b).sum()),
            }
        )
    return pd.DataFrame(linhas)


def _divergentes(det: pd.DataFrame, rotulos: pd.DataFrame) -> pd.DataFrame:
    comuns = det.index.intersection(rotulos.index)
    partes = []
    for col_det, col_vlm in PARES:
        a = det.loc[comuns, col_det].astype(bool)  # sem NaN: ver `concordancia`
        b = rotulos.loc[comuns, col_vlm].astype(bool)
        d = pd.DataFrame({"par": col_det, "detector": a, "vlm": b})[a != b]
        partes.append(d)
    return pd.concat(partes).rename_axis("video_id").reset_index()


def main(argv=None) -> int:
    acao = (argv or sys.argv[1:] or [""])[0]
    if acao not in ("capas", "concordancia"):
        print("uso: python -m previa.detectores capas|concordancia", file=sys.stderr)
        return 2
    cfg = ler_config_detectores()
    pasta = DADOS / f"det_thumb_v{cfg['versao']}"
    if acao == "capas":
        from previa.thumbs import PASTA

        ids = sorted(p.stem for p in PASTA.glob("*.jpg"))
        resumo = detectar_capas(
            ids,
            PASTA,
            pasta,
            criar_yunet(cfg["yunet"]),
            criar_detector_texto(cfg["texto"]),
            cfg["texto"],
            cfg["lote"],
        )
        print(resumo, flush=True)
        det = carregar_deteccoes(pasta)
        if resumo["erros"] > 0:
            print(f"{resumo['erros']} erro(s) nas capas: consolidado não gravado", file=sys.stderr)
            return 1
        if det.empty:
            print("nenhuma detecção nos lotes: consolidado não gravado", file=sys.stderr)
            return 1
        det.reset_index().to_parquet(DADOS / f"det_thumb_v{cfg['versao']}.parquet", index=False)
        return 0
    det = carregar_deteccoes(pasta)
    if det.empty:
        print(f"nenhuma detecção em {pasta}: rode `capas` antes", file=sys.stderr)
        return 1
    linhas = [json.loads(x) for x in (DADOS / "vlm" / "rubrica_v1.jsonl").open()]
    rot = pd.DataFrame(linhas).drop_duplicates("post_id", keep="last").set_index("post_id")
    print(concordancia(det, rot).to_string(index=False))
    _divergentes(det, rot).to_csv(DADOS / "det_thumb_divergentes.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
