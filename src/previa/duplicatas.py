"""Duplicatas e reuploads (aula 2.2, família 5; decisão #1 de 14/10, em docs/DECISOES.md).

Duplicata = título normalizado (caixa e espaços) e duração iguais E a mesma thumbnail: dHash de
64 bits a até 10 bits da cópia mais antiga. Vale também entre canais, como o criativo postado pela
marca e pela influenciadora (aula 2.2, família 5; decisão de 17/10). Título e duração sozinhos
juntam vídeos diferentes de título genérico ("🧡", "CH 01"): em 14/10, só 262 das 1.125 cópias
tinham a mesma imagem. Só a primeira cópia publicada vira linha de treino e de teste
(`linhas_do_modelo`); todas ficam na base, para o histórico de cada canal.

As thumbnails ficam só em memória; o cache guarda o hash (data/thumb_hash.parquet, apagado em
30/10 com os dados brutos).
"""

import io
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from previa.caminhos import CONFIGS
from previa.coletor import DADOS, sem_chave

CONFIG = CONFIGS / "duplicatas.yaml"
CACHE = DADOS / "thumb_hash.parquet"
URL_THUMB = "https://i.ytimg.com/vi/{}/hqdefault.jpg"


def ler_limiar(caminho=CONFIG) -> int:
    """Bits de distância até os quais duas thumbnails são a mesma arte (decisão #1)."""
    return int(yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))["limiar_bits"])


def dhash(conteudo: bytes) -> int:
    """dHash de 64 bits: o miolo vertical (sem as faixas pretas do hqdefault) em 9x8 tons de
    cinza, um bit por par de vizinhos na horizontal."""
    imagem = Image.open(io.BytesIO(conteudo)).convert("L")
    largura, altura = imagem.size
    miolo = int(altura * 9 / 16)
    esquerda = (largura - miolo) // 2
    imagem = imagem.crop((esquerda, 0, esquerda + miolo, altura)).resize((9, 8), Image.LANCZOS)
    pixels = np.asarray(imagem, dtype=int)
    bits = (pixels[:, 1:] > pixels[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def distancia(a: int, b: int) -> int:
    """Bits diferentes entre dois dHash (distância de Hamming)."""
    return bin(a ^ b).count("1")


def _transitorio(e: BaseException) -> bool:
    """Vale tentar de novo? Rede, 429 e 5xx, sim; 404 e o resto, não (aula 4.2)."""
    if isinstance(e, httpx.TransportError):  # timeout, conexão caiu
        return True
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 500, 502, 503)


@retry(
    retry=retry_if_exception(_transitorio),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=30),
    reraise=True,
)
def _baixar(cliente: httpx.Client, url: str) -> bytes:
    resposta = cliente.get(url)
    resposta.raise_for_status()
    return resposta.content


def baixar_thumbnail(video_id: str, cliente: httpx.Client) -> bytes:
    """O hqdefault do vídeo (i.ytimg.com, sem cota da API)."""
    return _baixar(cliente, URL_THUMB.format(video_id))


def _calcular(video_ids, baixar) -> list[tuple[str, int | None, str | None]]:
    """(video_id, dHash, erro) de cada vídeo, em 6 threads."""

    def um(video_id):
        try:
            return video_id, dhash(baixar(video_id)), None
        except Exception as erro:  # rede, 404 ou imagem ilegível: vira aviso, nunca silêncio
            return video_id, None, sem_chave(f"{type(erro).__name__}: {erro}")

    with ThreadPoolExecutor(6) as executor:
        return list(executor.map(um, video_ids))


def hashes_das_thumbnails(video_ids, cache=CACHE, baixar=None) -> dict[str, int | None]:
    """dHash de cada vídeo, do cache ou baixando só o que falta. Uma thumbnail que falha vira
    None, com aviso, e fica fora do cache: a próxima exportação tenta de novo."""
    cache = Path(cache)
    salvos = (
        pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["video_id", "dhash"])
    )
    hashes = {v: int(h, 16) for v, h in zip(salvos["video_id"], salvos["dhash"], strict=True)}
    faltam = sorted(set(video_ids) - set(hashes))
    if baixar is not None:
        resultados = _calcular(faltam, baixar)
    else:  # um cliente só, com as conexões reaproveitadas entre as threads
        with httpx.Client(timeout=20) as cliente:
            resultados = _calcular(faltam, partial(baixar_thumbnail, cliente=cliente))
    novos = [(v, h) for v, h, _ in resultados if h is not None]
    falhas = [(v, e) for v, h, e in resultados if h is None]
    if novos:
        agora = datetime.now(timezone.utc)
        linhas = pd.DataFrame(
            {"video_id": [v for v, _ in novos], "dhash": [f"{h:016x}" for _, h in novos]}
        ).assign(hashed_at=agora)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporario = cache.with_suffix(".tmp")  # escrita interrompida nunca corrompe o cache
        pd.concat([salvos, linhas], ignore_index=True).to_parquet(temporario, index=False)
        os.replace(temporario, cache)
        hashes.update(novos)
    if falhas:
        exemplos = "; ".join(f"{v} ({e})" for v, e in falhas[:3])
        warnings.warn(
            f"{len(falhas)} thumbnail(s) sem dHash: {exemplos}. Esses posts ficam pela regra de "
            "título e duração (conservador) até a próxima exportação.",
            UserWarning,
            stacklevel=2,
        )
    return {v: hashes.get(v) for v in video_ids}


def chave_de_titulo(rotulados: pd.DataFrame) -> pd.Series:
    """O primeiro filtro: título normalizado + duração, em qualquer canal."""
    titulo = rotulados["title"].fillna("").str.lower().str.split().str.join(" ")
    return titulo + "|" + rotulados["duration_s"].astype(str)


def candidatos(rotulados: pd.DataFrame) -> list[str]:
    """Os vídeos que dividem a chave de título com outro: só eles precisam da thumbnail."""
    chave = chave_de_titulo(rotulados)
    return sorted(rotulados.loc[chave.duplicated(keep=False), "video_id"])


def grupo_de_duplicatas(rotulados: pd.DataFrame, hashes=None, limiar=None) -> pd.Series:
    """O video_id da primeira cópia publicada do mesmo criativo (o próprio, se não houver).

    Dentro de cada chave de título, em ordem de publicação, um post junta-se à cópia mais antiga
    com a mesma arte; se nenhuma tiver, ele abre um grupo novo. Um post sem thumbnail fica pela
    regra de título e duração (conservador): junta-se à cópia mais antiga da chave. Só ele: os
    outros comparam as artes só com os grupos de hash conhecido.
    """
    hashes = hashes or {}
    limiar = ler_limiar() if limiar is None else limiar
    chave = chave_de_titulo(rotulados)
    grupo = rotulados["video_id"].copy()
    ordem = rotulados.assign(_chave=chave).sort_values(["published_at", "video_id"])
    repetidos = ordem[ordem["_chave"].duplicated(keep=False)]
    sem_hash = int(repetidos["video_id"].map(hashes).isna().sum())
    for _, posts in repetidos.groupby("_chave", sort=False):
        primeiras = []  # (video_id, hash) de cada grupo aberto, do mais antigo ao mais novo
        for indice, video_id in zip(posts.index, posts["video_id"], strict=True):
            h = hashes.get(video_id)
            if h is None:
                alvo = primeiras[0][0] if primeiras else None
            else:
                mesma_arte = (
                    v for v, hv in primeiras if hv is not None and distancia(h, hv) <= limiar
                )
                alvo = next(mesma_arte, None)
            if alvo is None:
                primeiras.append((video_id, h))
                alvo = video_id
            grupo[indice] = alvo
    if sem_hash:
        warnings.warn(
            f"{sem_hash} post(s) com cópia de título e duração e sem thumbnail: ficam pela regra "
            "de título e duração (conservador).",
            UserWarning,
            stacklevel=2,
        )
    return grupo


def linhas_do_modelo(base: pd.DataFrame) -> pd.DataFrame:
    """As linhas de treino e de teste: só a primeira cópia de cada criativo. O histórico do canal
    usa a base inteira (paridade treino/uso)."""
    return base[base["post_id"] == base["duplicate_group"]]
