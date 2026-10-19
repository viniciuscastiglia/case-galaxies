"""Thumbnails da base (decisão #12; aula 4.2): a capa de cada Short, sem cota da API.

As URLs são as que o `videos.list` devolve em `snippet.thumbnails` (i.ytimg.com/vi/<id>/...); o
coletor não as guardou, e o padrão é fixo. Baixa o maior tamanho que sempre existe e basta para o
SigLIP 2, que redimensiona tudo para 224 × 224 (`hqdefault`, 480 × 360), e cai para os menores se
ele faltar. Uma vez só (cache por arquivo, escrita atômica), retry só em erro transitório e pausa
entre vídeos; um 429 que persiste para o lote (`BloqueioDoYouTube`), porque insistir piora.

As imagens ficam em data/thumbs/ (fora do git) só até a extração das features (aula 4.2: o mínimo
necessário, e apagar depois). A capa é a da data da coleta, não necessariamente a da publicação
(erro comum 6; vai para o data card).

`uv run python -m previa.thumbs` baixa as primeiras cópias da base, em ordem sorteada (seed 0).
"""

import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from previa.coletor import DADOS

ORDEM = ["hqdefault", "mqdefault", "default"]  # 480×360, 320×180, 120×90
PASTA = DADOS / "thumbs"


class BloqueioDoYouTube(RuntimeError):
    """Pare o lote: o i.ytimg.com respondeu 429 mesmo depois das novas tentativas."""


def _transitorio(e: BaseException) -> bool:
    if isinstance(e, httpx.TransportError):  # timeout, conexão caiu
        return True
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 500, 502, 503)


def _baixar(cliente: httpx.Client, url: str, espera: float) -> bytes:
    for tentativa in Retrying(
        retry=retry_if_exception(_transitorio),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=espera, max=30),
        reraise=True,
    ):
        with tentativa:
            resp = cliente.get(url)
            resp.raise_for_status()
            return resp.content
    raise AssertionError("inalcançável")


def baixar_thumbnail(video_id: str, pasta: Path, cliente: httpx.Client, espera=1) -> Path | None:
    """A thumbnail de um Short em `pasta/<video_id>.jpg`; None se nenhum tamanho existir."""
    destino = Path(pasta) / f"{video_id}.jpg"
    if destino.exists():  # cache: nunca baixe duas vezes
        return destino
    for nome in ORDEM:
        try:
            conteudo = _baixar(cliente, f"https://i.ytimg.com/vi/{video_id}/{nome}.jpg", espera)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                continue  # este tamanho não existe: tente o próximo
            raise
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(".tmp")
        tmp.write_bytes(conteudo)
        os.replace(tmp, destino)  # atômico: nunca fica um JPEG pela metade
        return destino
    return None


def baixar_todas(ids, pasta: Path, cliente: httpx.Client, pausa_s=0.2, espera=1, log=print):
    """Baixa as thumbnails de `ids`, em ordem; falhas permanentes vão para `_falhas.txt`."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    resumo = {"ok": 0, "pulados": 0, "falhas": 0}
    for i, vid in enumerate(ids, start=1):
        if (pasta / f"{vid}.jpg").exists():
            resumo["pulados"] += 1
            continue
        motivo = None
        try:
            if baixar_thumbnail(vid, pasta, cliente, espera) is None:
                motivo = "nenhum tamanho existe"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise BloqueioDoYouTube(f"429 em {vid} depois das novas tentativas") from e
            motivo = f"HTTP {e.response.status_code}"
        except httpx.TransportError as e:
            motivo = f"rede: {e!r}"[:200]
        if motivo:
            with (pasta / "_falhas.txt").open("a", encoding="utf-8") as f:
                f.write(f"{vid}\t{motivo}\n")
            resumo["falhas"] += 1
        else:
            resumo["ok"] += 1
        if i % 500 == 0:
            log(f"{i}/{len(ids)} {resumo}")
        time.sleep(pausa_s)
    return resumo


def remover_barras(img: np.ndarray, limiar: int = 24, fracao: float = 0.05) -> np.ndarray:
    """Recorta faixas pretas (pillarbox/letterbox) em volta do conteúdo (aula 4.2)."""
    claro = img.max(axis=2) > limiar  # pixel "com conteúdo"
    cols = np.where(claro.mean(axis=0) > fracao)[0]  # tolera ruído de JPEG na faixa
    lins = np.where(claro.mean(axis=1) > fracao)[0]
    if len(cols) == 0:  # imagem toda preta: devolve como veio
        return img
    return img[lins[0] : lins[-1] + 1, cols[0] : cols[-1] + 1]


def recortar_9_16(img: np.ndarray) -> np.ndarray:
    """A faixa central 9:16 da thumbnail de um Short.

    Nas thumbnails dos Shorts da base (480 × 360), o quadro vertical vem no centro e as laterais
    trazem uma cópia desfocada dele, não faixas pretas: o `remover_barras` da aula não as acha, e
    sem o recorte o encoder veria o conteúdo duas vezes. Conferido no piloto de 20 (19/10).
    """
    altura, largura = img.shape[:2]
    faixa = min(round(altura * 9 / 16), largura)
    inicio = (largura - faixa) // 2
    return img[:, inicio : inicio + faixa]


def main() -> int:
    base = pd.read_parquet(
        DADOS / "base_minima.parquet", columns=["post_id", "duplicate_group", "published_at"]
    )  # sem curtidas
    ids = base.loc[base["duplicate_group"] == base["post_id"], "post_id"].sort_values().to_numpy()
    ids = np.random.default_rng(0).permutation(ids).tolist()  # parar no meio deixa uma amostra
    print(f"{len(ids)} Shorts -> {PASTA}", flush=True)
    with httpx.Client(timeout=10) as cliente:
        try:
            resumo = baixar_todas(ids, PASTA, cliente, log=lambda m: print(m, flush=True))
        except BloqueioDoYouTube as e:
            print(f"parou o lote: {e}", file=sys.stderr)
            return 1
    print(f"resumo: {resumo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
