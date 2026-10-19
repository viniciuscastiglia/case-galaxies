"""O histórico do canal na hora da previsão (Tool 2; aulas 1.5, "No projeto Galaxies", e 6.3).

A previsão de um Short novo precisa dos posts anteriores do canal (datas e curtidas), que viram as
features da M3 pela mesma função do treino. Duas fontes:

- `historico_api`: o caminho do coletor (playlist de uploads, `videos.list`, `parse_video`), com
  cache por canal e por dia (`data/cache/historico_api/`), para canais dentro ou fora da base;
- `historico_congelado`: o histórico salvo do golden set (#4), para testes e para a demo offline.

As curtidas vêm da data da consulta: para os posts maduros (os únicos que entram no histórico, com
a lacuna de 30 dias), é o mesmo valor estabilizado que o treino viu (medição da maturação, 19/10).
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from previa.coletor import DADOS, buscar_videos, listar_ids, parse_video, playlist_uploads

CACHE = DADOS / "cache" / "historico_api"
COLUNAS = ["post_id", "channel_id", "published_at", "y", "caption", "duration_s"]


def _para_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Linhas do `parse_video` -> posts do canal, só Shorts, com y = log1p(curtidas)."""
    if df.empty:
        return pd.DataFrame(columns=COLUNAS)
    s = df[df["is_short"].astype(bool)].copy()
    s["post_id"] = s["video_id"]
    s["published_at"] = pd.to_datetime(s["published_at"], utc=True)
    s["y"] = np.log1p(pd.to_numeric(s["likes"], errors="coerce").astype(float))  # oculta: NaN
    s["caption"] = s["title"].fillna("") + "\n" + s["description"].fillna("")
    return s[COLUNAS].sort_values("published_at", kind="stable").reset_index(drop=True)


def posts_de_itens(itens: list[dict], coletado_em: datetime) -> pd.DataFrame:
    return _para_posts(pd.DataFrame([parse_video(it, coletado_em) for it in itens]))


def historico_api(cli, channel_id: str, agora: datetime | None = None, pasta=CACHE):
    """Os Shorts do canal pela API, com cache por dia (o custo é de 1 unidade por 50 vídeos)."""
    agora = agora or datetime.now(timezone.utc)
    arquivo = Path(pasta) / f"{channel_id}_{agora.date().isoformat()}.parquet"
    if arquivo.exists():
        return pd.read_parquet(arquivo)
    itens = buscar_videos(cli, listar_ids(cli, playlist_uploads(channel_id)))
    posts = posts_de_itens(itens, agora)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    posts.to_parquet(arquivo, index=False)
    return posts


def historico_congelado(channel_id: str, arquivo=DADOS / "golden" / "historico_v1.parquet"):
    h = pd.read_parquet(arquivo)
    return _para_posts(h[h["channel_id"] == channel_id])
