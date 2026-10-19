"""Golden set congelado (decisão #4; aulas 6.6, camada 4, e 2.3): canais que NÃO estão na base.

As reservas que passaram na revisão de 08/10 e ficaram de fora só pelo teto de 70 canais. Por
estrato (2 marcas e 1 criador por setor), sorteio com seed fixa entre os canais com 5+ Shorts
maduros (30+ dias, curtidas visíveis) publicados no período de teste (01/04 a 14/09/2026); de cada
canal, os 5 mais recentes. A Salon Line sai: o canal dela está na base (#4).

Uma coleta só, pela API (o mesmo caminho do coletor, 1.5): os Shorts escolhidos e o histórico
inteiro dos canais, que a previsão usa como contexto. Os arquivos ficam em `data/golden/`, fora do
git, e são apagados com os dados brutos em 30/10.

`uv run python -m previa.golden coletar` (precisa da YOUTUBE_API_KEY no ambiente).
"""

import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from previa.coletor import DADOS, sem_chave

ESTRATOS = {
    ("beleza", "marca"): 2,
    ("beleza", "criador"): 1,
    ("automotivo", "marca"): 2,
    ("automotivo", "criador"): 1,
}
EXCLUIR = {"UCnjK9YWSbd0AAjHhM3vAu7A"}  # @salonlinebrasil: a Salon Line está na base (#4)
PASTA = DADOS / "golden"


def candidatos(canais: pd.DataFrame, canais_da_base: set[str]) -> pd.DataFrame:
    """As reservas revisadas, de fora da base."""
    revisada = canais["motivo"].fillna("").str.startswith("reserva:")
    fora = ~canais["channel_id"].isin(canais_da_base | EXCLUIR)
    return canais[revisada & fora].drop_duplicates("channel_id")


def escolher_shorts(shorts: pd.DataFrame, inicio, fim, idade_min=30, n=5) -> pd.DataFrame:
    """Os n Shorts mais recentes do período, maduros e com curtidas visíveis."""
    ok = (
        shorts["is_short"].astype(bool)
        & shorts["likes"].notna()
        & (shorts["published_at"] >= inicio)
        & (shorts["published_at"] < fim)
        & (shorts["age_days"] >= idade_min)
    )
    return shorts[ok].sort_values("published_at").tail(n)


def sortear(cands, estratos, buscar_shorts, inicio, fim, seed=0, n=5):
    """Percorre cada estrato em ordem sorteada até achar a cota de canais elegíveis."""
    rng = np.random.default_rng(seed)
    escolhidos, log = {}, []
    for (setor, tipo), cota in estratos.items():
        g = cands[(cands["setor"] == setor) & (cands["tipo"] == tipo)].sort_values("channel_id")
        achados = 0
        for cid in g["channel_id"].to_numpy()[rng.permutation(len(g))]:
            if achados == cota:
                break
            sel = escolher_shorts(buscar_shorts(cid), inicio, fim, n=n)
            log.append(f"{setor}/{tipo} {cid}: {len(sel)} Shorts elegíveis")
            if len(sel) == n:
                escolhidos[cid] = sel
                achados += 1
        if achados < cota:
            log.append(f"{setor}/{tipo}: só {achados} de {cota} canais elegíveis")
    return escolhidos, log


def main(argv=None) -> int:
    from googleapiclient.discovery import build

    from previa.coletor import (
        Cliente,
        buscar_videos,
        listar_ids,
        parse_video,
        playlist_uploads,
        preferir_ipv4,
    )
    from previa.splits import ler_avaliacao

    if (argv or sys.argv[1:] or [""])[0] != "coletar":
        print("uso: python -m previa.golden coletar", file=sys.stderr)
        return 2
    chave = os.environ.get("YOUTUBE_API_KEY")
    if not chave:
        print("YOUTUBE_API_KEY não definida: carregue o .env.", file=sys.stderr)
        return 2
    preferir_ipv4()
    cli = Cliente(build("youtube", "v3", developerKey=chave, cache_discovery=False))
    agora = datetime.now(timezone.utc)
    historicos = {}

    def buscar_shorts(cid):
        itens = buscar_videos(cli, listar_ids(cli, playlist_uploads(cid)))
        df = pd.DataFrame([parse_video(it, agora) for it in itens])
        historicos[cid] = df
        return df

    cfg = ler_avaliacao()
    base = pd.read_parquet(DADOS / "base_minima.parquet", columns=["channel_id"])
    cands = candidatos(pd.read_csv(DADOS / "canais.csv"), set(base["channel_id"]))
    try:
        escolhidos, log = sortear(
            cands, ESTRATOS, buscar_shorts, cfg["teste_inicio"], cfg["teste_fim"]
        )
    except Exception:
        print(sem_chave(traceback.format_exc()).replace(chave, "***"), file=sys.stderr)
        return 1
    print("\n".join(log))
    PASTA.mkdir(parents=True, exist_ok=True)
    meta = cands.set_index("channel_id")[["handle", "setor", "tipo"]]
    golden = pd.concat(escolhidos.values()).join(meta, on="channel_id")
    historico = pd.concat([historicos[c] for c in escolhidos]).join(meta, on="channel_id")
    golden.to_parquet(PASTA / "golden_v1.parquet", index=False)
    historico.to_parquet(PASTA / "historico_v1.parquet", index=False)
    print(f"{len(golden)} Shorts de {len(escolhidos)} canais -> {PASTA}")
    print(f"unidades da API hoje: {cli.gasto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
