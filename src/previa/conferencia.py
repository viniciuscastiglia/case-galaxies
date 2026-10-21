"""Conferência de 50 classificações de Short (decisão #19; aula 1.5).

Não existe `isShort` na API: o coletor usa a regra documentada (até 60 s antes de 15/10/2024, até
180 s depois, +1 s de folga, e não horizontal). A aula pede conferir umas 50 classificações com os
próprios olhos e anotar a concordância no data card. A lista sai da tabela `videos` do DuckDB, que
não tem curtidas, só com os canais da base, e mistura Shorts com verticais que a regra recusou por
até 30 s: as bordas, onde a regra pode errar.

`uv run python -m previa.conferencia lista` escreve data/conferencia_shorts.csv, às cegas (só o
número e a URL), e o gabarito ao lado; o Vinicius abre cada URL e preenche `no_youtube_e_short`
com sim ou não. `uv run python -m previa.conferencia resultado` imprime a concordância por estrato.
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from previa.caminhos import CONFIGS
from previa.coletor import DADOS, MUDOU_PARA_3_MIN

ARQUIVO = DADOS / "conferencia_shorts.csv"
GABARITO = DADOS / "conferencia_shorts_gabarito.csv"
BORDA_S = 30  # segundos acima do limite da época que ainda contam como borda


def ler_config(caminho=CONFIGS / "conferencia.yaml") -> dict:
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def estrato(videos: pd.DataFrame) -> pd.Series:
    """O estrato de cada vídeo na conferência; NA para quem não entra (horizontais e longos)."""
    depois = videos["published_at"] >= MUDOU_PARA_3_MIN
    limite = pd.Series(np.where(depois, 180, 60) + 1, index=videos.index)  # a folga do coletor
    dur = videos["duration_s"]
    short = videos["is_short"].astype(bool)
    vertical = videos["vertical"].fillna(False).astype(bool)
    borda = ~short & vertical & (dur > limite) & (dur <= limite + BORDA_S)
    s = pd.Series(pd.NA, index=videos.index, dtype="object")
    s.loc[short & ~depois] = "short_antes_ate_60s"
    s.loc[short & depois & (dur <= 61)] = "short_depois_ate_60s"
    s.loc[short & depois & (dur > 61)] = "short_depois_61_a_180s"
    s.loc[borda] = "nao_short_vertical_na_borda"
    return s


def sortear_lista(videos: pd.DataFrame, cotas: dict[str, int], seed=0) -> pd.DataFrame:
    """Sorteia a cota de cada estrato (ou todos, se faltar) e embaralha a ordem da lista."""
    rng = np.random.default_rng(seed)
    v = videos.assign(estrato=estrato(videos)).dropna(subset=["estrato"])
    v = v.sort_values("video_id")  # ordem fixa antes do sorteio
    partes = []
    for nome, k in cotas.items():
        g = v[v["estrato"] == nome]
        partes.append(g.iloc[np.sort(rng.choice(len(g), size=min(k, len(g)), replace=False))])
    lista = pd.concat(partes)
    lista = lista.iloc[rng.permutation(len(lista))].reset_index(
        drop=True
    )  # o estrato não aparece na ordem
    return pd.DataFrame(
        {
            "n": np.arange(1, len(lista) + 1),
            "post_id": lista["video_id"],
            "url": "https://www.youtube.com/shorts/" + lista["video_id"],
            "estrato": lista["estrato"],
            "regra": np.where(lista["is_short"].astype(bool), "Short", "não Short"),
            "published_at": lista["published_at"].dt.date.astype(str),
            "duration_s": lista["duration_s"],
            "no_youtube_e_short": "",
            "obs": "",
        }
    )


def _resposta(x) -> float:
    x = str(x).strip().lower()
    if x in {"sim", "s"}:
        return 1.0
    if x in {"não", "nao", "n"}:
        return 0.0
    if x in {"", "nan"}:
        return np.nan  # ainda não conferido
    raise ValueError(f"resposta não reconhecida: {x!r} (use sim ou não)")


def concordancia(lista: pd.DataFrame) -> pd.DataFrame:
    """Conferidos, concordâncias e taxa por estrato e no total; respostas em branco ficam fora."""
    d = lista.assign(yt=lista["no_youtube_e_short"].map(_resposta)).dropna(subset=["yt"])
    d = d.assign(concorda=(d["regra"] == "Short") == (d["yt"] == 1.0))
    tab = d.groupby("estrato").agg(conferidos=("concorda", "size"), concordam=("concorda", "sum"))
    tab.loc["total"] = tab.sum()
    tab["taxa"] = tab["concordam"] / tab["conferidos"]
    return tab


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m previa.conferencia")
    parser.add_argument("acao", choices=["lista", "resultado"])
    args = parser.parse_args(argv)
    if args.acao == "lista":
        if ARQUIVO.exists():
            print(f"{ARQUIVO} já existe; apague-o para sortear de novo.", file=sys.stderr)
            return 1
        cfg = ler_config()
        canais = pd.read_parquet(DADOS / "base_minima.parquet", columns=["channel_id"])
        with duckdb.connect(str(DADOS / "shorts.duckdb"), read_only=True) as con:
            videos = con.execute(
                "SELECT video_id, channel_id, published_at, duration_s, vertical, is_short "
                "FROM videos"
            ).df()
        videos = videos[videos["channel_id"].isin(canais["channel_id"].unique())]
        lista = sortear_lista(videos, cfg["estratos"], seed=cfg["seed"])
        lista[["n", "url", "no_youtube_e_short", "obs"]].to_csv(ARQUIVO, index=False)
        lista.drop(columns=["no_youtube_e_short", "obs"]).to_csv(GABARITO, index=False)
        print(f"{len(lista)} vídeos -> {ARQUIVO} (gabarito em {GABARITO.name})")
        print(lista["estrato"].value_counts().to_string())
        return 0
    preenchida = pd.read_csv(ARQUIVO, dtype=str, keep_default_na=False)
    preenchida["n"] = preenchida["n"].astype(int)
    lista = pd.read_csv(GABARITO).merge(
        preenchida[["n", "no_youtube_e_short"]], on="n", validate="one_to_one"
    )
    print(concordancia(lista).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
