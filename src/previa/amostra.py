"""Amostra multimodal (decisão #12 de 18/10/2026; aulas 4.1, 4.2 e 4.8).

A regra foi escrita antes de olhar qualquer resultado e é cega às curtidas: por canal, 20 Shorts
sorteados com seed 0 entre as primeiras cópias, divididos entre os blocos treino, calibração e
teste na proporção do que o canal tem em cada um (gap e pré-01/07/2021 ficam fora). A amostra lê
só ids, canal, data, grupo de duplicata e setor: as curtidas do teste nunca são lidas, então o
teste não é aberto e nada vai para data/logs/teste_aberto.log.

`ordem` é a ordem de processamento, também sorteada: se a extração parar no meio (cota do
Gemini, ritmo do yt-dlp), o que já existe é uma subamostra aleatória, e não os primeiros canais.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from previa.coletor import DADOS
from previa.splits import ler_avaliacao, split_treino_cal_teste

CONFIGS = Path("configs")
COLUNAS = ["post_id", "channel_id", "published_at", "duplicate_group", "sector"]
BLOCOS = ("treino", "cal", "teste")


def ler_config(caminho=CONFIGS / "multimodal.yaml") -> dict:
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def alocar(contagens: dict[str, int], n: int) -> dict[str, int]:
    """Divide n entre os blocos na proporção das contagens (maiores restos); nunca mais que há."""
    total = sum(contagens.values())
    if total <= n:
        return dict(contagens)
    cotas = {b: n * c / total for b, c in contagens.items()}
    saida = {b: int(np.floor(q)) for b, q in cotas.items()}
    falta = n - sum(saida.values())
    # sorted é estável: nos restos empatados, vale a ordem treino, cal, teste
    for b in sorted(cotas, key=lambda b: cotas[b] - saida[b], reverse=True)[:falta]:
        saida[b] += 1
    return saida


def sortear_amostra(df, cfg_avaliacao, por_canal=20, seed=0) -> pd.DataFrame:
    """A amostra da #12: uma linha por Short, com o bloco de tempo e a ordem de processamento."""
    d = df.loc[df["duplicate_group"] == df["post_id"], COLUNAS]
    treino, cal, teste = split_treino_cal_teste(d, cfg_avaliacao)
    bloco = pd.Series(pd.NA, index=d.index, dtype="string")
    for nome, idx in zip(BLOCOS, (treino, cal, teste), strict=True):
        bloco.loc[idx] = nome
    d = d.assign(bloco=bloco).dropna(subset=["bloco"])

    rng = np.random.default_rng(seed)
    partes = []
    ordenado = d.sort_values(["channel_id", "published_at", "post_id"])
    for _, g in ordenado.groupby("channel_id", sort=True):
        contagens = {b: int((g["bloco"] == b).sum()) for b in BLOCOS if (g["bloco"] == b).any()}
        for b, k in alocar(contagens, por_canal).items():
            gb = g[g["bloco"] == b]
            partes.append(gb.iloc[np.sort(rng.choice(len(gb), size=k, replace=False))])
    amostra = pd.concat(partes).reset_index(drop=True)
    amostra["ordem"] = rng.permutation(len(amostra))
    return amostra


def main() -> int:
    base = pd.read_parquet(DADOS / "base_minima.parquet", columns=COLUNAS)  # sem curtidas
    amostra = sortear_amostra(base, ler_avaliacao(), **ler_config()["amostra"])
    with duckdb.connect(str(DADOS / "shorts.duckdb"), read_only=True) as con:
        duracoes = con.execute("SELECT video_id AS post_id, duration_s FROM videos").df()
    amostra = amostra.merge(duracoes, on="post_id", how="left", validate="one_to_one")
    if amostra["duration_s"].isna().any():
        print(f"aviso: {amostra['duration_s'].isna().sum()} Shorts sem duração", file=sys.stderr)
    destino = DADOS / "amostra_multimodal.csv"
    amostra.to_csv(destino, index=False)
    print(f"{len(amostra)} Shorts em {amostra['channel_id'].nunique()} canais -> {destino}")
    print(amostra.groupby(["sector", "bloco"]).size().unstack(fill_value=0).to_string())
    print(f"duração total: {amostra['duration_s'].sum() / 3600:.1f} h de vídeo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
