"""Medição da maturação das curtidas (decisões #11 e #21; aulas 1.7 e 3.5, exercício "quanto
custa a lacuna", passo 4).

Com os snapshots diários do coletor (desde 05/10), cada Short jovem tem as curtidas medidas em
idades diferentes. Para cada vídeo, o crescimento entre o primeiro e o último snapshot, agrupado
pela idade no primeiro: se um Short com 30 dias ainda cresce pouco, a lacuna de 30 dias do
histórico (princípio 2) e o rótulo "primeiro snapshot com 30+ dias" (1.5) estão bem escolhidos.

`uv run python -m previa.maturacao` imprime a tabela.
"""

import sys

import duckdb
import pandas as pd

from previa.coletor import DADOS

FAIXAS = [0, 3, 7, 14, 21, 30, 45, 60]


def crescimento_por_idade(snap: pd.DataFrame, faixas=FAIXAS, min_intervalo=10) -> pd.DataFrame:
    """Crescimento relativo das curtidas (último / primeiro − 1) por faixa de idade no primeiro.

    Só vídeos com pelo menos `min_intervalo` dias entre o primeiro e o último snapshot e com
    curtidas visíveis nos dois.
    """
    s = snap.dropna(subset=["likes"]).sort_values(["video_id", "age_days"])
    g = s.groupby("video_id").agg(
        idade=("age_days", "first"),
        idade_fim=("age_days", "last"),
        ini=("likes", "first"),
        fim=("likes", "last"),
    )
    g = g[(g["idade_fim"] - g["idade"] >= min_intervalo) & (g["ini"] > 0)]
    g["crescimento"] = g["fim"] / g["ini"] - 1
    rotulos = [f"{a}-{b}" for a, b in zip(faixas[:-1], faixas[1:], strict=True)]
    g["faixa"] = pd.cut(g["idade"], faixas, labels=rotulos, right=False)
    t = g.groupby("faixa", observed=True)["crescimento"].agg(
        videos="size",
        mediana="median",
        p75=lambda x: x.quantile(0.75),
    )
    t["dias_medios"] = g.groupby("faixa", observed=True).apply(
        lambda x: (x["idade_fim"] - x["idade"]).mean(), include_groups=False
    )
    t.index = t.index.astype(str)
    return t.round(4)


def main() -> int:
    with duckdb.connect(str(DADOS / "shorts.duckdb"), read_only=True) as con:
        snap = con.execute("SELECT video_id, age_days, likes FROM snapshots").df()
    print(crescimento_por_idade(snap).to_string())
    print(f"\n{snap['video_id'].nunique()} vídeos com snapshot, {len(snap)} snapshots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
