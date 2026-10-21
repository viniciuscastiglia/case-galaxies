"""Medição da maturação das curtidas (decisões #11 e #21; aulas 1.7 e 3.5, exercício "quanto
custa a lacuna", passo 4).

Com os snapshots diários do coletor (desde 05/10), cada Short jovem tem as curtidas medidas em
idades diferentes. Para cada vídeo, o crescimento entre o primeiro e o último snapshot, agrupado
pela idade no primeiro: se um Short com 30 dias ainda cresce pouco, a lacuna de 30 dias do
histórico (princípio 2) e o rótulo "primeiro snapshot com 30+ dias" (1.5) estão bem escolhidos.

`uv run python -m previa.maturacao` imprime a tabela; `cauda` estende as faixas até anos, e
`recoletar` tira o segundo snapshot dos Shorts velhos, que só tinham um (#21).
"""

import json
import math
import os
import sys
import traceback

import duckdb
import pandas as pd

from previa.coletor import (
    DADOS,
    Cliente,
    CotaEsgotada,
    abrir_base,
    preferir_ipv4,
    renovar_vencidos,
    sem_chave,
)

FAIXAS = [0, 3, 7, 14, 21, 30, 45, 60]
# #21: os rótulos do treino têm de 188 a 4.528 dias; a última faixa não tem teto
FAIXAS_CAUDA = [*FAIXAS, 180, 365, 730, math.inf]


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
    pares = zip(faixas[:-1], faixas[1:], strict=True)
    rotulos = [f"{a}+" if b == math.inf else f"{a}-{b}" for a, b in pares]
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


def recoletar() -> int:
    """Um snapshot novo de todo Short sem snapshot nas últimas 24 h (#21).

    É a renovação dos 30 dias do coletor com prazo de 1 dia: 1 unidade a cada 50 Shorts, e os
    que sumiram da API são apagados (aula 1.8). Os rótulos não mudam: cada um é o primeiro
    snapshot com 30+ dias.
    """
    from googleapiclient.discovery import build

    chave = os.environ.get("YOUTUBE_API_KEY")
    if not chave:
        print("YOUTUBE_API_KEY não definida: carregue o .env antes.", file=sys.stderr)
        return 2
    preferir_ipv4()
    cli = Cliente(build("youtube", "v3", developerKey=chave, cache_discovery=False))
    con = abrir_base()
    try:
        resumo = renovar_vencidos(cli, con, prazo_dias=1)
    except CotaEsgotada as e:
        print(f"parou por cota ({e}): nada foi gravado; rode de novo amanhã", file=sys.stderr)
        return 3
    except Exception:  # sem a chave na mensagem (CLAUDE.md)
        print(sem_chave(traceback.format_exc()), file=sys.stderr)
        return 1
    finally:
        con.close()
    print(json.dumps(resumo))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["recoletar"]:
        return recoletar()
    faixas = FAIXAS_CAUDA if argv == ["cauda"] else FAIXAS
    with duckdb.connect(str(DADOS / "shorts.duckdb"), read_only=True) as con:
        snap = con.execute("SELECT video_id, age_days, likes FROM snapshots").df()
    print(crescimento_por_idade(snap, faixas=faixas).to_string())
    print(f"\n{snap['video_id'].nunique()} vídeos com snapshot, {len(snap)} snapshots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
