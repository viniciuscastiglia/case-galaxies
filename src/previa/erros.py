"""Análise de erros (aula 2.6; insumo da M9): onde o modelo erra, e os piores erros para assistir.

O resíduo é `e = y - p`, em log1p(curtidas): positivo quer dizer que o modelo subestimou, e o fator
e^|e| diz por quantas vezes errou. Roda nas previsões fora da amostra do walk-forward de 2025 de um
run do MLflow: o teste fica trancado até a H1. Os cortes (setor, tipo e tamanho do canal, duração,
histórico e horário) geram hipóteses, não conclusões (2.6, erro comum 6), por isso cada grupo vem
com o nº de posts e de canais. Os piores saem com o link, para o Vinicius assistir e classificar
nas categorias da 2.6 (M9; erro comum 7: dado quebrado e viral genuíno pedem ações opostas).

Uso: uv run python -m previa.erros RUN_ID
"""

import sys

import mlflow
import numpy as np
import pandas as pd

from previa.coletor import DADOS
from previa.dados import carregar_base
from previa.experimento import EXPERIMENTO, hash_base, ler_duracoes, preparar
from previa.splits import ler_avaliacao

CATEGORIAS = (
    "viral de verdade",
    "canal novo",
    "problema de dado",
    "alvo imaturo",
    "impulsionamento",
)
CORTES = {  # coluna -> (bordas [a, b), rótulos); sem bordas, o próprio valor é o grupo
    "sector": (None, None),
    "tipo": (None, None),
    "faixa": (None, None),
    "duracao_s": ([0, 20, 40, 60, np.inf], ["< 20 s", "20 a 39 s", "40 a 59 s", "60 s ou mais"]),
    "hist_n": ([0, 10, 30, np.inf], ["< 10 maduros", "10 a 29", "30+"]),
    "hora_local": ([0, 12, 18, 24], ["manhã", "tarde", "noite"]),
}


def residuos(t: pd.DataFrame) -> pd.DataFrame:
    """`residuo` (> 0: subestimou), `fator` (e^|e|) e `direcao`, a partir de `y` e `p`."""
    e = t["y"] - t["p"]
    direcao = np.where(e > 0, "subestimou", np.where(e < 0, "superestimou", "exato"))
    return t.assign(residuo=e, fator=np.exp(e.abs()), direcao=direcao)


def por_corte(t: pd.DataFrame, corte: str) -> pd.DataFrame:
    """MAE e viés (a média do resíduo) por grupo, com o nº de posts e de canais de cada um."""
    g = t.groupby(corte, observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "canais": g["channel_id"].nunique(),
            "mae": g["residuo"].apply(lambda r: r.abs().mean()),
            "vies": g["residuo"].mean(),
        }
    )


def agrupar(t: pd.DataFrame, col: str) -> pd.Series:
    """O grupo de cada linha no corte `col`, pelas bordas de `CORTES` (intervalos [a, b))."""
    bordas, rotulos = CORTES[col]
    return t[col] if bordas is None else pd.cut(t[col], bordas, labels=rotulos, right=False)


def piores(t: pd.DataFrame, n=20) -> pd.DataFrame:
    """Os `n` maiores |resíduo|, com o link e as colunas `categoria` e `nota` para preencher."""
    sel = t.loc[t["residuo"].abs().sort_values(ascending=False, kind="stable").index[:n]]
    return sel.assign(
        url="https://www.youtube.com/shorts/" + sel["post_id"],
        curtidas_reais=np.expm1(sel["y"]).round(),
        curtidas_previstas=np.expm1(sel["p"]).round(),
        categoria="",
        nota="",
    )


def tabela_do_run(run_id: str) -> pd.DataFrame:
    """As previsões do walk-forward do run, com y, o canal e as colunas dos cortes."""
    mlflow.set_experiment(EXPERIMENTO)
    base = carregar_base()  # sem o período de teste
    if (h := mlflow.get_run(run_id).data.tags.get("hash_base")) != hash_base(base):
        raise ValueError(f"o run usou a base {h}, e a atual é {hash_base(base)}: rode-o de novo")
    prev = mlflow.load_table("previsoes.json", run_ids=[run_id])
    prev = prev[prev["avaliacao"] == "walk_forward"]
    d = preparar(base, ler_duracoes(), ler_avaliacao()).set_index("post_id")
    cols = ["channel_id", "published_at", "y", "sector", "duracao_s", "hora_local", "hist_n"]
    t = d.loc[prev["post_id"], cols].reset_index().assign(p=prev["p"].to_numpy())
    canais = pd.read_csv(DADOS / "canais.csv")
    # um canal pode ter 2 linhas (o handle antigo fica como excluído): só os incluídos
    canais = canais[canais["incluido"] == "sim"].set_index("channel_id")
    t["canal"] = t["channel_id"].map(canais["nome"])
    t["tipo"] = t["channel_id"].map(canais["tipo"])
    t["faixa"] = t["channel_id"].map(canais["faixa_inscritos"])
    return residuos(t)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("uso: python -m previa.erros RUN_ID", file=sys.stderr)
        return 2
    run_id = argv[0]
    t = tabela_do_run(run_id)
    cortes = []
    for col in CORTES:
        r = por_corte(t.assign(grupo=agrupar(t, col)), "grupo")
        cortes.append(r.rename_axis("grupo").reset_index().assign(corte=col))
    destino = DADOS / "erros"
    destino.mkdir(exist_ok=True)
    cols = ["corte", "grupo", "n", "canais", "mae", "vies"]
    pd.concat(cortes)[cols].to_csv(destino / f"cortes_{run_id[:8]}.csv", index=False)
    lista = piores(t)[
        ["url", "canal", "sector", "tipo", "published_at", "duracao_s", "hist_n"]
        + ["curtidas_reais", "curtidas_previstas", "fator", "direcao", "categoria", "nota"]
    ]
    lista.to_csv(destino / f"piores_{run_id[:8]}.csv", index=False)
    print(f"{len(t)} previsões do walk-forward -> {destino}/cortes_ e piores_{run_id[:8]}.csv")
    print(f"categorias da 2.6: {', '.join(CATEGORIAS)}")
    print(f"comando: uv run python -m previa.erros {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
