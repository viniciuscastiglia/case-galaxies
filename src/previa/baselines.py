"""Baselines do canal e a leitura da M2 na base de desenvolvimento (aulas 2.1, 2.2 e 2.4).

Os baselines, todos nas mesmas linhas (2.1, regra 3):
- `mediana_madura`: a mediana do canal só com posts publicados ao menos 30 dias antes (2.2;
  princípio 2). É a régua honesta, e provisória até a M3 (o features.py a substitui);
- `mediana_com_imaturos`: a da 2.1, `shift(1).expanding()`, que inclui posts ainda imaturos.
  Serve só para comparar na leitura; nunca vira feature;
- `media_ultimos_k`: a média dos últimos k posts maduros, com o k escolhido no walk-forward de
  2025 (a 2.1 pede o baseline mais forte que você conseguir montar, auditoria F25);
- `mediana_janela`: a mediana dos posts maduros dos 90 dias antes do limite de maturidade, com os
  10 últimos maduros quando a janela tem poucos (o `historico_recente` da 6.1, só com maduros).

A leitura roda nos testes do walk-forward de desenvolvimento (2025): o teste fica trancado e a
calibração não é tocada. `uv run python -m previa.baselines janela` imprime a tabela.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from previa.caminhos import CONFIGS
from previa.dados import carregar_base
from previa.metricas import (
    ic_par_a_par,
    ic_par_a_par_peso_posts,
    pairwise_janela,
    par_a_par_peso_igual,
    par_a_par_peso_posts,
    por_canal,
    spearman_intra_canal,
)
from previa.splits import folds_de_desenvolvimento, ler_avaliacao

BASELINES = ["p_madura", "p_com_imaturos"]
MIN_NA_JANELA = 3  # menos que isso na janela de 90 dias: cai para os 10 últimos (6.1)
ULTIMOS_DO_FALLBACK = 10


def ler_config(caminho=CONFIGS / "baselines.yaml") -> dict:
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def alvo(df) -> pd.Series:
    """y = log1p(curtidas); NaN onde a contagem falta."""
    return np.log1p(df["metric_target"].astype(float))


def mediana_madura(df, dias=30, min_posts=3) -> pd.Series:
    """Mediana de y dos posts do canal publicados até `dias` antes de cada post (inclusive).

    Como na 2.2: com os posts do canal em ordem de data, a mediana acumulada até o último post
    maduro (`np.searchsorted`). Posts sem y não entram na mediana nem na contagem; com menos de
    `min_posts` posts maduros com y, o valor é NaN (cold start, 2.1).
    """
    saida = pd.Series(np.nan, index=df.index)
    ordenado = df.sort_values("published_at", kind="stable")
    for _, g in ordenado.groupby("channel_id", sort=False):
        datas = g["published_at"].dt.tz_convert(None).to_numpy()
        acumulada = g["y"].expanding().median().to_numpy()  # ignora os NaN
        com_y = g["y"].notna().cumsum().to_numpy()
        ultimo = np.searchsorted(datas, datas - np.timedelta64(dias, "D"), side="right") - 1
        k = np.maximum(ultimo, 0)
        ok = (ultimo >= 0) & (com_y[k] >= min_posts)
        saida.loc[g.index] = np.where(ok, acumulada[k], np.nan)
    return saida


def _maduros_do_canal(df, dias):
    """Por canal, em ordem de data: (índice, y dos posts com y, datas deles, nº de maduros).

    `n_maduros[i]` conta os posts com y publicados até `dias` antes do post i (inclusive, como o
    `mediana_madura`); os maduros do post i são os `n_maduros[i]` primeiros de y. Posts sem y não
    entram nem na contagem.
    """
    ordenado = df.sort_values("published_at", kind="stable")
    for _, g in ordenado.groupby("channel_id", sort=False):
        datas = g["published_at"].dt.tz_convert(None).to_numpy()
        com_y = g["y"].notna().to_numpy()
        y, datas_y = g["y"].to_numpy(float)[com_y], datas[com_y]
        maduros = np.searchsorted(datas_y, datas - np.timedelta64(dias, "D"), side="right")
        yield g.index, y, datas_y, datas, maduros


def media_ultimos_k(df, k, dias=30, min_posts=3) -> pd.Series:
    """Média de y dos últimos `k` posts maduros do canal, para cada post (aula 2.1, F25).

    Maduro é o da `mediana_madura`: publicado até `dias` antes, inclusive (princípio 2). Com menos
    de `k` maduros, usa os que houver; com menos de `min_posts`, o valor é NaN (cold start). Posts
    sem y não entram. O `k` é um hiperparâmetro: escolha-o no walk-forward de desenvolvimento.
    """
    if k < 1:
        raise ValueError(f"media_ultimos_k: k precisa ser >= 1 (veio {k})")
    saida = pd.Series(np.nan, index=df.index)
    for indice, y, _, _, maduros in _maduros_do_canal(df, dias):
        saida.loc[indice] = [
            y[max(n - k, 0) : n].mean() if n >= max(min_posts, 1) else np.nan for n in maduros
        ]
    return saida


def mediana_janela(df, janela_dias=90, dias=30, min_posts=3) -> pd.Series:
    """Mediana de y dos posts maduros publicados nos `janela_dias` antes do limite (aula 2.1, F25).

    O limite é o de maturidade: para um post em t, a janela vai de t - dias - janela_dias até
    t - dias, inclusive nas duas pontas (de t - 120 a t - 30 dias, com os padrões). Com menos de
    `MIN_NA_JANELA` posts na janela, usa os 10 últimos maduros, como o `historico_recente` da 6.1
    (que lá olha os 90 dias até o post; aqui, só com maduros, princípio 2). Com menos de
    `min_posts` maduros no total, o valor é NaN.
    """
    saida = pd.Series(np.nan, index=df.index)
    for indice, y, datas_y, datas, maduros in _maduros_do_canal(df, dias):
        inicio = np.searchsorted(
            datas_y, datas - np.timedelta64(dias + janela_dias, "D"), side="left"
        )
        valores = []
        for ini, n in zip(inicio, maduros, strict=True):
            if n < max(min_posts, 1):  # sem nenhum maduro não há o que resumir
                valores.append(np.nan)
            elif n - ini >= MIN_NA_JANELA:
                valores.append(np.median(y[ini:n]))
            else:
                valores.append(np.median(y[max(n - ULTIMOS_DO_FALLBACK, 0) : n]))
        saida.loc[indice] = valores
    return saida


def mediana_com_imaturos(df) -> pd.Series:
    """A mediana da 2.1: todos os posts anteriores do canal, maduros ou não (só para comparar)."""
    ordenado = df.sort_values("published_at", kind="stable")
    m = ordenado.groupby("channel_id")["y"].transform(lambda s: s.shift(1).expanding().median())
    return m.reindex(df.index)


def linhas_de_avaliacao(df, cfg_avaliacao) -> pd.Index:
    """Os testes do walk-forward de desenvolvimento: só a primeira cópia, com y."""
    folds = folds_de_desenvolvimento(df, cfg_avaliacao)
    teste = folds[0][1]
    for _, t in folds[1:]:
        teste = teste.union(t)
    sel = df.loc[teste]
    ok = (sel["duplicate_group"] == sel["post_id"]) & sel["y"].notna()
    return sel.index[ok]


def colunas_baselines(cfg) -> list[str]:
    """Os baselines do painel: os dois da 2.1 e os mais fortes (um por k e a mediana de 90 dias)."""
    return [*BASELINES, *[f"p_ultimos_{k}" for k in cfg["ks"]], "p_mediana_90d"]


def leitura_janela(df, cfg, cfg_avaliacao) -> pd.DataFrame:
    """O painel da leitura da M2: uma coluna por baseline, nas mesmas linhas.

    Os baselines novos (média dos últimos k e mediana de 90 dias) só ficam NaN onde a
    `mediana_madura` já fica (menos de `min_posts` maduros), então o filtro de linhas não encolhe.
    """
    d = df.copy()
    d["y"] = alvo(d)
    d["p_madura"] = mediana_madura(d, cfg["maturidade_dias"], cfg["min_posts"])
    d["p_com_imaturos"] = mediana_com_imaturos(d)
    for k in cfg["ks"]:
        d[f"p_ultimos_{k}"] = media_ultimos_k(d, k, cfg["maturidade_dias"], cfg["min_posts"])
    d["p_mediana_90d"] = mediana_janela(
        d, cfg["janela_baseline_dias"], cfg["maturidade_dias"], cfg["min_posts"]
    )
    colunas = colunas_baselines(cfg)
    aval = d.loc[linhas_de_avaliacao(d, cfg_avaliacao)]
    aval = aval[aval[colunas].notna().all(axis=1)]
    boot = {"n_boot": cfg["n_boot"], "seed": cfg["seed"]}
    n = aval.groupby("channel_id").size()  # posts de cada canal: o peso da manchete (#6)

    painel = {}
    for col in colunas:
        tab = por_canal(aval["y"], aval[col], aval["channel_id"])
        tab_j = pairwise_janela(aval, col, cfg["janela_dias"])
        ic_i, ic_j = ic_par_a_par(tab, **boot), ic_par_a_par(tab_j, **boot)
        ic_m = ic_par_a_par_peso_posts(tab_j, n, **boot)
        painel[col] = {
            "linhas": len(aval),
            "canais": aval["channel_id"].nunique(),
            "manchete_janela_peso_posts": par_a_par_peso_posts(tab_j, n),
            "ic_manchete_lo": ic_m[0],
            "ic_manchete_hi": ic_m[1],
            "par_a_par_intra": tab["acertos"].sum() / tab["pares"].sum(),
            "ic_intra_lo": ic_i[0],
            "ic_intra_hi": ic_i[1],
            "par_a_par_janela": tab_j["acertos"].sum() / tab_j["pares"].sum(),
            "ic_janela_lo": ic_j[0],
            "ic_janela_hi": ic_j[1],
            "pares_janela": int(tab_j["pares"].sum()),
            "fatia_3_maiores": tab_j["pares"].nlargest(3).sum() / tab_j["pares"].sum(),
            "par_a_par_janela_peso_igual": par_a_par_peso_igual(tab_j),
            "spearman_intra": spearman_intra_canal(aval["y"], aval[col], aval["channel_id"]),
        }
    return pd.DataFrame(painel)


def k_escolhido(painel) -> tuple[int, float]:
    """O k da média dos últimos k posts com a maior manchete (a `manchete_janela_peso_posts`).

    O painel vem do walk-forward de 2025, com o teste trancado: é a validação, e o k escolhido
    aqui é o que vale para o baseline final. Empate fica com o menor k.
    """
    colunas = [c for c in painel.columns if c.startswith("p_ultimos_")]
    if not colunas:
        raise ValueError("k_escolhido: o painel não tem colunas p_ultimos_k")
    colunas = sorted(colunas, key=lambda c: int(c.removeprefix("p_ultimos_")))  # empate: menor k
    manchetes = painel.loc["manchete_janela_peso_posts", colunas].astype(float)
    if manchetes.isna().any():  # o idxmax pularia o NaN em silêncio (princípio 7)
        raise ValueError(f"k_escolhido: manchete NaN em {list(manchetes[manchetes.isna()].index)}")
    melhor = manchetes.idxmax()
    return int(melhor.removeprefix("p_ultimos_")), float(manchetes[melhor])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m previa.baselines")
    parser.add_argument("acao", choices=["janela"])
    parser.parse_args(argv)
    tabela = leitura_janela(carregar_base(), ler_config(), ler_avaliacao())
    print(tabela.round(3).to_string())
    k, manchete = k_escolhido(tabela)
    print(
        f"\nk escolhido no walk-forward de 2025 (teste trancado): k = {k}, manchete {manchete:.3f}"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    sujo = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    print(f"\ncommit {commit}{' + mudanças não commitadas' if sujo else ''}")
    print("comando: uv run python -m previa.baselines janela")
    return 0


if __name__ == "__main__":
    sys.exit(main())
