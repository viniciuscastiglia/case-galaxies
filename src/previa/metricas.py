"""Métricas do projeto Galaxies (aula 2.4). y e p em escala log: y = log1p(curtidas).

O código da aula, sem mudanças: o MAPE ignora y = 0; o Spearman usa ranks médios e vale 0 para
séries constantes; o par-a-par pula empates em y e dá meio ponto para empates na previsão; o
intra-canal pondera por n_c, com mínimo de 3 posts. Os ICs reamostram CANAIS, nunca posts (erro
comum 4): as métricas intra-canal são somas das linhas de `por_canal`.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def mae_log(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse_log(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mape(y, p):
    """MAPE em %, ignorando y == 0 (onde a métrica não existe)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = y != 0
    return float(100 * np.mean(np.abs((y[ok] - p[ok]) / y[ok]))) if ok.any() else np.nan


def spearman_global(y, p):
    """Pearson entre os ranks (empate = rank médio). 0 se uma série for constante."""
    ry, rp = pd.Series(y).rank().to_numpy(), pd.Series(p).rank().to_numpy()
    if ry.std() == 0 or rp.std() == 0:
        return 0.0
    return float(np.corrcoef(ry, rp)[0, 1])


def _pares(y, p):
    """(acertos, total) nos pares com y diferente; empate na previsão vale 0,5."""
    i, j = np.triu_indices(len(y), k=1)
    sy, sp = np.sign(y[i] - y[j]), np.sign(p[i] - p[j])
    validos = sy != 0
    return float((sy == sp)[validos].sum() + 0.5 * (sp == 0)[validos].sum()), int(validos.sum())


def por_canal(y, p, canal):
    """Uma linha por canal. As métricas intra-canal são somas ponderadas destas linhas."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float), "canal": canal})
    linhas = []
    for c, g in d.groupby("canal", sort=True):
        acertos, pares = _pares(g["y"].to_numpy(), g["p"].to_numpy())
        linhas.append(
            {
                "canal": c,
                "n": len(g),
                "spearman": spearman_global(g["y"], g["p"]),
                "acertos": acertos,
                "pares": pares,
            }
        )
    return pd.DataFrame(linhas).set_index("canal")


def spearman_intra_canal(y, p, canal, min_posts=3):
    """Spearman dentro de cada canal (com >= min_posts), ponderado pelo nº de posts."""
    t = por_canal(y, p, canal)
    t = t[t["n"] >= min_posts]
    return float((t["spearman"] * t["n"]).sum() / t["n"].sum()) if len(t) else np.nan


def pairwise_accuracy(y, p, canal=None):
    """Fração de pares ordenados corretamente. Com `canal`, só pares do mesmo canal."""
    if canal is None:
        acertos, pares = _pares(np.asarray(y, float), np.asarray(p, float))
    else:
        t = por_canal(y, p, canal)
        acertos, pares = t["acertos"].sum(), t["pares"].sum()
    return float(acertos / pares) if pares else np.nan


# convenção da aula até a decisão #7 (faixas: quantis do canal × cortes fixos)
CORTES = [-np.inf, -0.5, 0.5, 1.5, np.inf]  # log da razão vs. mediana do canal
FAIXAS = ["abaixo", "na média", "acima", "viral"]


def faixa(r):
    """Alvo relativo r = y - mediana_canal -> faixa (convenção do projeto)."""
    return pd.cut(np.asarray(r, float), CORTES, labels=FAIXAS, right=False).astype(str)


def f1_faixas(r, r_pred):
    return float(f1_score(faixa(r), faixa(r_pred), labels=FAIXAS, average="macro", zero_division=0))


def cobertura(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def ic_bootstrap_canal(tab, n_boot=2000, seed=0):
    """IC 95% das métricas intra-canal reamostrando CANAIS (linhas de `por_canal`)."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(tab), size=(n_boot, len(tab)))  # 1 linha = 1 reamostra
    w = np.where(tab["n"] >= 3, tab["n"], 0).astype(float)  # mesmo min_posts=3
    rho = tab["spearman"].to_numpy()
    sp = (rho[k] * w[k]).sum(axis=1) / w[k].sum(axis=1)
    pw = tab["acertos"].to_numpy()[k].sum(axis=1) / tab["pares"].to_numpy()[k].sum(axis=1)
    return np.percentile(sp, [2.5, 97.5]), np.percentile(pw, [2.5, 97.5])


# --- auxiliares da leitura por canal (Claude) ---


def ic_par_a_par(tab, n_boot=2000, seed=0):
    """IC 95% do par-a-par reamostrando canais; só usa `acertos` e `pares`.

    Serve à tabela de `por_canal` e à da `pairwise_janela`. Mesma matriz `k` do
    `ic_bootstrap_canal`: com a mesma seed, os dois sorteiam os mesmos canais.
    """
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(tab), size=(n_boot, len(tab)))
    with np.errstate(invalid="ignore"):  # reamostra só com canais sem par: NaN, fica de fora
        pw = tab["acertos"].to_numpy(float)[k].sum(axis=1) / tab["pares"].to_numpy(float)[k].sum(
            axis=1
        )
    return np.nanpercentile(pw, [2.5, 97.5])


def par_a_par_peso_igual(tab):
    """Par-a-par com peso igual por canal (decisão #6): a média simples das taxas por canal."""
    t = tab[tab["pares"] > 0]
    return float((t["acertos"] / t["pares"]).mean()) if len(t) else np.nan


def _taxa_e_peso(tab, n):
    """Taxa de cada canal e o peso dele (nº de posts); canal sem par fica com peso 0."""
    pares = tab["pares"].to_numpy(float)
    com_par = pares > 0
    taxa = np.where(com_par, tab["acertos"].to_numpy(float) / np.where(com_par, pares, 1), 0.0)
    peso = np.where(com_par, n.reindex(tab.index).to_numpy(float), 0.0)
    return taxa, peso


def par_a_par_peso_posts(tab, n):
    """Par-a-par com cada canal pesando pelo nº de posts `n` (decisão #6, a manchete).

    É a regra que a 2.4 usa no Spearman intra-canal (peso n_c): um canal com 3 posts não vale
    tanto quanto um com 40, e um canal que posta muito não domina a nota, como na média por par
    (os pares crescem mais rápido que os posts).
    """
    taxa, peso = _taxa_e_peso(tab, n)
    return float((taxa * peso).sum() / peso.sum()) if peso.sum() else np.nan


def ic_par_a_par_peso_posts(tab, n, n_boot=2000, seed=0):
    """IC 95% do `par_a_par_peso_posts`, reamostrando canais (a mesma matriz `k` dos outros ICs)."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(tab), size=(n_boot, len(tab)))
    taxa, peso = _taxa_e_peso(tab, n)
    with np.errstate(invalid="ignore"):  # reamostra só com canais sem par: NaN, fica de fora
        media = (taxa[k] * peso[k]).sum(axis=1) / peso[k].sum(axis=1)
    return np.nanpercentile(media, [2.5, 97.5])


def ic_ganho_pareado(tab_modelo, tab_base, n, n_boot=2000, seed=0):
    """IC 95% do ganho do modelo sobre o baseline na manchete da #6, pareado por canal (#5).

    Os dois são medidos nos mesmos pares; cada reamostra sorteia canais (a mesma matriz `k` dos
    outros ICs) e calcula a diferença das duas médias ponderadas nela.
    """
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(tab_modelo), size=(n_boot, len(tab_modelo)))
    taxa_m, peso = _taxa_e_peso(tab_modelo, n)
    taxa_b, _ = _taxa_e_peso(tab_base.reindex(tab_modelo.index), n)
    with np.errstate(invalid="ignore"):
        dif = ((taxa_m - taxa_b)[k] * peso[k]).sum(axis=1) / peso[k].sum(axis=1)
    return np.nanpercentile(dif, [2.5, 97.5])


# --- M2: par-a-par na janela de 14 dias (exercício da 2.4; caso à mão de 19/10 nos testes) ---


def pairwise_janela(d, col, dias=14):
    """Par-a-par intra-canal só entre posts publicados a até `dias` um do outro (exercício da 2.4).

    A decisão da Lumina compara dois Shorts do mesmo momento; o par-a-par da aula mistura posts
    com meses de diferença. `d` tem as colunas channel_id, published_at (UTC), y e `col` (a
    previsão). Devolve uma linha por canal, como `por_canal` (índice channel_id; colunas
    `acertos` e `pares`), para o IC por canal e o peso igual (#6) reusarem a tabela. As regras
    são as do `_pares`: pares com o mesmo y ficam de fora, empate na previsão vale 0,5. A 14 dias
    exatos, o par entra. A taxa do conjunto é acertos.sum() / pares.sum().
    """
    finitos = np.isfinite(d["y"].to_numpy(float)) & np.isfinite(d[col].to_numpy(float))
    if not finitos.all():  # falha nunca silenciosa: um NaN contaria como par e erro
        raise ValueError("pairwise_janela: y ou previsão com NaN (filtre as linhas antes)")
    linhas = []
    for canal, g in d.groupby("channel_id", sort=True):
        t = g["published_at"].dt.tz_convert(None).to_numpy()  # datetime64 sem fuso: subtrai
        y, p = g["y"].to_numpy(float), g[col].to_numpy(float)
        i, j = np.triu_indices(len(g), k=1)  # todos os pares (i, j) do canal, cada um uma vez
        perto = np.abs(t[i] - t[j]) <= np.timedelta64(dias, "D")  # só pares do mesmo momento
        i, j = i[perto], j[perto]
        sy, sp = np.sign(y[i] - y[j]), np.sign(p[i] - p[j])  # quem rendeu mais: real e previsto
        validos = sy != 0  # par com o mesmo y não tem resposta certa
        acertos = (sy == sp)[validos].sum() + 0.5 * (sp == 0)[validos].sum()  # empate vale 0,5
        pares = validos.sum()
        linhas.append({"channel_id": canal, "acertos": float(acertos), "pares": int(pares)})
    return pd.DataFrame(linhas, columns=["channel_id", "acertos", "pares"]).set_index("channel_id")
