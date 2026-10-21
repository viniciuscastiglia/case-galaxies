"""M6, o núcleo: três quantis do LightGBM, CQR calibrado por data e cobertura (aulas 5.2 e 5.3).

Os passos são do Vinicius (caso à mão da Lumina, 9 Shorts de calibração, alpha = 0,2), e viram
os comentários abaixo. Tudo em log1p, a escala do modelo; `em_curtidas` converte no fim.
A calibração é por data, à mão: o `train_conformalize_test_split` do MAPIE é aleatório (5.3,
erro comum 1). O q̂ vem só de posts maduros que o modelo não viu no treino (erros 2 e 3).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb  # antes de qualquer torch/mlflow (OpenMP no macOS)
import numpy as np
import pandas as pd
import yaml

from previa.experimento import FIXOS, Config, coluna_alvo, linhas_de_treino

SE_BASE = 0.8 * 0.2  # variância de um acerto/erro com cobertura alvo de 80%


@dataclass
class ConfigIntervalo:
    alphas: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    alpha_cqr: float = 0.2
    # extremos pedem folhas maiores (5.2, erro comum 2; conflitos #18)
    min_data_in_leaf: dict[float, int] = field(default_factory=lambda: {0.1: 40, 0.5: 20, 0.9: 40})
    faixas_posts: list[int] = field(default_factory=lambda: [10, 50])


def ler_config(caminho) -> ConfigIntervalo:
    cru = yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))
    cru["min_data_in_leaf"] = {float(a): int(v) for a, v in cru["min_data_in_leaf"].items()}
    return ConfigIntervalo(**cru)


def treinar_quantis(d, idx, cfg: Config, feats, cfg_q: ConfigIntervalo) -> dict[float, lgb.Booster]:
    """Um LightGBM por quantil, no mesmo alvo e nas mesmas linhas do `treinar_gbm`.

    Early stopping nos últimos `val_dias` do treino com `metric="quantile"` do mesmo alpha
    (5.2, erro comum 3); refit no treino inteiro com o número de árvores escolhido.
    """
    y = coluna_alvo(cfg)
    tr = d.loc[linhas_de_treino(d, idx, cfg)].sort_values("published_at", kind="stable")
    corte = tr["published_at"].max() - pd.Timedelta(days=cfg.val_dias)
    ajuste, val = tr[tr["published_at"] <= corte], tr[tr["published_at"] > corte]
    base = {k: v for k, v in cfg.lgbm.items() if k not in ("objective", "alpha", "metric")}
    modelos = {}
    for alfa in cfg_q.alphas:
        params = {
            **base,
            **FIXOS,
            "seed": cfg.seed,
            "objective": "quantile",
            "metric": "quantile",
            "alpha": alfa,
            "min_data_in_leaf": cfg_q.min_data_in_leaf[alfa],
        }
        d_ajuste = lgb.Dataset(ajuste[feats], ajuste[y])
        busca = lgb.train(
            params,
            d_ajuste,
            num_boost_round=cfg.max_arvores,
            valid_sets=[lgb.Dataset(val[feats], val[y], reference=d_ajuste)],
            callbacks=[lgb.early_stopping(cfg.paciencia, verbose=False), lgb.log_evaluation(0)],
        )
        n = max(busca.best_iteration, 1)
        modelos[alfa] = lgb.train(params, lgb.Dataset(tr[feats], tr[y]), num_boost_round=n)
    return modelos


def prever_quantis(modelos, x: pd.DataFrame, ref: pd.Series, cfg: Config) -> tuple[np.ndarray, int]:
    """(n, 3) em log1p absoluto, ordenado por linha, e quantas linhas tinham cruzamento (5.2).

    Publicar sem ordenar é o erro comum 4: um criativo estranho pode cruzar em produção.
    """
    cols = [modelos[a].predict(x) for a in sorted(modelos)]
    q = np.column_stack(cols)
    if cfg.alvo == "relativo":
        q = q + ref.to_numpy()[:, None]
    cruzou = int((np.diff(q, axis=1) < 0).any(axis=1).sum())
    return np.sort(q, axis=1), cruzou


def escores_cqr(q10, q90, y) -> np.ndarray:
    """Passo 2: score de cada post da calibração = max(q10 − y, y − q90).

    Negativo se o y caiu dentro de [q10, q90]; positivo se caiu fora, e o valor é o quanto.
    """
    q10, q90, y = (np.asarray(v, dtype=float) for v in (q10, q90, y))
    return np.maximum(q10 - y, y - q90)


def q_chapeu(escores, alpha: float) -> float:
    """Passo 3: ordena os scores; k = ceil((n + 1)(1 − alpha)); q̂ = o k-ésimo menor.

    Com k > n a calibração é pequena demais para o nível pedido: q̂ = infinito (intervalo sem
    limite, falha explícita em vez de promessa falsa).
    """
    ordenados = np.sort(np.asarray(escores, dtype=float))
    n = len(ordenados)
    k = math.ceil(round((n + 1) * (1 - alpha), 9))  # round: 10 * 0,8 não pode virar 9
    if k > n:
        return float("inf")
    return float(ordenados[max(k, 1) - 1])


def intervalo(q10, q90, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    """Passo 4: o intervalo final em log é [q10 − q̂, q90 + q̂]."""
    return np.asarray(q10, dtype=float) - qhat, np.asarray(q90, dtype=float) + qhat


def em_curtidas(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """Passo 4: em curtidas com expm1, cortado em 0 (curtida não é negativa)."""
    return np.maximum(np.expm1(lo), 0.0), np.maximum(np.expm1(hi), 0.0)


def conter_o_numero(lo, hi, numero) -> tuple[np.ndarray, np.ndarray]:
    """Alarga o intervalo só se preciso, para que lo <= número <= hi (o número é o q50).

    Só aumenta a cobertura (o intervalo nunca encolhe): a garantia da 5.3 se mantém.
    """
    return np.minimum(lo, numero), np.maximum(hi, numero)


def faixa_de_posts(n_anteriores: pd.Series, cfg_q: ConfigIntervalo) -> pd.Series:
    """Grupo por posts anteriores do canal: < 10, 10 a 50, > 50 (limites em `faixas_posts`)."""
    a, b = cfg_q.faixas_posts
    return n_anteriores.map(lambda n: f"< {a}" if n < a else (f"{a} a {b}" if n <= b else f"> {b}"))


def cobertura(y, lo, hi, grupos: pd.Series | None) -> pd.DataFrame:
    """Passo 1: cobertura = fração dos y dentro de [lo, hi]; total e por grupo (5.2, erro 5).

    Traz n, erro-padrão sqrt(0,8·0,2/n) e a largura média em log, que vai ao lado.
    """
    t = pd.DataFrame(
        {
            "dentro": (np.asarray(y) >= np.asarray(lo)) & (np.asarray(y) <= np.asarray(hi)),
            "largura": np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float),
            "grupo": np.asarray(grupos) if grupos is not None else "total",
        }
    )

    def resumo(g):
        n = len(g)
        return pd.Series(
            {
                "n": n,
                "cobertura": g["dentro"].mean(),
                "erro_padrao": math.sqrt(SE_BASE / n),
                "largura": g["largura"].mean(),
            }
        )

    por_grupo = t.groupby("grupo", sort=True).apply(resumo, include_groups=False)
    if grupos is None:
        por_grupo.index = ["total"]
        return por_grupo
    total = resumo(t).to_frame("total").T
    return pd.concat([total, por_grupo])
