"""A âncora da cadeia de cold start (#34, F10; aula 2.1).

A previsão do modelo é relativa ao canal (`previa + ancora`); o degrau da âncora depende de
quantos posts maduros o canal tem na data da previsão:

1. 3 ou mais posts maduros: a mediana do canal (`previa.modelo.referencia`), o regime avaliado;
2. 0 a 2 posts maduros, com inscritos: a regressão em log dos inscritos, ajustada nos canais do
   treino (`ajustar_ancora`);
3. sem inscritos: a mediana do setor no treino.

Só o degrau 1 é o regime avaliado; os outros saem sempre com o flag `canal_novo` e intervalo largo.
Post com curtidas ocultas (y ausente, F10) não conta como maduro.
"""

import warnings

import numpy as np
import pandas as pd

from previa.features import ler_config
from previa.modelo import MIN_POSTS_ANCORA, referencia


def ajustar_ancora(
    base_treino: pd.DataFrame, inscritos: pd.Series, fim_treino, lacuna_dias: int | None = None
) -> dict:
    """Ajusta os degraus 2 e 3 no bloco de treino (princípio 2: nada fora do treino).

    `fim_treino`: onde o bloco de treino termina (`cal_inicio - gap_dias`, `modelo.fim_do_treino`).
    A função só usa posts com `published_at < fim_treino - lacuna_dias` (a lacuna de maturação do
    treino, `configs/features.yaml`): o que é imaturo ou posterior ao treino não entra.
    `base_treino`: posts do treino, com `channel_id`, `sector`, `y` (log1p das curtidas).
    `inscritos`: `channel_id` -> inscritos. ATENÇÃO (F4): são os inscritos do dia da coleta
    (`subscribers_at_collection` da base mínima, ou `inscritos_verificacao` de `data/canais.csv`),
    não os do dia da publicação; servem só para esta âncora e nunca entram nas features do modelo.
    Devolve `coef` ([inclinação, intercepto] de log1p(mediana do canal) ~ log1p(inscritos), mínimos
    quadrados) e `mediana_setor` (mediana de y por setor).
    """
    if lacuna_dias is None:
        lacuna_dias = ler_config()["lacuna_dias"]
    limite = pd.Timestamp(fim_treino) - pd.Timedelta(days=lacuna_dias)
    base = base_treino[base_treino["published_at"] < limite].dropna(subset=["y"])
    mediana = base.groupby("channel_id")["y"].median()
    brutos = pd.to_numeric(inscritos.reindex(mediana.index), errors="coerce").astype(float)
    x = np.log1p(brutos.where(brutos >= 0)).dropna()
    if len(x) < len(mediana):
        warnings.warn(
            f"{len(mediana) - len(x)} de {len(mediana)} canais ficaram fora da regressão por "
            "inscritos ausentes ou inválidos",
            UserWarning,
            stacklevel=2,
        )
    if len(x) < 2 or x.nunique() < 2:
        raise ValueError(
            f"a regressão dos inscritos precisa de 2 canais com inscritos distintos; há {len(x)}"
        )
    inclinacao, intercepto = np.polyfit(x.to_numpy(), mediana.loc[x.index].to_numpy(), 1)
    return {
        "coef": [float(inclinacao), float(intercepto)],
        "mediana_setor": {s: float(v) for s, v in base.groupby("sector")["y"].median().items()},
    }


def ancora(
    posts_canal: pd.DataFrame, inscritos: int | None, setor: str, quando, params: dict
) -> tuple[float, str, int]:
    """(âncora em log1p, degrau, posts maduros) do canal em `quando`.

    `posts_canal`: `published_at` (UTC) e `y` (NaN se as curtidas estão ocultas, F10).
    """
    ref, n = referencia(posts_canal, quando)
    if n >= MIN_POSTS_ANCORA:
        return ref, "mediana_do_canal", n
    if inscritos is not None and not pd.isna(inscritos) and inscritos >= 0:
        a, b = params["coef"]
        return a * float(np.log1p(inscritos)) + b, "inscritos", n
    if setor not in params["mediana_setor"]:
        raise ValueError(
            f"setor {setor!r} sem mediana no treino: {sorted(params['mediana_setor'])}"
        )
    return params["mediana_setor"][setor], "mediana_do_setor", n
