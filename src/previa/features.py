"""Features do canal e do tempo (M3; aula 3.5, com a regra dos posts maduros da 2.2).

Para cada post, o histórico do canal só usa posts publicados ao menos `lacuna_dias` antes (30, o
princípio 2: "anteriores e maduros") e com curtidas. É a versão robusta da 3.5, com
`np.searchsorted`, que trata empates e a lacuna de maturação; os 30 dias contam inclusive, como na
`mediana_madura` do baseline (um post de exatamente 30 dias atrás entra). A cadência e o
calendário não leem curtidas: usam todos os posts anteriores, sem lacuna.

`montar_features` calcula tudo sobre a base inteira, em ordem de data, e só depois alguém separa
treino e teste (erro comum 3). É a mesma função no treino e na previsão (princípio 4): para um
Short novo, acrescente a linha dele (y vazio) aos posts do canal e chame de novo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dateutil.easter import easter

from previa.caminhos import CONFIGS

HISTORICO = [
    "hist_n",
    "hist_mediana",
    "hist_media",
    "hist_desvio",
    "hist_med5",
    "hist_med20",
    "tendencia",
]
CADENCIA = ["dias_desde_ultimo", "posts_30d"]
CALENDARIO = ["hora_local", "dia_semana_local", "feriado", "idade_canal_dias"]
FEATURES = HISTORICO + CADENCIA + CALENDARIO  # as 13 colunas da 3.5 que vão para o modelo


def ler_config(caminho=CONFIGS / "features.yaml") -> dict:
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def _sem_fuso(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert(None).to_numpy()  # datetime64 em UTC: subtrai e compara


def historico_do_canal(df, lacuna_dias=30) -> pd.DataFrame:
    """Estatísticas de y dos posts do canal publicados até `lacuna_dias` antes de cada post.

    Com `lacuna_dias=0`, só posts estritamente anteriores (nem o próprio, nem os simultâneos).
    `hist_ultimo_ts` é a data do post mais recente usado, para o teste de vazamento.
    """
    lado = "right" if lacuna_dias > 0 else "left"  # 30 dias inclusive; com 0, estritamente antes
    partes = []
    for _, g in df.groupby("channel_id", sort=False):
        t = _sem_fuso(g["published_at"])
        h = g[g["y"].notna()].sort_values("published_at", kind="stable")
        th, y = _sem_fuso(h["published_at"]), h["y"].reset_index(drop=True)
        k = np.searchsorted(th, t - np.timedelta64(lacuna_dias, "D"), side=lado)  # nº de maduros
        stats = pd.DataFrame(
            {
                "hist_mediana": y.expanding().median(),
                "hist_media": y.expanding().mean(),
                "hist_desvio": y.expanding().std(),
                "hist_med5": y.rolling(5, min_periods=1).median(),
                "hist_med20": y.rolling(20, min_periods=1).median(),
            }
        )
        if len(y):
            f = stats.iloc[np.maximum(k - 1, 0)].reset_index(drop=True)  # até o post k-1
            ultimo = pd.Series(th[np.maximum(k - 1, 0)])
        else:  # canal sem nenhuma curtida
            f = pd.DataFrame(np.nan, index=range(len(t)), columns=stats.columns)
            ultimo = pd.Series(pd.NaT, index=range(len(t)), dtype="datetime64[ns]")
        f.loc[k == 0, :] = np.nan  # sem histórico maduro ainda
        f.insert(0, "hist_n", k)
        f["tendencia"] = f["hist_med5"] - f["hist_med20"]  # > 0: acima do próprio normal
        f["hist_ultimo_ts"] = ultimo.where(k > 0).dt.tz_localize("UTC")
        f.index = g.index
        partes.append(f)
    return pd.concat(partes).loc[df.index]


def cadencia(df) -> pd.DataFrame:
    """Dias desde o post anterior e posts nos 30 dias antes (todos os posts, sem lacuna)."""
    partes = []
    for _, g in df.groupby("channel_id", sort=False):
        t = _sem_fuso(g["published_at"])
        todos = np.sort(t)
        antes = np.searchsorted(todos, t, side="left")  # posts estritamente anteriores
        dias = (t - todos[np.maximum(antes - 1, 0)]) / np.timedelta64(1, "D")
        k30 = np.searchsorted(todos, t - np.timedelta64(30, "D"), side="left")
        partes.append(
            pd.DataFrame(
                {"dias_desde_ultimo": np.where(antes > 0, dias, np.nan), "posts_30d": antes - k30},
                index=g.index,
            )
        )
    return pd.concat(partes).loc[df.index]


def feriados_br(anos):
    """Feriados nacionais fixos + Sexta-feira Santa + segunda e terça de Carnaval (aula 3.5)."""
    dias = []
    for a in anos:
        for md in ["01-01", "04-21", "05-01", "09-07", "10-12", "11-02", "11-15", "11-20", "12-25"]:
            dias.append(pd.Timestamp(f"{a}-{md}").date())
        pascoa = pd.Timestamp(easter(a))
        dias += [(pascoa - pd.Timedelta(days=n)).date() for n in (2, 47, 48)]
    return dias


def calendario(df, fuso="America/Sao_Paulo") -> pd.DataFrame:
    """Hora e dia da semana locais (a API devolve UTC), feriado e idade do canal na base."""
    local = df["published_at"].dt.tz_convert(fuso)
    anos = range(local.dt.year.min(), local.dt.year.max() + 1)
    primeiro = df.groupby("channel_id")["published_at"].transform("min")  # proxy da 3.5
    return pd.DataFrame(
        {
            "hora_local": local.dt.hour,
            "dia_semana_local": local.dt.dayofweek,  # 0 = segunda
            "feriado": local.dt.date.isin(feriados_br(anos)).astype(int),
            "idade_canal_dias": (df["published_at"] - primeiro).dt.days,
        },
        index=df.index,
    )


def montar_features(df, lacuna_dias=30, fuso="America/Sao_Paulo") -> pd.DataFrame:
    """As 13 features da 3.5 (mais `hist_ultimo_ts`), sobre a base inteira, no índice de `df`."""
    return pd.concat(
        [historico_do_canal(df, lacuna_dias), cadencia(df), calendario(df, fuso)], axis=1
    )
