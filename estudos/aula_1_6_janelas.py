"""Exercício da aula 1.6: alvo relativo com janela expansiva x janela de 90 dias.

Base sintética da seção "Na prática" da aula (NÃO são dados reais): a Lumina CRESCE ao
longo do ano. Rode com: uv run python estudos/aula_1_6_janelas.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
linhas = []
for canal, nivel, cresc in [("garagem22", 5.5, 0.0), ("lumina", 7.0, 0.03), ("megamoda", 9.5, 0.0)]:
    datas = pd.date_range("2026-01-05", periods=40, freq="4D", tz="UTC")
    log_c = nivel + cresc * np.arange(40) + rng.normal(0, 0.8, 40)
    linhas += [
        {
            "video_id": f"{canal}-{i:02d}",
            "channel_id": canal,
            "published_at": d,
            "likes": float(np.round(np.exp(v))),
        }
        for i, (d, v) in enumerate(zip(datas, log_c, strict=True))
    ]
df = pd.DataFrame(linhas).sort_values(["channel_id", "published_at", "video_id"])

MIN_HIST = 10  # posts anteriores necessários para existir "histórico do canal"


def mediana_expansiva(df: pd.DataFrame) -> pd.Series:
    """Mediana das curtidas de TODOS os posts anteriores do canal (o próprio post fica de fora).
    NaN enquanto o canal tiver menos de MIN_HIST posts anteriores."""
    return df.groupby("channel_id")["likes"].transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_HIST).median()
    )


def mediana_90d(df: pd.DataFrame) -> pd.Series:
    """Mediana das curtidas dos posts dos 90 DIAS anteriores do canal (o próprio post fica de
    fora). NaN se a janela tiver menos de MIN_HIST posts."""
    # TODO(human), no exercício da aula 2.1 (baseline de 90 dias, na base real):
    # rolling("90D", closed="left") numa série indexada por published_at, canal a
    # canal. Devolva a Series com o MESMO índice de df, senão o pandas desalinha as linhas.
    raise NotImplementedError


def alvo_relativo(likes: pd.Series, mediana: pd.Series) -> pd.Series:
    """log((curtidas + 1) / (mediana dos anteriores + 1)): 0 = no normal do canal."""
    return np.log((likes + 1) / (mediana + 1))


df["y_exp"] = alvo_relativo(df["likes"], mediana_expansiva(df))
print(df.groupby("channel_id")["y_exp"].mean().round(2))  # confira com a seção "Na prática"

df["y_90d"] = alvo_relativo(df["likes"], mediana_90d(df))
print(df.groupby("channel_id")[["y_exp", "y_90d"]].mean().round(2))

# Respostas
# 1) Qual janela reduz o viés positivo no canal que cresce (lumina)? Por quê?
#    -> a janela de 90d reduz o vies no canal que cresce, porque na expansiva os valores de
#    videos antigos vao puxar a mediana pra baixo sem serem de fato piores, mas sim por estarem
#    atrasados no crescimento do canal, a de 90 dias tira os antigos da conta
# 2) Em que situação real você preferiria a outra janela?
#    -> em uma situacao em que o canal tem poucos videos recentes postados
