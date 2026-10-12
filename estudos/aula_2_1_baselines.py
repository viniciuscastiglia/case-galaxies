import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from previa.sintetico import gerar_base

df = gerar_base()
df = df.sort_values(["canal", "data_pub"]).reset_index(drop=True)
g = df.groupby("canal")["y"]

# shift(1): o post atual e os futuros nunca entram na conta
df["n_anteriores"] = df.groupby("canal").cumcount()
df["media_canal"] = g.transform(lambda s: s.shift(1).expanding().mean())
df["mediana_canal"] = g.transform(lambda s: s.shift(1).expanding().median())
df["media_ult5"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
df["log_inscritos"] = np.log(df["inscritos"])

cols = ["y", "n_anteriores", "mediana_canal", "media_ult5"]
print(df.loc[df["canal"] == "c00", ["data_pub"] + cols].head(6)
        .assign(data_pub=lambda d: d["data_pub"].dt.date).round(2).to_string(index=False))

def spearman(a, b):
    """Correlação entre os ranks; 0 se uma das séries for constante (como no site)."""
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return 0.0 if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])

def spearman_intra(d, col, min_posts=3):
    """Spearman dentro de cada canal, média ponderada pelo nº de posts."""
    partes = [(spearman(x["y"], x[col]), len(x)) for _, x in d.groupby("canal")
              if len(x) >= min_posts]
    return sum(r * n for r, n in partes) / sum(n for _, n in partes)

corte = pd.Timestamp("2026-03-01")
treino, teste = df[df["data_pub"] < corte], df[df["data_pub"] >= corte].copy()

# cold start: regressão linear y ~ log(inscritos), ajustada só no treino
reg = LinearRegression().fit(treino[["log_inscritos"]], treino["y"])
teste["b_global"] = treino["y"].median()
teste["b_inscritos"] = reg.predict(teste[["log_inscritos"]])
poucos = teste["n_anteriores"] < 3  # histórico curto demais: cai no cold start
for col in ["media_canal", "mediana_canal", "media_ult5"]:
    teste["b_" + col] = teste[col].where(~poucos, teste["b_inscritos"])

linhas = [{"baseline": col[2:],
           "MAE_log": (teste["y"] - teste[col]).abs().mean(),
           "spearman_global": spearman(teste["y"], teste[col]),
           "spearman_intra": spearman_intra(teste, col)}
          for col in teste.columns if col.startswith("b_")]
print(f"treino={len(treino)}  teste={len(teste)}  cold start no teste={poucos.sum()}")
print(pd.DataFrame(linhas).round(3).to_string(index=False))

def mediana_90d(grupo):
    s = grupo.set_index("data_pub")["y"]
    return pd.Series(s.rolling("90D", closed="left").median().to_numpy(), index = grupo.index)
df["mediana_90d"] = df.groupby("canal", group_keys=False)[["data_pub", "y"]].apply(mediana_90d)

def media_ult(k):
    return df.groupby("canal")["y"].transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean())

val = df[(df["data_pub"] >= "2025-09-01") & (df["data_pub"] < corte) & (df["n_anteriores"] >= 3)]
for k in [3, 5, 10, 20]:
    print(f"k={k:2d}  MAE na validação = {(val['y'] - media_ult(k)[val.index]).abs().mean():.3f}")

df["media_ult10"] = media_ult(10)
linhas = teste.index[(teste["n_anteriores"] >= 3) & df.loc[teste.index, "mediana_90d"].notna()]
for col in ["media_ult5", "media_ult10", "mediana_90d"]:
    mae = (df.loc[linhas, "y"] - df.loc[linhas, col]).abs().mean()
    print(f"{col:12s} MAE no teste = {mae:.3f}")

# item 3 do exercício: o LightGBM do passo 4 (mesmas features e parâmetros), avaliado nas
# MESMAS `linhas` dos baselines, com a redução do MAE e o Spearman intra-canal.

features = ["log_inscritos", "mediana_canal", "media_ult5", "n_anteriores", "hora",
            "duracao_s", "rosto", "texto_tela", "cortes_10s", "titulo_pergunta", "gancho_vlm"]

X_treino = df.loc[treino.index, features]
y_treino = df.loc[treino.index, "y"]

X_teste = df.loc[linhas, features]
y_teste = df.loc[linhas, "y"]

model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          random_state=0, verbose=-1)
model.fit(X_treino, y_treino)

previsoes = model.predict(X_teste)

df_teste_filtrado = df.loc[linhas].copy()
df_teste_filtrado["previsao_lgb"] = previsoes

mae_lgb = mean_absolute_error(y_teste, previsoes)
mae_baseline = mean_absolute_error(df.loc[linhas, 'y'], df.loc[linhas, "media_ult10"])
spearman_intra_baseline = spearman_intra(df_teste_filtrado, "media_ult10")
spearman_intra_lgb = spearman_intra(df_teste_filtrado, "previsao_lgb")

reducao_pct = ((mae_baseline - mae_lgb)/mae_baseline) * 100

print(f"reducao MAE {reducao_pct:.3f} %")
print(f"spearman intra baseline {spearman_intra_baseline:.3f}")
print(f"spearman intra modelo {spearman_intra_lgb:.3f}")