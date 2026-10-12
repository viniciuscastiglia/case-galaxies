import lightgbm as lgb
import numpy as np
import pandas as pd

from previa.sintetico import gerar_base

df = gerar_base().sort_values(["canal", "data_pub"]).reset_index(drop=True)
g = df.groupby("canal")["y"]
df["log_inscritos"] = np.log(df["inscritos"])
# honesta: só posts anteriores
df["mediana_canal"] = g.transform(lambda s: s.shift(1).expanding().median())
# vazamentos que aparecem em projetos reais:
df["mediana_canal_tudo"] = g.transform("median")  # base inteira: inclui o futuro
df["media_ult5_sem_shift"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
df["log_views"] = np.log1p(df["views"])  # só existe depois da publicação
df["log_comentarios"] = np.log1p(df["comentarios"])  # idem

conteudo = ["hora", "duracao_s", "rosto", "texto_tela", "cortes_10s",
            "titulo_pergunta", "gancho_vlm"]
base = ["log_inscritos", "mediana_canal"] + conteudo
cenarios = {
    "honesto": base,
    "+ comentários": base + ["log_comentarios"],
    "+ views": base + ["log_views"],
    "mediana da base inteira": ["log_inscritos", "mediana_canal_tudo"] + conteudo,
    "últimos 5 sem shift(1)": ["log_inscritos", "media_ult5_sem_shift"] + conteudo,
}

def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return 0.0 if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])

def spearman_intra(d, col, min_posts=3):
    partes = [(spearman(x["y"], x[col]), len(x)) for _, x in d.groupby("canal")
              if len(x) >= min_posts]
    return sum(r * n for r, n in partes) / sum(n for _, n in partes)

corte = pd.Timestamp("2026-03-01")
treino, teste = df[df["data_pub"] < corte], df[df["data_pub"] >= corte].copy()
linhas, modelos = [], {}
for nome, cols in cenarios.items():
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          random_state=0, verbose=-1)
    m.fit(treino[cols].astype(float), treino["y"])
    teste["pred"] = m.predict(teste[cols].astype(float))
    modelos[nome] = m
    linhas.append({"cenario": nome, "MAE_log": (teste["y"] - teste["pred"]).abs().mean(),
                   "spearman": spearman(teste["y"], teste["pred"]),
                   "spearman_intra": spearman_intra(teste, "pred")})
print(pd.DataFrame(linhas).round(3).to_string(index=False))

def mediana_madura(grupo, dias=30):
    """Mediana dos posts do canal publicados pelo menos `dias` antes de cada post."""
    datas = grupo["data_pub"].to_numpy()
    acumulada = grupo["y"].expanding().median().to_numpy()  # mediana dos posts 0..i
    ultimo = np.searchsorted(datas, datas - np.timedelta64(dias, "D"), side="right") - 1
    valores = np.where(ultimo >= 0, acumulada[np.maximum(ultimo, 0)], np.nan)
    return pd.Series(valores, index=grupo.index)

m = modelos["+ views"]
imp = pd.Series(m.booster_.feature_importance("gain"), index=cenarios["+ views"])
print((imp / imp.sum()).sort_values(ascending=False).head(4).round(3).to_string())

df["mediana_madura"] = (df.groupby("canal", group_keys=False)[["data_pub", "y"]]
                        .apply(mediana_madura))
print(df.loc[df["canal"] == "c00", ["data_pub", "y", "mediana_canal", "mediana_madura"]]
        .iloc[3:9].assign(data_pub=lambda d: d["data_pub"].dt.date).round(2).to_string(index=False))

treino, teste = df[df["data_pub"] < corte], df[df["data_pub"] >= corte].copy()
versoes = {"shift(1)": base,
           "madura (30 dias)": ["log_inscritos", "mediana_madura"] + conteudo}
for nome, cols in versoes.items():
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          random_state=0, verbose=-1)
    m.fit(treino[cols].astype(float), treino["y"])
    teste["pred"] = m.predict(teste[cols].astype(float))
    print(f"{nome:17s} MAE={(teste['y'] - teste['pred']).abs().mean():.3f}  "
          f"spearman={spearman(teste['y'], teste['pred']):.3f}  "
          f"intra={spearman_intra(teste, 'pred'):.3f}")