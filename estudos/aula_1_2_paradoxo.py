import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(42)
n = 3000

# Base sintética com efeitos conhecidos (NÃO são dados reais)
df = pd.DataFrame({
    "dia": np.sort(rng.integers(0, 540, n)),            # dia de publicação
    "log_inscritos": rng.normal(11.5, 1.2, n),          # ~100 mil inscritos
    "gancho": rng.integers(0, 11, n),                   # nota 0-10
    "duracao_s": rng.integers(8, 180, n),
})
sinal = (
    0.8 * df["log_inscritos"]
    + 0.25 * np.sin(df["gancho"] / 2)                   # efeito não linear
    - 0.004 * df["duracao_s"]
)
drift = 0.002 * df["dia"]
df["y"] = sinal + drift + rng.normal(0, 0.6, n)                 # y = log1p(curtidas)

X = df[["log_inscritos", "gancho", "duracao_s"]]
y = df["y"]

# "Nunca visto" = futuro: treino no passado, validação nos últimos 20%
corte = df["dia"].quantile(0.8)
tr, va = df["dia"] <= corte, df["dia"] > corte

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("MSE validação temporal  MSE validação aleatória")
m_temporal = DecisionTreeRegressor(max_depth=6, random_state=0)
m_aleatoria = DecisionTreeRegressor(max_depth=6, random_state=0)
temporal = m_temporal.fit(X[tr], y[tr])
aleatoria = m_aleatoria.fit(X_train, y_train)
mse_vaT = mean_squared_error(y[va], m_temporal.predict(X[va]))
mse_vaA = mean_squared_error(y_test, m_aleatoria.predict(X_test))
print(f"{mse_vaT:.4f}     {mse_vaA:.4f}")