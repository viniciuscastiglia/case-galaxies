"""Comparáveis sem vazamento (A5; aula 6.4): Shorts parecidos que vieram antes.

Dois filtros, como na aula: o **mesmo setor** (um índice por setor) e o **passado maduro** (só
posts publicados até 30 dias antes da data planejada, inclusive, a regra do princípio 2 e do
histórico da M3). Os posts de cada setor ficam em ordem cronológica, então "maduro antes de T"
vira um prefixo `[0, n)`. A similaridade é o cosseno (vetores com norma 1).

Desvio da aula: a busca é exata com numpy (o mesmo resultado do `faiss.IndexFlatIP`), sem FAISS.
Com ~15 mil Shorts por setor, uma busca leva milissegundos, e o processo não junta FAISS e
LightGBM, que no macOS derrubam o Python (6.4, quadro do segfault).

Dois papéis: **evidência** na resposta (`buscar`, com o relativo ao canal) e **feature** no treino
(`vizinhos_no_tempo`: a mediana do alvo relativo dos k vizinhos de cada post, com a mesma regra).
"""

import numpy as np
import pandas as pd

MATURIDADE_DIAS = 30


def _normalizar(x) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _sem_fuso(s) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(s, utc=True)).tz_convert(None).to_numpy()


class IndiceComparaveis:
    """Um índice por setor, em ordem cronológica. `posts` precisa de post_id, channel_id,
    published_at, sector e rel (o alvo relativo, log); `emb` é alinhado às linhas de `posts`.

    `limiar_dist` e `periodo_limiar` são o limiar da regra "fora da distribuição" (5.4) e o período
    de calibração de onde ele saiu (`previa.indice.construir_indice`). No nível da classe, `None`:
    um índice gravado antes deles carrega sem o limiar, e a regra não roda.
    """

    limiar_dist: float | None = None
    periodo_limiar: dict | None = None

    def __init__(self, posts: pd.DataFrame, emb, maturidade_dias: int = MATURIDADE_DIAS):
        self.maturidade = np.timedelta64(maturidade_dias, "D")
        x = _normalizar(emb)
        self._colunas = [*posts.columns, "similarity"]
        ok = posts["rel"].notna().to_numpy()
        self._por_setor = {}
        for setor in posts.loc[ok, "sector"].unique():
            pos = np.flatnonzero(ok & (posts["sector"] == setor).to_numpy())
            t = _sem_fuso(posts["published_at"].to_numpy()[pos])
            ordem = np.argsort(t, kind="stable")
            pos, t = pos[ordem], t[ordem]
            self._por_setor[setor] = (x[pos], posts.iloc[pos].reset_index(drop=True), t)

    def buscar(self, emb, setor: str, antes_de, k: int = 10) -> pd.DataFrame:
        """Os k vizinhos mais parecidos do mesmo setor, maduros (30+ dias) em `antes_de`."""
        if setor not in self._por_setor:
            return pd.DataFrame(columns=self._colunas)
        x, posts, t = self._por_setor[setor]
        limite = _sem_fuso(pd.Series([antes_de]))[0] - self.maturidade
        n = int(np.searchsorted(t, limite, side="right"))  # 30 dias exatos: entra
        if n == 0:
            return posts.iloc[0:0].assign(similarity=np.array([], np.float32))
        sims = x[:n] @ _normalizar(emb)
        top = np.argsort(-sims, kind="stable")[: min(k, n)]
        return posts.iloc[top].assign(similarity=sims[top]).reset_index(drop=True)


def vizinhos_no_tempo(indice: IndiceComparaveis, posts: pd.DataFrame, emb, k: int = 10):
    """Para cada post, a regra da tool com a data dele como "data planejada" (6.4).

    `rel_vizinhos`: mediana do alvo relativo dos k vizinhos; `sim_vizinho`: a similaridade do
    mais próximo (insumo da regra "fora da distribuição" do confidence_flag, 5.4).
    """
    emb = np.asarray(emb)
    linhas = []
    for i, (setor, quando) in enumerate(zip(posts["sector"], posts["published_at"], strict=True)):
        v = indice.buscar(emb[i], setor, quando, k)
        linhas.append(
            (v["rel"].median() if len(v) else np.nan, v["similarity"].max() if len(v) else np.nan)
        )
    return pd.DataFrame(linhas, index=posts.index, columns=["rel_vizinhos", "sim_vizinho"])


def limiar_distancia(
    indice: IndiceComparaveis, posts: pd.DataFrame, emb, percentil: float
) -> float:
    """O limiar de "fora da distribuição" (5.4): o percentil da distância ao vizinho mais parecido.

    Distância = 1 - cosseno ao vizinho maduro mais parecido (`vizinhos_no_tempo` com k = 1: a regra
    da tool). Os `posts` são os da CALIBRAÇÃO, nunca os do treino (5.4, erro comum 3). Post sem
    vizinho maduro fica de fora; sem nenhum, levanta (princípio 7).
    """
    d = (1.0 - vizinhos_no_tempo(indice, posts, emb, k=1)["sim_vizinho"]).dropna()
    if d.empty:
        raise ValueError("limiar_distancia: nenhum post da calibração tem vizinho maduro no índice")
    return float(np.percentile(d.to_numpy(float), percentil))
