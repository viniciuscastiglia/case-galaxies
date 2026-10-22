"""O índice dos comparáveis do pacote (aula 6.4), salvo em `ARTEFATOS/comparaveis.joblib`.

Lê a base congelada do pacote (`DADOS / bundle.base`) e os embeddings SigLIP das thumbnails
(`DADOS / "emb_thumb_v1"`), fica com os Shorts canônicos (a 1ª cópia de cada criativo) com
curtidas e embedding, e entrega ao `IndiceComparaveis` (um índice por setor, só o passado maduro).
`rel` é o alvo relativo ao canal (y menos a mediana madura do canal); posts sem histórico maduro
ficam de fora, como no índice do treino. O índice guarda o limiar da regra "fora da distribuição"
(5.4; decisão #14): o p99 (`configs/flag.yaml`) da distância ao vizinho mais parecido dos posts do
período de CALIBRAÇÃO do pacote, nunca do treino (erro comum 3).

    uv run python -m previa.indice
"""

import sys

import joblib
import numpy as np
import pandas as pd

from previa.caminhos import ARTEFATOS, DADOS
from previa.comparaveis import IndiceComparaveis, limiar_distancia
from previa.features import historico_do_canal
from previa.flag import carregar_cfg as carregar_cfg_flag


def construir_indice(
    bundle, base: pd.DataFrame, emb_thumb: pd.DataFrame, cfg_flag: dict | None = None
) -> IndiceComparaveis:
    """Índice por setor dos Shorts canônicos com curtidas e embedding da capa (índice = post_id).

    Com `bundle.calibracao`, também o limiar de "fora da distribuição" nos posts do índice
    publicados em [início, fim) da calibração; sem calibração, o limiar fica `None` (sem a regra).
    """
    d = base[base["duplicate_group"] == base["post_id"]].copy()
    d["y"] = np.log1p(pd.to_numeric(d["metric_target"], errors="coerce").astype(float))
    d = d[d["y"].notna() & d["post_id"].isin(emb_thumb.index)]
    if d.empty:
        raise ValueError("nenhum Short canônico com curtidas e embedding da capa: índice vazio")
    d["rel"] = d["y"] - historico_do_canal(d, bundle.lacuna_dias)["hist_mediana"]
    d["likes"] = d["metric_target"].astype(int)
    posts = d[["post_id", "channel_id", "published_at", "sector", "likes", "rel"]]
    posts = posts.reset_index(drop=True)
    emb = emb_thumb.loc[posts["post_id"]].to_numpy(np.float32)
    indice = IndiceComparaveis(posts, emb)
    cal = getattr(bundle, "calibracao", None) or {}
    if cal.get("inicio") and cal.get("fim"):
        quando = pd.to_datetime(posts["published_at"], utc=True)
        no_periodo = (quando >= pd.Timestamp(cal["inicio"], tz="UTC")) & (
            quando < pd.Timestamp(cal["fim"], tz="UTC")
        )
        cfg_flag = cfg_flag or carregar_cfg_flag()
        indice.limiar_dist = limiar_distancia(
            indice, posts[no_periodo], emb[no_periodo.to_numpy()], cfg_flag["percentil_knn"]
        )
        indice.periodo_limiar = {
            "inicio": cal["inicio"], "fim": cal["fim"], "n": int(no_periodo.sum())
        }  # fmt: skip
    return indice


def main(argv=None) -> int:
    from previa.modelo import carregar
    from previa.visual import carregar_embeddings

    bundle = carregar()
    base = pd.read_parquet(DADOS / bundle.base)
    emb = carregar_embeddings(DADOS / "emb_thumb_v1")
    indice = construir_indice(bundle, base, emb)
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    destino = ARTEFATOS / "comparaveis.joblib"
    joblib.dump(indice, destino)
    print(f"índice dos comparáveis salvo em {destino}")
    if indice.limiar_dist is None:
        print("AVISO: o pacote não tem calibração; a regra 'fora da distribuição' fica desligada")
    else:
        per = indice.periodo_limiar
        print(
            f"limiar de 'fora da distribuição': distância {indice.limiar_dist:.3f} "
            f"(p99 de {per['n']} posts da calibração, {per['inicio']} a {per['fim']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
