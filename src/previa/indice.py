"""O índice dos comparáveis do pacote (aula 6.4), salvo em `ARTEFATOS/comparaveis.joblib`.

Lê a base congelada do pacote (`DADOS / bundle.base`) e os embeddings SigLIP das thumbnails
(`DADOS / "emb_thumb_v1"`), fica com os Shorts canônicos (a 1ª cópia de cada criativo) com
curtidas e embedding, e entrega ao `IndiceComparaveis` (um índice por setor, só o passado maduro).
`rel` é o alvo relativo ao canal (y menos a mediana madura do canal); posts sem histórico maduro
ficam de fora, como no índice do treino.

    uv run python -m previa.indice
"""

import sys

import joblib
import numpy as np
import pandas as pd

from previa.caminhos import ARTEFATOS, DADOS
from previa.comparaveis import IndiceComparaveis
from previa.features import historico_do_canal


def construir_indice(bundle, base: pd.DataFrame, emb_thumb: pd.DataFrame) -> IndiceComparaveis:
    """Índice por setor dos Shorts canônicos com curtidas e embedding da capa (índice = post_id)."""
    d = base[base["duplicate_group"] == base["post_id"]].copy()
    d["y"] = np.log1p(pd.to_numeric(d["metric_target"], errors="coerce").astype(float))
    d = d[d["y"].notna() & d["post_id"].isin(emb_thumb.index)]
    if d.empty:
        raise ValueError("nenhum Short canônico com curtidas e embedding da capa: índice vazio")
    d["rel"] = d["y"] - historico_do_canal(d, bundle.lacuna_dias)["hist_mediana"]
    d["likes"] = d["metric_target"].astype(int)
    posts = d[["post_id", "channel_id", "published_at", "sector", "likes", "rel"]]
    posts = posts.reset_index(drop=True)
    return IndiceComparaveis(posts, emb_thumb.loc[posts["post_id"]].to_numpy(np.float32))


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
