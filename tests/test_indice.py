# ruff: noqa: F811
"""O índice dos comparáveis do pacote (aula 6.4): construído de uma base e embeddings sintéticos."""

import joblib
import numpy as np
import pandas as pd
import pytest
from test_capa import EncoderFalso
from test_modelo import base, cfg_av  # noqa: F401
from test_previsao import PLANEJADO, _capa_480x360, _req, pacote  # noqa: F401

from previa.caminhos import ARTEFATOS
from previa.comparaveis import IndiceComparaveis
from previa.indice import construir_indice
from previa.previsao import prever_criativo


def _embs(base):
    rng = np.random.default_rng(11)
    return pd.DataFrame(rng.normal(size=(len(base), 8)), index=pd.Index(base["post_id"]))


def test_construir_indice_so_canonicos_com_curtidas_e_embedding(pacote):  # noqa: F811
    bundle, base = pacote
    base = base.copy()
    base.loc[base.index[100], "duplicate_group"] = base["post_id"].iloc[0]  # uma cópia
    base.loc[base.index[200], "metric_target"] = pd.NA  # curtidas ocultas
    emb = _embs(base).drop(index=base["post_id"].iloc[300])  # sem embedding
    indice = construir_indice(bundle, base, emb)
    assert isinstance(indice, IndiceComparaveis)
    ids = set()
    for _, posts, _ in indice._por_setor.values():
        ids |= set(posts["post_id"])
    for fora in (base["post_id"].iloc[100], base["post_id"].iloc[200], base["post_id"].iloc[300]):
        assert fora not in ids
    assert len(ids) > 1000


def test_prever_criativo_usa_o_indice_construido(pacote, tmp_path):  # noqa: F811
    bundle, base = pacote
    base = base.copy()
    base["metric_target"] = base["metric_target"].astype("Int64")
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(construir_indice(bundle, base, _embs(base)), ARTEFATOS / "comparaveis.joblib")
    capa = tmp_path / "capa.jpg"
    _capa_480x360(capa)
    r = prever_criativo(_req(str(capa)), bundle=bundle, encoder=EncoderFalso())
    assert len(r.comparables) == 3
    limite = pd.Timestamp(PLANEJADO).tz_convert("UTC") - pd.Timedelta(days=30)
    por_id = base.set_index("post_id")
    for c in r.comparables:
        assert por_id.loc[c.post_id, "sector"] == "beleza"
        assert c.published_at <= limite


def test_indice_vazio_levanta(pacote):  # noqa: F811
    bundle, base = pacote
    with pytest.raises(ValueError, match="índice vazio"):
        construir_indice(bundle, base, pd.DataFrame(index=pd.Index([], name="post_id")))
