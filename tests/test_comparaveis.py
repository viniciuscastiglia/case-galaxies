"""Testes dos comparáveis sem vazamento (A5; aula 6.4): mesmo setor, passado maduro (30+ dias),
nunca o próprio post; a feature de treino usa a mesma regra da tool."""

import numpy as np
import pandas as pd
import pytest

from previa.comparaveis import IndiceComparaveis, limiar_distancia, vizinhos_no_tempo

T0 = pd.Timestamp("2025-01-01", tz="UTC")


def _posts():
    """6 posts: 4 de beleza, 2 automotivos; embeddings 2D com ângulos conhecidos."""
    angulos = [0, 10, 80, 20, 5, 0]  # graus
    emb = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in angulos])
    posts = pd.DataFrame(
        {
            "post_id": [f"p{k}" for k in range(6)],
            "channel_id": ["UCa", "UCb", "UCa", "UCb", "UCc", "UCd"],
            "published_at": T0 + pd.to_timedelta([0, 10, 20, 100, 0, 5], "D"),
            "sector": ["beleza"] * 4 + ["automotivo"] * 2,
            "rel": [0.5, -0.2, 0.1, 0.3, 0.9, -0.9],
            "metric_target": [100, 200, 300, 400, 500, 600],
            "caption": ["a", "b", "c", "d", "e", "f"],
        }
    )
    return posts, emb


def test_mesmo_setor_e_so_o_passado_maduro():
    posts, emb = _posts()
    indice = IndiceComparaveis(posts, emb)
    consulta = np.array([1.0, 0.0])  # ângulo 0
    # em 2025-02-05: maduros (até 06/01) são p0 (01/01) e p1 (11/01? não: 11/01 > 06/01)
    v = indice.buscar(consulta, "beleza", T0 + pd.Timedelta(days=35), k=5)
    assert v["post_id"].tolist() == ["p0"]
    # em 2025-03-01 (limite 30/01): p0, p1 e p2, em ordem de similaridade com o ângulo 0
    v = indice.buscar(consulta, "beleza", T0 + pd.Timedelta(days=59), k=5)
    assert v["post_id"].tolist() == ["p0", "p1", "p2"]
    assert v["similarity"].iloc[0] == pytest.approx(1.0)
    assert set(indice.buscar(consulta, "automotivo", T0 + pd.Timedelta(days=60))["post_id"]) == {
        "p4",
        "p5",
    }


def test_exatamente_30_dias_entra():
    posts, emb = _posts()
    v = IndiceComparaveis(posts, emb).buscar(emb[0], "beleza", T0 + pd.Timedelta(days=30), k=5)
    assert v["post_id"].tolist() == ["p0"]


def test_sem_passado_maduro_devolve_vazio():
    posts, emb = _posts()
    v = IndiceComparaveis(posts, emb).buscar(emb[0], "beleza", T0 + pd.Timedelta(days=5))
    assert v.empty and "similarity" in v


def test_feature_no_tempo_usa_a_regra_da_tool_e_nunca_o_proprio_post():
    posts, emb = _posts()
    indice = IndiceComparaveis(posts, emb)
    f = vizinhos_no_tempo(indice, posts, emb, k=2)
    # p3 (dia 100, ângulo 20): candidatos p0, p1, p2; os 2 mais parecidos: p1 (10°) e p0 (0°)
    assert f.loc[3, "rel_vizinhos"] == pytest.approx(np.median([-0.2, 0.5]))
    assert f.loc[3, "sim_vizinho"] == pytest.approx(np.cos(np.radians(10)))
    assert np.isnan(f.loc[0, "rel_vizinhos"])  # o primeiro post não tem passado
    for i in posts.index:  # a mesma regra da tool, post a post
        v = indice.buscar(emb[i], posts.loc[i, "sector"], posts.loc[i, "published_at"], k=2)
        esperado = v["rel"].median() if len(v) else np.nan
        assert f.loc[i, "rel_vizinhos"] == pytest.approx(esperado, nan_ok=True)
        assert posts.loc[i, "post_id"] not in v["post_id"].tolist()


def test_setor_sem_indice_devolve_as_mesmas_colunas():
    posts, emb = _posts()
    indice = IndiceComparaveis(posts, emb)
    vazio = indice.buscar(emb[0], "moda", T0 + pd.Timedelta(days=60))
    cheio = indice.buscar(emb[0], "beleza", T0 + pd.Timedelta(days=60))
    assert vazio.empty and list(vazio.columns) == list(cheio.columns)


def test_limiar_e_o_percentil_das_distancias_ao_vizinho():
    """5.4 (fora da distribuição): distância = 1 - cosseno ao vizinho maduro mais parecido."""
    sims = [1.0, 0.9, 0.8, 0.7, 0.6]  # distâncias 0,0 a 0,4 ao post de referência
    angulos = [0.0] + [float(np.degrees(np.arccos(c))) for c in sims]
    emb = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in angulos])
    posts = pd.DataFrame({
        "post_id": [f"p{k}" for k in range(6)], "channel_id": "UCa",
        "published_at": [T0] + [T0 + pd.Timedelta(days=60)] * 5,
        "sector": "beleza", "rel": 0.0,
    })  # fmt: skip
    indice = IndiceComparaveis(posts, emb)
    cal = posts.iloc[1:]
    assert limiar_distancia(indice, cal, emb[1:], 50) == pytest.approx(0.2, abs=1e-6)
    assert limiar_distancia(indice, cal, emb[1:], 100) == pytest.approx(0.4, abs=1e-6)


def test_limiar_sem_nenhum_vizinho_levanta():
    posts, emb = _posts()
    indice = IndiceComparaveis(posts, emb)
    with pytest.raises(ValueError, match="nenhum"):
        limiar_distancia(indice, posts.iloc[[0]], emb[[0]], 99)  # p0 não tem passado maduro


def test_indice_sem_limiar_tem_os_atributos_vazios():
    posts, emb = _posts()
    indice = IndiceComparaveis(posts, emb)
    assert indice.limiar_dist is None and indice.periodo_limiar is None
