# ruff: noqa: F811
"""A demo: a regra de seleção (#18), escrita antes de rodar, e o golden com alarme (6.6)."""

from pathlib import Path

import pandas as pd
import pytest
from test_capa import EncoderFalso
from test_modelo import base, cfg_av  # noqa: F401
from test_previsao import PLANEJADO, capa, pacote  # noqa: F401

from previa.caminhos import ARTEFATOS, DADOS, RAIZ
from previa.demo import (
    avaliar_golden,
    config_do_servidor,
    ler_config_demo,
    sanidade_por_tamanho,
    selecionar,
)

CFG = {"golden": {"cobertura_min": 0.6, "cobertura_max": 0.97, "par_a_par_min": 0.5}}


def test_o_mais_recente_de_cada_canal_e_o_par_dos_dois_mais_recentes():
    g = pd.DataFrame({
        "handle": ["@a", "@a", "@a", "@b", "@b"],
        "video_id": ["a1", "a2", "a3", "b1", "b2"],
        "published_at": pd.to_datetime(
            ["2026-08-01", "2026-08-10", "2026-08-05", "2026-07-01", "2026-07-20"], utc=True
        ),
    })  # fmt: skip
    recentes, pares = selecionar(g)
    assert list(recentes["video_id"]) == ["a2", "b2"]
    assert [(a["video_id"], b["video_id"]) for a, b in pares] == [("a3", "a2"), ("b1", "b2")]


def test_canal_com_um_short_so_nao_forma_par():
    g = pd.DataFrame({
        "handle": ["@a"],
        "video_id": ["a1"],
        "published_at": pd.to_datetime(["2026-08-01"], utc=True),
    })  # fmt: skip
    recentes, pares = selecionar(g)
    assert list(recentes["video_id"]) == ["a1"] and pares == []


def _prev(reais, previstos, canais, lo=None, hi=None):
    return pd.DataFrame({
        "channel_id": canais, "real": reais, "predicted_metric": previstos,
        "interval_low": lo or [0] * len(reais), "interval_high": hi or [1e9] * len(reais),
    })  # fmt: skip


def test_golden_conta_dentro_e_pares_do_mesmo_canal():
    canais = ["a", "a", "a", "b"]
    p = _prev([10, 20, 30, 5], [1, 2, 3, 9], canais, lo=[5, 5, 5, 6], hi=[15, 25, 25, 9])
    r = avaliar_golden(p, CFG)
    assert (r["n"], r["dentro"]) == (4, 2)  # 30 > 25 e 5 < 6 ficam fora
    assert (r["acertos"], r["pares"]) == (3.0, 3)  # o canal b não forma par
    assert r["par_a_par"] == 1.0


def test_golden_empate_no_real_fica_fora_e_na_previsao_vale_meio():
    r = avaliar_golden(_prev([10, 10, 20], [1, 2, 2], ["a"] * 3), CFG)
    assert (r["acertos"], r["pares"]) == (1.5, 2)


def test_golden_alarme_fora_dos_limites():
    r = avaliar_golden(_prev([10, 20], [2, 1], ["a", "a"]), CFG)  # cobertura 1,0 e par-a-par 0
    assert any("cobertura" in a for a in r["alarmes"]) and any("par" in a for a in r["alarmes"])


def test_golden_nos_limites_nao_alarma():
    r = avaliar_golden(_prev([10, 20, 30, 40], [1, 2, 3, 4], ["a"] * 4, hi=[1e9, 1e9, 1e9, 1]), CFG)
    assert r["cobertura"] == 0.75 and r["alarmes"] == []


def test_config_da_demo_tem_os_limiares_do_golden():
    assert ler_config_demo()["golden"] == CFG["golden"]


def test_config_do_servidor_usa_caminhos_absolutos_e_outro_diretorio():
    """6.5, erro comum 2: o host lança o servidor de outro diretório; só caminho absoluto serve."""
    c = config_do_servidor()["previa"]
    assert c["transport"] == "stdio" and c["command"] == "uv"
    assert c["args"][:3] == ["run", "--directory", str(RAIZ)]
    assert c["args"][-2:] == ["-m", "previa.servidor_mcp"]
    assert Path(c["cwd"]).resolve() != RAIZ.resolve()
    assert c["env"]["GALAXIES_ARTEFATOS"] == str(ARTEFATOS)
    assert c["env"]["PREVIA_DADOS"] == str(DADOS)
    assert all(Path(c["env"][v]).is_absolute() for v in ("GALAXIES_ARTEFATOS", "PREVIA_DADOS"))


def test_sanidade_o_numero_cresce_com_o_canal(pacote, capa):
    """1.4: um canal fictício por faixa de tamanho; o número tem de seguir o tamanho do canal."""
    bundle, _ = pacote
    cfg = {"sanidade": {"curtidas": [20, 500, 20000], "posts": 30}}
    t = sanidade_por_tamanho(cfg, PLANEJADO, "beleza", capa, bundle=bundle, encoder=EncoderFalso())
    assert list(t["curtidas_canal"]) == [20, 500, 20000]
    assert t["predicted_metric"].is_monotonic_increasing and t["predicted_metric"].is_unique
    assert (t["interval_low"] <= t["predicted_metric"]).all()
    assert (t["predicted_metric"] <= t["interval_high"]).all()
    assert (t["razao_intervalo"] >= 1).all()
    assert all("canal_novo" not in m for m in t["motivos"])  # 30 posts maduros


def test_sanidade_mede_a_largura_na_escala_do_modelo(pacote, capa):
    """A largura em log1p (a do modelo): alto/baixo cru exagera nos canais pequenos, pelo +1."""
    bundle, _ = pacote
    cfg = {"sanidade": {"curtidas": [20, 500, 20000], "posts": 30}}
    t = sanidade_por_tamanho(cfg, PLANEJADO, "beleza", capa, bundle=bundle, encoder=EncoderFalso())
    esperado = (t["interval_high"] + 1) / (t["interval_low"] + 1)
    assert t["razao_intervalo"].to_numpy() == pytest.approx(esperado.to_numpy())


def test_config_da_sanidade():
    assert ler_config_demo()["sanidade"] == {"curtidas": [20, 500, 20000], "posts": 30}
