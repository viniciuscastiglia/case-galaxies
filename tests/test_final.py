"""Testes do teste final (F16; aulas 2.3, erro comum 4, e 2.5, erro comum 4: o teste abre uma vez).

Base sintética de test_experimento (6 canais, 2021-05 a 2025-12) com um avaliacao.yaml temporário
cujas datas cabem nela. Nada aqui toca a base real, o registro real nem o mlflow.db do projeto.
"""

import functools

import mlflow
import pandas as pd
import pytest
from test_experimento import _base, mlflow_temporario  # noqa: F401

import previa.final as final
from previa.dados import carregar_base
from previa.experimento import EXPERIMENTO, ler_config, preparar
from previa.final import (
    alarmes,
    avaliar_final,
    conferir_cobertura,
    criterio,
    folds_finais,
    ja_aberto,
)
from previa.splits import ler_avaliacao

AVALIACAO = """\
teste_inicio: "2025-06-01"
treino_inicio: "2021-07-01"
gap_dias: 30
cal_inicio: "2025-04-01"
teste_fim: "2025-12-01"
walk_forward:
  n_folds: 4
  inicio: "2024-01-01"
canal_novo:
  n_grupos: 2
  seed: 0
final:
  ganho_minimo: 0.03
  cobertura_min_extra: 0.99
  alarme_manchete: 0.80
  alarme_spearman: 0.80
"""

CONFIG = """\
nome: teste
hipotese: H1
descricao: GBM pequeno
features: [hist_n, hist_mediana, tendencia, posts_30d, hora_local, duracao_s, setor]
lacuna_dias: 30
val_dias: 180
seed: 0
max_arvores: 40
paciencia: 10
lgbm: {objective: regression, learning_rate: 0.1, num_leaves: 7, min_data_in_leaf: 20}
"""


@pytest.fixture
def cfg_av(tmp_path):
    (tmp_path / "avaliacao.yaml").write_text(AVALIACAO)
    return ler_avaliacao(tmp_path / "avaliacao.yaml")


@pytest.fixture
def d(cfg_av):
    base, dur = _base()
    return preparar(base, dur, cfg_av, lacuna_dias=30)


def test_folds_finais_sem_teste_no_treino(d, cfg_av):
    f = folds_finais(d, cfg_av)
    assert set(f) == {"temporal", "canal_novo"}
    ((treino, teste),) = f["temporal"]
    corte = cfg_av["cal_inicio"] - pd.Timedelta(days=cfg_av["gap_dias"])
    assert len(treino) > 0 and len(teste) > 0
    assert d.loc[treino, "published_at"].max() < corte  # nem calibração nem gap no treino
    assert d.loc[treino, "published_at"].max() < cfg_av["teste_inicio"]
    assert not treino.intersection(teste).size
    assert d.loc[treino.union(teste), "linha"].all()  # só linhas marcadas (cópia única, com alvo)
    for tr_g, _ in f["canal_novo"]:
        assert d.loc[tr_g, "published_at"].max() < corte


def test_folds_finais_canal_novo_fora_do_treino(d, cfg_av):
    pares = folds_finais(d, cfg_av)["canal_novo"]
    assert len(pares) == cfg_av["canal_novo"]["n_grupos"]
    vistos = set()
    for tr_g, te_g in pares:
        canais_te = set(d.loc[te_g, "channel_id"])
        assert canais_te and not canais_te & set(d.loc[tr_g, "channel_id"])
        vistos |= canais_te
    assert vistos == set(d["channel_id"])  # cada canal é novo uma vez


def test_teste_so_entre_inicio_e_fim(d, cfg_av):
    f = folds_finais(d, cfg_av)
    testes = [f["temporal"][0][1]] + [te for _, te in f["canal_novo"]]
    for te in testes:
        datas = d.loc[te, "published_at"]
        assert datas.min() >= cfg_av["teste_inicio"]
        assert datas.max() < cfg_av["teste_fim"]  # os posts depois de teste_fim ficam de fora
    assert d["published_at"].max() >= cfg_av["teste_fim"]  # a base sintética tem posts depois


def test_avaliar_final_devolve_os_dois_cenarios(cfg_av):
    base, dur = _base()
    cfg = ler_config_texto(CONFIG)
    r = avaliar_final(cfg, base, dur, cfg_av, n_boot=50, seed=0)
    assert set(r) == {"temporal", "canal_novo"}
    assert {"manchete_modelo", "ganho", "ic_ganho_lo", "spearman_intra_modelo"} <= set(
        r["temporal"]
    )


def ler_config_texto(texto, pasta=None):
    import tempfile
    from pathlib import Path

    p = Path(pasta or tempfile.mkdtemp()) / "exp.yaml"
    p.write_text(texto)
    return ler_config(p)


def _res(g_t=0.05, lo_t=0.01, g_c=0.05, lo_c=0.01, mm=0.6, sp=0.3):
    def um(g, lo):
        return {
            "ganho": g,
            "ic_ganho_lo": lo,
            "ic_ganho_hi": g + 0.02,
            "manchete_modelo": mm,
            "manchete_base": mm - g,
            "spearman_intra_modelo": sp,
        }

    return {"temporal": um(g_t, lo_t), "canal_novo": um(g_c, lo_c)}


CFG = {
    "final": {
        "ganho_minimo": 0.03,
        "cobertura_min_extra": 0.99,
        "alarme_manchete": 0.8,
        "alarme_spearman": 0.8,
    }
}


def test_criterio_da_5():
    assert criterio(_res(0.05, 0.01, 0.02, 0.005), CFG) == {"temporal": True, "canal_novo": True}
    assert criterio(_res(0.02, 0.01), CFG)["temporal"] is False  # ganho abaixo de +0,03
    assert criterio(_res(0.05, -0.01), CFG)["temporal"] is False  # IC com o zero
    assert criterio(_res(0.05, 0.01, 0.05, 0.0), CFG)["canal_novo"] is False  # IC encosta no zero


def test_alarme_bom_demais(cfg_av):
    assert alarmes(_res(), cfg_av) == []
    msgs = alarmes(_res(mm=0.85), cfg_av)
    assert msgs and all("bom demais, procure vazamento" in m for m in msgs)
    assert alarmes(_res(sp=0.9), cfg_av)
    assert alarmes(_res(mm=0.80, sp=0.80), cfg_av) == []  # só passa de 0,80 se for maior


def test_sem_a_chave_final_ou_um_limite_levanta(cfg_av):
    sem = {k: v for k, v in cfg_av.items() if k != "final"}
    with pytest.raises(ValueError, match="final.ganho_minimo"):
        alarmes(_res(), sem)
    sem_um = {**cfg_av, "final": {"alarme_manchete": 0.8, "alarme_spearman": 0.8}}
    with pytest.raises(ValueError, match="final.ganho_minimo"):
        criterio(_res(), sem_um)


def test_ja_aberto_le_o_registro(tmp_path):
    reg = tmp_path / "teste_aberto.log"
    assert ja_aberto(reg) == []
    reg.write_text("a\nb\n")
    assert ja_aberto(reg) == ["a", "b"]


@pytest.fixture
def ambiente(tmp_path, cfg_av, monkeypatch, mlflow_temporario):  # noqa: F811
    """Base sintética em parquet, avaliacao e registro temporários; o resto é o código real."""
    base, dur = _base()
    caminho = tmp_path / "base.parquet"
    base.to_parquet(caminho)
    reg = tmp_path / "logs" / "teste_aberto.log"
    (tmp_path / "exp.yaml").write_text(CONFIG)
    monkeypatch.setattr(final, "REGISTRO", reg)
    monkeypatch.setattr(final, "caminho_da_base", lambda cfg_av: caminho)
    monkeypatch.setattr(final, "ler_avaliacao", lambda: cfg_av)
    monkeypatch.setattr(final, "ler_duracoes", lambda: dur)
    monkeypatch.setattr(
        final, "ler_config_baselines", lambda: {"n_boot": 50, "seed": 0, "min_posts": 3}
    )
    monkeypatch.setattr(
        final,
        "carregar_base",
        functools.partial(
            carregar_base,
            caminho_base=caminho,
            caminho_config=tmp_path / "avaliacao.yaml",
            registro=reg,
        ),
    )
    return {"reg": reg, "cfg": str(tmp_path / "exp.yaml"), "dur": dur, "dir": tmp_path}


def _nao_le_dados(*a, **k):
    raise AssertionError("carregar_base foi chamado: a recusa tem de vir antes de ler dados")


def test_cli_recusa_sem_flag(ambiente, capsys, monkeypatch):
    monkeypatch.setattr(final, "carregar_base", _nao_le_dados)
    assert final.main([ambiente["cfg"]]) != 0
    assert "--abrir-o-teste" in capsys.readouterr().out
    assert not ambiente["reg"].exists()  # nada foi aberto


def test_cli_recusa_segunda_abertura(ambiente, capsys, monkeypatch):
    monkeypatch.setattr(final, "carregar_base", _nao_le_dados)
    ambiente["reg"].parent.mkdir(parents=True)
    ambiente["reg"].write_text("2026-10-21T01:02:03+00:00 teste aberto (x)\n")
    assert final.main([ambiente["cfg"], "--abrir-o-teste"]) != 0
    saida = capsys.readouterr().out
    assert "2026-10-21T01:02:03" in saida and "--reabrir" in saida  # diz quando foi a primeira
    assert len(ja_aberto(ambiente["reg"])) == 1  # não abriu de novo


def test_cli_abre_uma_vez_e_registra(ambiente, capsys):
    with pytest.warns(UserWarning, match="avaliação final"):
        assert final.main([ambiente["cfg"], "--abrir-o-teste"]) == 0
    assert len(ja_aberto(ambiente["reg"])) == 1  # uma linha a mais, não duas
    saida = capsys.readouterr().out
    assert "temporal" in saida and "canal_novo" in saida and "critério" in saida.lower()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENTO])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.avaliacao"] == "final (teste)"
    assert runs.iloc[0]["tags.abertura"] == "1"
    assert runs.iloc[0]["tags.data_corte"] == "2025-06-01"  # o início do teste, não o do dev
    assert runs.iloc[0]["tags.teste_fim"] == "2025-12-01"


def test_cli_reabrir_registra_a_segunda(ambiente):
    ambiente["reg"].parent.mkdir(parents=True)
    ambiente["reg"].write_text("2026-10-21T01:02:03+00:00 teste aberto (x)\n")
    with pytest.warns(UserWarning, match="avaliação final"):
        assert final.main([ambiente["cfg"], "--abrir-o-teste", "--reabrir"]) == 0
    assert len(ja_aberto(ambiente["reg"])) == 2
    runs = mlflow.search_runs(experiment_names=[EXPERIMENTO])
    assert runs.iloc[0]["tags.abertura"] == "2"


def test_cli_recusa_so_amostra_antes_de_abrir(ambiente, monkeypatch):
    monkeypatch.setattr(final, "carregar_base", _nao_le_dados)
    texto = CONFIG + "so_amostra: true\n"
    (ambiente["dir"] / "amostra.yaml").write_text(texto)
    with pytest.raises(ValueError, match="so_amostra"):
        final.main([str(ambiente["dir"] / "amostra.yaml"), "--abrir-o-teste"])
    assert not ambiente["reg"].exists()


def test_cli_cobertura_faltando_recusa_sem_abrir(ambiente, monkeypatch):
    """Um post do período de teste sem linha no extra: recusa com a contagem e não abre."""
    base, _ = _base()
    ids = base.loc[
        (base["published_at"] >= "2025-06-01") & (base["published_at"] < "2025-12-01"), "post_id"
    ]
    fora_do_teste = base.loc[base["published_at"] < "2025-06-01", "post_id"]
    extra = pd.DataFrame(
        {"vlm_x": 1.0}, index=pd.Index([*fora_do_teste, *ids.iloc[30:]], name="post_id")
    )
    monkeypatch.setattr(final, "ler_extra", lambda nome: extra)
    monkeypatch.setattr(final, "carregar_base", _nao_le_dados)
    (ambiente["dir"] / "ext.yaml").write_text(CONFIG + "extras: [vlm]\n")
    with pytest.raises(ValueError, match=r"NÃO foi aberto.*vlm"):
        final.main([str(ambiente["dir"] / "ext.yaml"), "--abrir-o-teste"])
    assert not ambiente["reg"].exists()


def test_cli_registra_antes_de_mostrar(ambiente, monkeypatch):
    ordem = []
    real = final.registrar
    monkeypatch.setattr(
        final, "registrar", lambda *a, **k: ordem.append("registrar") or real(*a, **k)
    )
    monkeypatch.setattr(final, "_imprimir", lambda *a, **k: ordem.append("imprimir") or [])
    with pytest.warns(UserWarning):
        final.main([ambiente["cfg"], "--abrir-o-teste"])
    assert ordem == ["registrar", "imprimir"]


@pytest.fixture
def mini(tmp_path, cfg_av):
    """200 posts canônicos na janela de teste, 5 duplicatas não canônicas e 10 fora da janela."""
    n = 200
    ids = [f"p{k:03d}" for k in range(n)]
    linhas = [(i, pd.Timestamp("2025-07-01", tz="UTC"), i) for i in ids]
    linhas += [(f"d{k}", pd.Timestamp("2025-07-02", tz="UTC"), ids[k]) for k in range(5)]
    linhas += [(f"f{k}", pd.Timestamp("2025-01-02", tz="UTC"), f"f{k}") for k in range(10)]
    df = pd.DataFrame(linhas, columns=["post_id", "published_at", "duplicate_group"])
    df["metric_target"] = 99  # a checagem não pode ler isto
    caminho = tmp_path / "mini.parquet"
    df.to_parquet(caminho)
    tudo = pd.DataFrame({"x": 1.0}, index=pd.Index(df["post_id"], name="post_id"))
    return {"caminho": caminho, "cfg_av": cfg_av, "ids": ids, "tudo": tudo}


def _bloco(ids):
    return pd.DataFrame({"v": 1.0}, index=pd.Index(ids, name="post_id"))


def test_cobertura_duplicata_nao_canonica_sem_extra_nao_bloqueia(mini, capsys):
    extra = _bloco(mini["ids"])  # nenhum "d*" tem o extra
    conferir_cobertura(mini["caminho"], mini["cfg_av"], {"duracoes": mini["tudo"]}, {"vlm": extra})
    assert "200/200 (100.0%)" in capsys.readouterr().out  # a cobertura é sempre impressa


def test_cobertura_canonico_sem_duracao_bloqueia(mini):
    dur = mini["tudo"].drop(index=["p007"])
    with pytest.raises(ValueError, match=r"NÃO foi aberto.*'duracoes': 1"):
        conferir_cobertura(mini["caminho"], mini["cfg_av"], {"duracoes": dur}, {})


def test_cobertura_do_extra_98_bloqueia_e_995_passa(mini, capsys):
    obrig = {"duracoes": mini["tudo"]}
    com_98 = _bloco(mini["ids"][:196])  # 196/200 = 98%
    with pytest.raises(ValueError, match=r"NÃO foi aberto.*vlm 98.0%"):
        conferir_cobertura(mini["caminho"], mini["cfg_av"], obrig, {"vlm": com_98})
    assert "196/200 (98.0%)" in capsys.readouterr().out  # impresso mesmo recusando
    com_995 = _bloco(mini["ids"][:199])  # 199/200 = 99,5%
    conferir_cobertura(mini["caminho"], mini["cfg_av"], obrig, {"vlm": com_995})
    assert "199/200 (99.5%)" in capsys.readouterr().out
