"""Testes da abertura dos resultados lacrados (#36; aulas 2.4, 2.5, 2.6, 4.8).

Runs falsos no MLflow temporário e uma tabela de ablação sintética com as colunas da real
(`previa.ablacao.ablar`); nenhum resultado real é lido.
"""

import time

import mlflow
import pandas as pd
import pytest
import yaml
from test_experimento import mlflow_temporario  # noqa: F401

from previa.abertura import abrir, argumento_unico, run_da_config, veredito_27, veredito_36
from previa.caminhos import CONFIGS
from previa.experimento import EXPERIMENTO


def ganhos(g, glo, ghi, d, dlo, dhi):
    """A coluna de uma avaliação, como o `comparar` devolve (índice = nome da métrica)."""
    return pd.Series(
        {
            "ganho": g,
            "ic_ganho_lo": glo,
            "ic_ganho_hi": ghi,
            "dif_mae_log": d,
            "ic_dif_mae_lo": dlo,
            "ic_dif_mae_hi": dhi,
        }
    )


def test_veredito_27_fica_com_ic_sem_zero_sem_piorar_a_outra():
    # manchete melhora (IC sem o zero); o mae_log não muda (IC cruza o zero): fica
    assert veredito_27(ganhos(0.02, 0.01, 0.03, 0.0, -0.02, 0.02), "padrao")
    # o mae_log melhora (dif < 0, IC todo abaixo de zero); a manchete cruza o zero: fica
    assert veredito_27(ganhos(0.0, -0.02, 0.02, -0.05, -0.08, -0.02), "padrao")
    # nenhum dos dois com IC sem o zero: não fica
    assert not veredito_27(ganhos(0.01, -0.01, 0.03, -0.01, -0.04, 0.02), "padrao")
    # H11/H8/H9: decide o mae_log; a manchete sozinha não basta
    assert not veredito_27(ganhos(0.02, 0.01, 0.03, 0.0, -0.02, 0.02), "mae_log")
    assert veredito_27(ganhos(0.0, -0.02, 0.02, -0.05, -0.08, -0.02), "mae_log")
    # a ilustração nunca é candidata
    assert not veredito_27(ganhos(0.02, 0.01, 0.03, -0.05, -0.08, -0.02), "ilustracao")


def test_veredito_27_nao_fica_se_piora_a_outra():
    # mae_log melhora, mas o IC da manchete fica inteiro do lado ruim: piora a outra
    assert not veredito_27(ganhos(-0.02, -0.03, -0.01, -0.05, -0.08, -0.02), "padrao")
    assert not veredito_27(ganhos(-0.02, -0.03, -0.01, -0.05, -0.08, -0.02), "mae_log")
    # manchete melhora, mas o mae_log piora (IC todo acima de zero): não fica
    assert not veredito_27(ganhos(0.02, 0.01, 0.03, 0.05, 0.02, 0.08), "padrao")
    # o IC da outra só cruza o zero: ainda não é "piorar"
    assert veredito_27(ganhos(-0.01, -0.03, 0.01, -0.05, -0.08, -0.02), "mae_log")
    # H6 no walk-forward: não piorar em nenhuma das duas, sem exigir ganho
    assert veredito_27(ganhos(-0.01, -0.03, 0.01, 0.01, -0.02, 0.04), "nao_piorar")
    assert not veredito_27(ganhos(-0.02, -0.03, -0.01, 0.0, -0.02, 0.02), "nao_piorar")
    assert not veredito_27(ganhos(0.0, -0.02, 0.02, 0.05, 0.02, 0.08), "nao_piorar")


def _linha(desenho, bloco, a, b, g, glo, ghi, d, dlo, dhi, avaliacao="walk_forward"):
    # as colunas de `ablar`: desenho, bloco, a, b, avaliacao, as de `ganho_entre`, n_features_b
    return {
        "desenho": desenho,
        "bloco": bloco,
        "a": a,
        "b": b,
        "avaliacao": avaliacao,
        "manchete_a": 0.5,
        "manchete_b": 0.5 + g,
        "ganho": g,
        "ic_ganho_lo": glo,
        "ic_ganho_hi": ghi,
        "mae_log_a": 1.0,
        "mae_log_b": 1.0 + d,
        "dif_mae_log": d,
        "ic_dif_mae_lo": dlo,
        "ic_dif_mae_hi": dhi,
        "linhas": 1000,
        "n_features_b": 10,
    }


def _ablacao():
    """Quatro blocos: x sólido nas três leituras; y passa na escada mas some na remoção; z não
    passa em nada; w passa nas duas mas não supera o ruído."""
    forte = (0.04, 0.02, 0.06, -0.10, -0.14, -0.06)
    fraco = (0.0, -0.02, 0.02, 0.0, -0.03, 0.03)
    ruido = (0.01, -0.01, 0.03, 0.01, -0.01, 0.03)
    quase = (0.008, 0.002, 0.014, -0.004, -0.007, -0.001)
    linhas = []
    for bloco, escada, remocao in (
        ("x", forte, forte),
        ("y", forte, fraco),
        ("z", fraco, fraco),
        ("w", quase, quase),
    ):
        linhas.append(_linha("cumulativa", bloco, "ant", f"escada-{bloco}", *escada))
        linhas.append(_linha("todos menos um", bloco, f"sem-{bloco}", "completo", *remocao))
    linhas.append(_linha("controle negativo", "ruido", "completo", "completo+ruido", *ruido))
    # no canal novo o x some (a leitura é por avaliação)
    linhas.append(_linha("cumulativa", "x", "ant", "escada-x", *fraco, avaliacao="canal_novo"))
    linhas.append(
        _linha("todos menos um", "x", "sem-x", "completo", *forte, avaliacao="canal_novo")
    )
    linhas.append(
        _linha("controle negativo", "ruido", "completo", "completo+ruido", *ruido, "canal_novo")
    )
    return pd.DataFrame(linhas)


def test_veredito_36_argumento_so_com_as_tres():
    v = veredito_36(_ablacao())
    wf = v[v["avaliacao"] == "walk_forward"].drop_duplicates("bloco").set_index("bloco")
    assert wf.loc["x", "argumento"] == "sim"  # escada e remoção sem o zero, acima do ruído
    assert wf.loc["y", "argumento"] == "não"  # passa na escada, some na remoção
    assert wf.loc["z", "argumento"] == "não"
    assert wf.loc["w", "argumento"] == "não"  # sólido, mas não acima do ruído
    assert "veredito" not in v.columns  # "sai"/"no modelo" é da #27 da hipótese, não da ablação
    cn = v[v["avaliacao"] == "canal_novo"].drop_duplicates("bloco").set_index("bloco")
    assert cn.loc["x", "argumento"] == "não"  # cada avaliação tem a sua leitura
    x = v[(v["bloco"] == "x") & (v["avaliacao"] == "walk_forward")]
    assert list(x["leitura"]) == ["escada", "remocao", "ruido"]


def test_argumento_unico_h10_escada_igual_remocao():
    rui = pd.Series(
        _linha("controle negativo", "ruido", "c", "c+r", 0.01, -0.01, 0.03, 0, -0.02, 0.02)
    )
    forte = ganhos(0.04, 0.02, 0.06, 0.0, -0.02, 0.02)
    fraco_ruido = ganhos(0.005, 0.001, 0.009, 0.0, -0.02, 0.02)  # sem o zero, mas abaixo do ruído
    cruza = ganhos(0.04, -0.02, 0.06, 0.0, -0.02, 0.02)  # efeito grande, IC com o zero
    assert argumento_unico(forte, rui)
    assert not argumento_unico(fraco_ruido, rui)
    assert not argumento_unico(cruza, rui)


def test_veredito_36_falha_alto_sem_o_controle_de_ruido():
    sem = _ablacao()
    sem = sem[sem["desenho"] != "controle negativo"]
    with pytest.raises(ValueError, match="controle negativo"):
        veredito_36(sem)


def test_run_da_config_pega_o_mais_recente(mlflow_temporario):  # noqa: F811
    mlflow.set_experiment(EXPERIMENTO)
    ids = []
    for config in ("configs/experimentos/a.yaml", "configs/experimentos/a.yaml", "outra.yaml"):
        with mlflow.start_run() as run:
            mlflow.set_tag("config", config)
        ids.append(run.info.run_id)
        time.sleep(0.02)
    # dois runs da mesma config: o mais antigo (ids[0]) perde; o último de outra config não conta
    assert run_da_config("configs/experimentos/a.yaml") == ids[1]
    assert run_da_config("outra.yaml") == ids[2]
    with mlflow.start_run() as falho:  # um run mais novo que falhou: não conta
        mlflow.set_tag("config", "configs/experimentos/a.yaml")
        mlflow.end_run(status="FAILED")
    assert falho.info.run_id != ids[1]
    assert run_da_config("configs/experimentos/a.yaml") == ids[1]
    with pytest.raises(LookupError, match="nunca.yaml"):
        run_da_config("nunca.yaml")  # sem run: falha alto, não devolve nada


def test_abrir_usa_o_run_de_cada_config_e_o_veredito_da_hipotese(mlflow_temporario, tmp_path):  # noqa: F811
    mlflow.set_experiment(EXPERIMENTO)
    run_de = {}
    for config in ("c.yaml", "t.yaml"):
        with mlflow.start_run() as run:
            mlflow.set_tag("config", config)
        run_de[config] = run.info.run_id
        time.sleep(0.02)
    tabela = _ablacao()
    destino = tmp_path / "abl.csv"
    tabela.to_csv(destino, index=False)
    cfg = {
        "ablacao": str(destino),
        "blocos": {"x": "HX", "y": None, "z": "HY", "w": None, "unico": "HX"},
        "comparacoes": {
            "HX": {
                "controle": "c.yaml",
                "tratamento": "t.yaml",
                "regras": {"walk_forward": "padrao", "canal_novo": "nao_piorar"},
            },
            "HY": {
                "controle": "t.yaml",
                "tratamento": "c.yaml",
                "regras": {"walk_forward": "padrao"},
                "exige": "HX",
            },
        },
    }
    chamadas = []

    def falso(a, b, *args):
        chamadas.append((a, b))
        return pd.DataFrame(
            {
                "walk_forward": ganhos(0.03, 0.01, 0.05, 0.0, -0.02, 0.02),
                "canal_novo": ganhos(-0.03, -0.05, -0.01, 0.0, -0.02, 0.02),  # piora: HX não fica
            }
        )

    r = abrir(cfg, comparar_fn=falso)
    assert chamadas[0] == (run_de["c.yaml"], run_de["t.yaml"])
    hx = r[(r["tipo"] == "hipotese") & (r["id"] == "HX")].set_index("avaliacao")
    assert hx.loc["walk_forward", "veredito"] == "fica"
    assert hx.loc["canal_novo", "veredito"] == "não fica"
    assert set(hx["veredito_final"]) == {"não fica"}  # todas as avaliações têm de passar
    hy = r[(r["tipo"] == "hipotese") & (r["id"] == "HY")]
    assert set(hy["veredito_final"]) == {"não fica"}  # exige HX, que não ficou
    b = r[(r["tipo"] == "bloco") & (r["avaliacao"] == "walk_forward")]
    assert set(b["leitura"]) == {"escada", "remocao", "ruido"}
    por_bloco = b.drop_duplicates("bloco").set_index("bloco")
    assert por_bloco.loc["x", "argumento"] == "sim"
    assert por_bloco.loc["x", "modelo"] == "HX: não fica"  # vem da #27 da hipótese
    assert por_bloco.loc["y", "modelo"] == "base da Opção A (sem hipótese)"
    assert por_bloco.loc["z", "modelo"] == "HY: não fica"
    # o bloco que não está na ablação (H10): escada = remoção = a comparação da hipótese; o falso
    # dá ganho 0.03 (IC 0.01 a 0.05) no walk_forward, acima do ruído de 0.01
    assert por_bloco.loc["unico", "argumento"] == "sim"
    unico = b[b["bloco"] == "unico"]
    assert list(unico["leitura"]) == ["escada", "remocao", "ruido"]
    assert unico.iloc[0]["run_a"] == run_de["c.yaml"]  # os ids dos runs aparecem na saída


def test_configs_de_abertura_apontam_para_configs_que_existem():
    cfg = yaml.safe_load((CONFIGS / "abertura.yaml").read_text(encoding="utf-8"))
    nomes = {h: c for h, c in cfg["comparacoes"].items()}
    assert {"H2", "H3", "H4a", "H4b", "H5", "H6", "H7", "H8", "H9", "H10", "H11"} <= set(nomes)
    assert "H1" not in nomes
    blocos = cfg["blocos"]  # bloco da ablação -> comparação da hipótese (ou None: base da Opção A)
    assert {"historico", "texto", "capa", "detectores"} <= set(blocos)
    assert blocos["calendario"] is None and blocos["metadados"] is None
    assert blocos["detectores"] == "H10"
    assert all(h in nomes for h in blocos.values() if h)  # o teste final não faz parte da abertura
    for h, c in nomes.items():
        for chave in ("controle", "tratamento"):
            assert (CONFIGS.parent / c[chave]).exists(), f"{h}: {c[chave]} não existe"
        assert set(c["regras"].values()) <= {"padrao", "mae_log", "nao_piorar", "ilustracao"}
        if "exige" in c:
            assert c["exige"] in nomes
