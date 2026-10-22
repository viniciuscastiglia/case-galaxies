"""Testes da análise de erros (aula 2.6; insumo da M9)."""

import numpy as np
import pandas as pd
import pytest

from previa.erros import CORTES, agrupar, origem_do_canal, piores, por_corte, residuos


def _tabela():
    # o "viral" é subestimado por e^4 ≈ 55×; o "ruim" é superestimado por e^2 ≈ 7×
    return residuos(
        pd.DataFrame(
            {
                "post_id": ["v_ok_00000", "v_viral000", "v_ruim0000", "v_ok_11111"],
                "channel_id": ["UC1", "UC1", "UC2", "UC3"],
                "y": [5.0, 9.0, 4.0, 6.0],
                "p": [5.5, 5.0, 6.0, 6.0],
                "grupo": ["A", "A", "A", "B"],
            }
        )
    )


def test_residuo_positivo_quer_dizer_que_o_modelo_subestimou():
    t = _tabela()
    assert t["residuo"].tolist() == [-0.5, 4.0, -2.0, 0.0]
    assert t["direcao"].tolist() == ["superestimou", "subestimou", "superestimou", "exato"]
    assert t["fator"].round(2).tolist() == [1.65, 54.6, 7.39, 1.0]  # e^|e|: errou por quantas vezes


def test_piores_por_erro_absoluto_com_link_e_categoria_para_preencher():
    p = piores(_tabela(), n=2)
    assert p["post_id"].tolist() == ["v_viral000", "v_ruim0000"]
    assert p["url"].iloc[0] == "https://www.youtube.com/shorts/v_viral000"
    assert p["curtidas_reais"].tolist() == [round(np.expm1(9.0)), round(np.expm1(4.0))]
    assert p["curtidas_previstas"].iloc[0] == round(np.expm1(5.0))
    assert (p["categoria"] == "").all() and (p["nota"] == "").all()  # o Vinicius assiste e preenche


def test_por_corte_traz_n_canais_mae_e_vies():
    # 2.6: o grupo vem com o nº de posts e de canais, para não ler ruído como achado (erro comum 6)
    r = por_corte(_tabela(), "grupo")
    assert r.loc["A", "n"] == 3 and r.loc["A", "canais"] == 2
    assert r.loc["A", "mae"] == pytest.approx(6.5 / 3)
    assert r.loc["A", "vies"] == pytest.approx(1.5 / 3)
    assert r.loc["B", "n"] == 1 and r.loc["B", "mae"] == 0.0


def test_bordas_dos_cortes():
    t = pd.DataFrame(
        {"duracao_s": [19, 20, 181], "hist_n": [9, 10, 30], "hora_local": [11, 12, 23]}
    )
    # 181 s: a folga de 1 s da regra de Short (o arredondamento da duração da API)
    assert agrupar(t, "duracao_s").tolist() == ["< 20 s", "20 a 39 s", "60 s ou mais"]
    assert agrupar(t, "hist_n").tolist() == ["< 10 maduros", "10 a 29", "30+"]
    assert agrupar(t, "hora_local").tolist() == ["manhã", "tarde", "noite"]


def test_origem_do_canal_separa_a_busca():
    # 1.4: "o modelo erra mais nos canais que vieram da busca?"; a fonte é texto livre do canais.csv
    fonte = pd.Series(
        ["busca:canal", "lista manual+busca", "colab", "lista manual", None, "BUSCA:video"]
    )
    o = origem_do_canal(fonte)
    assert o.tolist() == [
        "busca",
        "busca",
        "fora da busca",
        "fora da busca",
        "sem registro",
        "busca",  # sem distinguir maiúsculas
    ]
    assert CORTES["origem"] == (None, None)  # o próprio valor é o grupo

    t = residuos(
        pd.DataFrame(
            {
                "channel_id": ["UC1", "UC1", "UC2", "UC3"],
                "y": [5.0, 7.0, 4.0, 6.0],
                "p": [5.5, 5.0, 6.0, 6.0],
                "origem": origem_do_canal(pd.Series(["busca:canal"] * 2 + ["lista manual"] * 2)),
            }
        )
    )
    r = por_corte(t, "origem")
    assert r.loc["busca", "n"] == 2 and r.loc["busca", "canais"] == 1
    assert r.loc["fora da busca", "n"] == 2 and r.loc["fora da busca", "canais"] == 2
    assert r.loc["busca", "mae"] == pytest.approx((0.5 + 2.0) / 2)
    assert r.loc["fora da busca", "vies"] == pytest.approx(-1.0)


def test_erros_confere_os_cortes(monkeypatch):
    import previa.erros as erros
    from previa.splits import ler_avaliacao

    class FakeRun:
        def __init__(self, run_id):
            self.data = type("Dados", (), {"tags": {"hash_base": "h", "data_corte": "2024-07-01"}})

    monkeypatch.setattr(erros.mlflow, "set_experiment", lambda nome: None)
    monkeypatch.setattr(erros.mlflow, "get_run", FakeRun)
    monkeypatch.setattr(erros, "carregar_base", lambda: pd.DataFrame())
    monkeypatch.setattr(erros, "hash_base", lambda b: "h")
    monkeypatch.setattr(erros, "ler_avaliacao", ler_avaliacao)
    with pytest.raises(ValueError, match="data_corte"):
        erros.tabela_do_run("A")
