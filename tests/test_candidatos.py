from pathlib import Path

import pandas as pd

CAMINHO = Path(__file__).parents[1] / "data" / "candidatos.csv"


def checar_candidatos(df: pd.DataFrame) -> None:
    """A lista manual segue o mix do plano; as linhas da expansão (colab, destaque) chegam com
    setor e tipo vazios, que a revisão humana preenche na planilha de curadoria."""
    assert list(df.columns) == ["handle", "nome", "setor", "tipo", "fonte", "observacao"]
    assert df["handle"].str.startswith("@").all()
    assert df["handle"].str.lower().is_unique
    assert set(df["fonte"]) <= {"lista manual", "colab", "destaque"}
    manual = df[df["fonte"] == "lista manual"]
    assert set(manual["setor"]) == {"beleza", "automotivo"}
    assert set(manual["tipo"]) == {"marca", "criador"}
    assert len(manual) >= 90  # aula 1.4: pelo menos 90 candidatos
    por_celula = manual.groupby(["setor", "tipo"]).size()
    assert len(por_celula) == 4 and (por_celula >= 15).all()
    assert (manual["nome"] != "").all()  # o nome alimenta a busca de reserva
    expansao = df[df["fonte"] != "lista manual"]
    assert set(expansao["setor"]) <= {"", "beleza", "automotivo"}
    assert set(expansao["tipo"]) <= {"", "marca", "criador"}


def test_lista_de_candidatos_tem_o_formato_e_o_mix_do_plano():
    checar_candidatos(pd.read_csv(CAMINHO, dtype=str).fillna(""))


def test_lista_continua_valida_depois_da_expansao():
    df = pd.read_csv(CAMINHO, dtype=str).fillna("")
    expansao = pd.DataFrame(
        [
            {
                "handle": "@garagem22",
                "nome": "",
                "setor": "",
                "tipo": "",
                "fonte": "colab",
                "observacao": "citado 2x nas descrições da base",
            },
            {
                "handle": "@destaque",
                "nome": "Destaque",
                "setor": "",
                "tipo": "",
                "fonte": "destaque",
                "observacao": "em destaque na página de um incluído",
            },
        ]
    )
    checar_candidatos(pd.concat([df, expansao], ignore_index=True))
