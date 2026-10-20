import pandas as pd
import pytest

from previa.dados import carregar_base


@pytest.fixture
def arquivos(tmp_path):
    base = pd.DataFrame(
        {
            "post_id": ["antes000000", "nafronteira", "depois00000"],
            "published_at": pd.to_datetime(
                ["2026-03-31T23:59:00Z", "2026-04-01T00:00:00Z", "2026-05-10T19:00:00Z"]
            ),
        }
    )
    base.to_parquet(tmp_path / "base.parquet", index=False)
    (tmp_path / "avaliacao.yaml").write_text('teste_inicio: "2026-04-01"\n')
    return {
        "caminho_base": tmp_path / "base.parquet",
        "caminho_config": tmp_path / "avaliacao.yaml",
        "registro": tmp_path / "logs" / "teste_aberto.log",
    }


def test_desenvolvimento_nao_ve_o_periodo_de_teste(arquivos):
    """Aulas 1.2 e 2.3: o teste é separado no dia 1 e só abre no fim."""
    base = carregar_base(**arquivos)
    assert base["post_id"].tolist() == ["antes000000"]  # a fronteira já é teste
    assert not arquivos["registro"].exists()


def test_avaliacao_final_abre_o_teste_com_aviso_e_registro(arquivos):
    with pytest.warns(UserWarning, match="teste"):
        base = carregar_base(final=True, **arquivos)
    assert base["post_id"].tolist() == ["antes000000", "nafronteira", "depois00000"]
    with pytest.warns(UserWarning):
        carregar_base(final=True, **arquivos)
    assert len(arquivos["registro"].read_text().splitlines()) == 2  # cada abertura conta


def test_le_a_base_congelada_do_config(arquivos, tmp_path):
    """Aula 2.5, erro comum 2: os experimentos leem a cópia congelada que o config aponta."""
    congelada = tmp_path / "base_v1.parquet"
    pd.read_parquet(arquivos["caminho_base"]).iloc[:1].to_parquet(congelada, index=False)
    arquivos["caminho_config"].write_text(f'teste_inicio: "2026-04-01"\nbase: "{congelada}"\n')
    base = carregar_base(caminho_config=arquivos["caminho_config"], registro=arquivos["registro"])
    assert base["post_id"].tolist() == ["antes000000"]
    sem_chave = tmp_path / "sem_base.yaml"
    sem_chave.write_text('teste_inicio: "2026-04-01"\n')
    with pytest.raises(KeyError, match="base"):  # falha nunca silenciosa: sem cair na base viva
        carregar_base(caminho_config=sem_chave, registro=arquivos["registro"])
