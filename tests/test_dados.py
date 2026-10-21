import pandas as pd
import pytest
from pandera.errors import SchemaErrors

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


# --- a base v2 da #33 ---


def _base_minima_valida(n=12):
    """Uma base mínima no esquema do briefing (BaseShorts), pequena."""
    ids = [f"post{i:07d}" for i in range(n)]
    pub = pd.date_range("2026-01-01", periods=n, freq="7D", tz="UTC").astype("datetime64[us, UTC]")
    return pd.DataFrame(
        {
            "post_id": ids,
            "channel_id": "UC" + "a" * 22,
            "published_at": pub,
            "caption": "título\ndescrição",
            "media_ref": ["https://www.youtube.com/shorts/" + i for i in ids],
            "metric_target": pd.array([100 + i for i in range(n)], dtype="Int64"),
            "target_age_days": 40.0,
            "subscribers_at_collection": pd.array([5000] * n, dtype="Int64"),
            "collected_at": (pub + pd.Timedelta(days=40)).astype("datetime64[us, UTC]"),
            "sector": "beleza",
            "duplicate_group": ids,
        }
    )


def test_congelar_copia_e_devolve_o_hash(tmp_path):
    from previa.dados import congelar
    from previa.experimento import hash_base

    origem, destino = tmp_path / "base_minima.parquet", tmp_path / "base_v2.parquet"
    base = _base_minima_valida()
    base.to_parquet(origem, index=False)
    h = congelar(origem, destino)
    assert destino.exists()
    assert h == hash_base(pd.read_parquet(destino)) == hash_base(pd.read_parquet(origem))


def test_congelar_recusa_sobrescrever(tmp_path):
    from previa.dados import congelar

    origem, destino = tmp_path / "base_minima.parquet", tmp_path / "base_v2.parquet"
    _base_minima_valida().to_parquet(origem, index=False)
    destino.write_bytes(b"v1 intocada")
    with pytest.raises(FileExistsError, match="base_v2"):
        congelar(origem, destino)
    assert destino.read_bytes() == b"v1 intocada"


def test_congelar_recusa_base_fora_do_esquema(tmp_path):
    from previa.dados import congelar

    origem, destino = tmp_path / "base_minima.parquet", tmp_path / "base_v2.parquet"
    _base_minima_valida().drop(columns=["caption"]).to_parquet(origem, index=False)
    with pytest.raises(SchemaErrors):
        congelar(origem, destino)
    assert not destino.exists()


def test_avaliacao_v2_nao_mexe_na_v1():
    import yaml

    from previa.caminhos import CONFIGS

    v1 = yaml.safe_load((CONFIGS / "avaliacao.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((CONFIGS / "avaliacao_v2.yaml").read_text(encoding="utf-8"))
    assert v1["base"] == "base_v1.parquet"
    assert v2["base"] == "base_v2.parquet"


def test_v2_sem_teste():
    from previa.caminhos import CONFIGS
    from previa.splits import ler_avaliacao, split_treino_cal_teste

    cfg = ler_avaliacao(CONFIGS / "avaliacao_v2.yaml")
    datas = pd.date_range("2026-05-20", "2026-09-20", freq="5D", tz="UTC")
    df = pd.DataFrame({"published_at": datas})
    treino, cal, teste = split_treino_cal_teste(df, cfg)
    assert len(teste) == 0
    assert len(cal) > 0
    assert df.loc[cal, "published_at"].min() >= pd.Timestamp("2026-07-01", tz="UTC")
    assert df.loc[cal, "published_at"].max() < pd.Timestamp("2026-09-15", tz="UTC")
    assert df.loc[treino, "published_at"].max() < pd.Timestamp("2026-06-01", tz="UTC")


def _arquivos_v2(tmp_path):
    base = pd.DataFrame(
        {
            "post_id": ["abril000000", "agosto00000"],
            "published_at": pd.to_datetime(["2026-04-10T00:00:00Z", "2026-08-10T00:00:00Z"]),
        }
    )
    base.to_parquet(tmp_path / "base.parquet", index=False)
    (tmp_path / "avaliacao_v2.yaml").write_text(
        'teste_inicio: "2026-09-15"\nexige_teste_aberto: true\n'
    )
    return {
        "caminho_base": tmp_path / "base.parquet",
        "caminho_config": tmp_path / "avaliacao_v2.yaml",
        "registro": tmp_path / "logs" / "teste_aberto.log",
    }


def test_v2_recusa_antes_do_teste_final(tmp_path):
    """#33: a v2 gasta os meses de teste da v1; antes da avaliação final, recusa."""
    arq = _arquivos_v2(tmp_path)
    with pytest.raises(ValueError, match="previa.final"):
        carregar_base(**arq)
    arq["registro"].parent.mkdir(parents=True)
    arq["registro"].write_text("")  # registro vazio também recusa
    with pytest.raises(ValueError, match="teste"):
        carregar_base(**arq)


def test_v2_carrega_depois_do_teste_final(tmp_path):
    arq = _arquivos_v2(tmp_path)
    arq["registro"].parent.mkdir(parents=True)
    arq["registro"].write_text("2026-10-21T20:00:00+00:00 teste aberto (x)\n")
    assert len(carregar_base(**arq)) == 2


def test_avaliacao_v2_exige_teste_aberto():
    import yaml

    from previa.caminhos import CONFIGS

    v2 = yaml.safe_load((CONFIGS / "avaliacao_v2.yaml").read_text(encoding="utf-8"))
    assert v2["exige_teste_aberto"] is True
