from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa
import pytest

from previa.coletor import abrir_base, parse_video, salvar, salvar_canais
from previa.duplicatas import dhash
from previa.esquema import BaseShorts, Post, exportar_base_minima, montar_base_minima

from fakes import CANAL_ID, canal, thumbnail, thumbnail_por_id, video


def _rotulados():
    return pd.DataFrame(
        {
            "video_id": ["Lm9Qk2VbT0s", "aB3_x-Z9k0Q"],
            "channel_id": ["UCabc123LuminaBeauty0000"] * 2,
            "published_at": pd.to_datetime(["2026-08-01T22:00:05Z", "2026-08-03T19:00:00Z"]),
            "title": ["1 produto, 20 segundos", "Pele natural em 5 passos"],
            "description": ["Base nova #shorts", ""],
            "duration_s": [22.0, 55.0],
            "likes": pd.array([2410, 880], dtype="Int64"),
            "age_days": [31.2, 30.5],
            "collected_at": pd.to_datetime(["2026-09-01T03:10:00Z"] * 2),
        }
    )


def _inscritos():
    return pd.DataFrame(
        {
            "channel_id": ["UCabc123LuminaBeauty0000"],
            "collected_date": pd.to_datetime(["2026-09-01"]).date,
            "subscribers": pd.array([180000], dtype="Int64"),
        }
    )


def _thumbs(tmp_path, baixar=thumbnail_por_id):
    return {"cache": tmp_path / "thumb_hash.parquet", "baixar": baixar}


def test_base_minima_tem_o_esquema_do_briefing_e_valida():
    base = montar_base_minima(_rotulados(), {"UCabc123LuminaBeauty0000": "beleza"}, _inscritos())
    BaseShorts.validate(base, lazy=True)
    assert base.loc[0, "media_ref"] == "https://www.youtube.com/shorts/Lm9Qk2VbT0s"
    assert base.loc[0, "caption"] == "1 produto, 20 segundos\nBase nova #shorts"
    assert base.loc[0, "subscribers_at_collection"] == 180000


def test_base_minima_marca_as_copias_pela_regra_das_duplicatas():
    """Decisão #1 (aula 2.2): a base leva o duplicate_group de cada post. Os casos da regra
    estão em test_duplicatas.py; aqui, só a ligação com a base mínima."""
    outro = "UCdef456Garagem2200000000"
    rotulados = pd.DataFrame(
        {
            "video_id": ["bbbbbbbbbbb", "kkkkkkkkkkk", "ccccccccccc"],
            "channel_id": [CANAL_ID, CANAL_ID, outro],
            "published_at": pd.to_datetime(
                ["2023-10-24T12:00:00Z", "2023-09-04T12:00:00Z", "2023-09-10T12:00:00Z"]
            ),
            "title": ["Natura Essencial Ato"] * 3,
            "description": [""] * 3,
            "duration_s": [16.0] * 3,
            "likes": pd.array([10, 20, 30], dtype="Int64"),
            "age_days": [700.0, 750.0, 745.0],
            "collected_at": pd.to_datetime(["2026-10-05T08:00:00Z"] * 3),
        }
    )
    mesma_arte = dhash(thumbnail(1))
    base = montar_base_minima(
        rotulados,
        {CANAL_ID: "beleza", outro: "beleza"},
        _inscritos().iloc[0:0],
        hashes=dict.fromkeys(rotulados["video_id"], mesma_arte),
    )
    BaseShorts.validate(base, lazy=True)
    grupos = dict(zip(base["post_id"], base["duplicate_group"], strict=True))
    assert grupos == {
        "kkkkkkkkkkk": "kkkkkkkkkkk",  # a primeira cópia
        "bbbbbbbbbbb": "kkkkkkkkkkk",  # o reupload no mesmo canal
        "ccccccccccc": "kkkkkkkkkkk",  # o mesmo criativo em outro canal (decisão de 17/10)
    }


def test_pandera_pega_duplicata_e_setor_invalido():
    base = montar_base_minima(_rotulados(), {"UCabc123LuminaBeauty0000": "beleza"}, _inscritos())
    ruim = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    ruim.loc[0, "sector"] = "games"
    with pytest.raises(pa.errors.SchemaErrors):
        BaseShorts.validate(ruim, lazy=True)


def test_post_recusa_rotulo_jovem():
    registro = {
        "post_id": "Lm9Qk2VbT0s",
        "channel_id": "UCabc123LuminaBeauty0000",
        "published_at": "2026-08-01T22:00:05Z",
        "caption": "1 produto",
        "media_ref": "https://www.youtube.com/shorts/Lm9Qk2VbT0s",
        "metric_target": 2410,
        "target_age_days": 12,
        "sector": "beleza",
        "channel_stats_at_collection": {
            "subscribers": 180000,
            "video_count": 912,
            "collected_at": "2026-09-01T03:10:00Z",
        },
    }
    with pytest.raises(ValueError):
        Post.model_validate(registro)


def test_exportar_base_minima_de_uma_base_real_valida_e_grava(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    coletado = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    lote = [video(f"v{d}xxxxxxxxx", f"2026-08-0{d}T12:00:00Z", "PT22S") for d in range(1, 4)]
    salvar(con, [parse_video(item, coletado) for item in lote])
    salvar_canais(con, [canal()], coletado)
    planilha = tmp_path / "canais.csv"
    planilha.write_text(f"channel_id,setor,incluido\n{CANAL_ID},beleza,sim\n")
    destino = tmp_path / "base_minima.parquet"
    assert exportar_base_minima(con, planilha, destino, **_thumbs(tmp_path)) == 3
    base = pd.read_parquet(destino)
    assert set(base["sector"]) == {"beleza"}
    assert (base["subscribers_at_collection"] == 180000).all()


def test_exportar_base_minima_nao_aplica_o_teto_por_canal(tmp_path):
    """O teto (aula 1.4) decide as linhas de treino, não de quais posts sai o histórico do
    canal: ele vira parâmetro do experimento, e a base exportada leva todos os rotulados."""
    con = abrir_base(tmp_path / "t.duckdb")
    coletado = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    lote = [video(f"v{i:010d}", "2026-08-01T12:00:00Z", "PT22S") for i in range(401)]
    salvar(con, [parse_video(item, coletado) for item in lote])
    salvar_canais(con, [canal()], coletado)
    planilha = tmp_path / "canais.csv"
    planilha.write_text(f"channel_id,setor,incluido\n{CANAL_ID},beleza,sim\n")
    destino = tmp_path / "base_minima.parquet"
    assert exportar_base_minima(con, planilha, destino, **_thumbs(tmp_path)) == 401


def test_exportar_base_minima_deixa_fora_canais_excluidos_na_revisao(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    outro = "UCdef456Garagem2200000000"
    coletado = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    lote = [video(f"a{d}xxxxxxxxx", f"2026-08-0{d}T12:00:00Z", "PT22S") for d in range(1, 4)]
    lote += [
        video(f"b{d}xxxxxxxxx", f"2026-08-0{d}T12:00:00Z", "PT22S", canal_id=outro)
        for d in range(1, 4)
    ]
    salvar(con, [parse_video(item, coletado) for item in lote])
    salvar_canais(con, [canal(), canal(outro, handle="@garagem22")], coletado)
    planilha = tmp_path / "canais.csv"
    planilha.write_text(
        f"channel_id,setor,incluido\n{CANAL_ID},beleza,sim\n{outro},automotivo,não\n"
    )
    destino = tmp_path / "base_minima.parquet"
    assert exportar_base_minima(con, planilha, destino, **_thumbs(tmp_path)) == 3
    assert set(pd.read_parquet(destino)["channel_id"]) == {CANAL_ID}


def test_exportar_base_minima_marca_o_reupload_pela_thumbnail(tmp_path):
    """Os 3 vídeos do FakeYT têm o mesmo título e a mesma duração; só a thumbnail separa."""
    con = abrir_base(tmp_path / "t.duckdb")
    coletado = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    lote = [video(f"v{d}xxxxxxxxx", f"2026-08-0{d}T12:00:00Z", "PT22S") for d in range(1, 4)]
    salvar(con, [parse_video(item, coletado) for item in lote])
    salvar_canais(con, [canal()], coletado)
    planilha = tmp_path / "canais.csv"
    planilha.write_text(f"channel_id,setor,incluido\n{CANAL_ID},beleza,sim\n")
    destino = tmp_path / "base_minima.parquet"
    arte = {
        "v1xxxxxxxxx": thumbnail(1),
        "v2xxxxxxxxx": thumbnail(2),
        "v3xxxxxxxxx": thumbnail(1, 15),
    }
    exportar_base_minima(con, planilha, destino, **_thumbs(tmp_path, arte.__getitem__))
    base = pd.read_parquet(destino)
    assert dict(zip(base["post_id"], base["duplicate_group"], strict=True)) == {
        "v1xxxxxxxxx": "v1xxxxxxxxx",
        "v2xxxxxxxxx": "v2xxxxxxxxx",
        "v3xxxxxxxxx": "v1xxxxxxxxx",
    }
