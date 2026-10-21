"""Contrato de `prever_criativo` (A1; aula 6.1). Os pedidos 1 a 8 são os do caso à mão."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from previa import schemas
from previa.schemas import PredictionRequest, PredictionResponse

AGORA = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
SP = timezone(timedelta(hours=-3))


@pytest.fixture(autouse=True)
def _agora_fixo(monkeypatch):
    monkeypatch.setattr(schemas, "agora", lambda: AGORA)


def base(**mudancas):
    pedido = dict(
        creative={"tipo": "video", "arquivo": "lumina_b.mp4", "duration_s": 20, "vertical": True},
        caption="1 produto, 20 segundos #antesedepois",
        planned_datetime="2026-10-23T19:00:00-03:00",
        channel_id="UCabc",
        sector="beleza",
    )
    pedido.update(mudancas)
    return pedido


def posts(n=3):
    return [
        {
            "published_at": (datetime(2026, 1, 1, 12, tzinfo=SP) + timedelta(days=i)).isoformat(),
            "likes": 100 * (i + 1),
        }
        for i in range(n)
    ]


def recusa(**mudancas):
    with pytest.raises(ValidationError) as e:
        PredictionRequest(**base(**mudancas))
    return str(e.value)


def test_1_pedido_base_passa():
    assert PredictionRequest(**base()).sector == "beleza"


def test_2_data_sem_fuso_recusa():
    assert "fuso" in recusa(planned_datetime="2026-10-23T19:00:00")


def test_3_setor_fora_da_lista_recusa():
    assert "beleza" in recusa(sector="moda")


def test_4_campo_extra_recusa():
    assert "curtidas_primeira_hora" in recusa(curtidas_primeira_hora=900)


def test_5_data_a_120_dias_recusa():
    msg = recusa(planned_datetime=(AGORA + timedelta(days=120)).isoformat())
    assert "90 dias" in msg


def test_5b_noventa_dias_exatos_passa():
    PredictionRequest(**base(planned_datetime=(AGORA + timedelta(days=90)).isoformat()))


def test_6_data_no_passado_passa():
    PredictionRequest(**base(planned_datetime="2026-01-05T19:00:00-03:00"))


def test_7a_channel_stats_com_posts_datados_passa():
    r = PredictionRequest(
        **base(channel_id=None, channel_stats={"subscribers": 5000, "posts": posts(10)})
    )
    assert len(r.channel_stats.posts) == 10


def test_7b_curtidas_soltas_sem_data_recusa():
    msg = recusa(channel_id=None, channel_stats={"subscribers": 5000, "posts": [{"likes": 10}] * 3})
    assert "published_at" in msg


def test_8_canal_com_2_posts_passa_no_contrato():
    PredictionRequest(**base(channel_id=None, channel_stats={"posts": posts(2)}))


def test_channel_stats_sem_posts_e_sem_inscritos_recusa():
    msg = recusa(channel_id=None, channel_stats={"posts": []})
    assert "inscritos" in msg and "posts" in msg


def test_channel_stats_so_inscritos_passa():
    PredictionRequest(**base(channel_id=None, channel_stats={"subscribers": 800, "posts": []}))


def test_inscritos_negativos_recusa():
    recusa(channel_id=None, channel_stats={"subscribers": -1, "posts": []})


def test_f10_curtidas_ocultas_passam():
    p = [{"published_at": "2026-03-01T10:00:00-03:00", "likes": None}]
    PredictionRequest(**base(channel_id=None, channel_stats={"posts": p}))


def test_channel_id_e_channel_stats_juntos_recusa():
    msg = recusa(channel_stats={"subscribers": 10, "posts": []})
    assert "apenas um" in msg


def test_nenhum_dos_dois_recusa():
    msg = recusa(channel_id=None)
    assert "channel_id" in msg and "channel_stats" in msg


def test_f34_video_de_200s_recusa():
    c = {"tipo": "video", "arquivo": "x.mp4", "duration_s": 200, "vertical": True}
    assert "180" in recusa(creative=c)


def test_f34_video_horizontal_recusa():
    c = {"tipo": "video", "arquivo": "x.mp4", "duration_s": 30, "vertical": False}
    assert "vertical" in recusa(creative=c)


def test_video_sem_duracao_nem_orientacao_passa():
    PredictionRequest(**base(creative={"tipo": "video", "arquivo": "x.mp4"}))


def test_tipo_texto_sem_arquivo_passa():
    PredictionRequest(**base(creative={"tipo": "texto"}))


def test_video_sem_arquivo_recusa():
    assert "arquivo" in recusa(creative={"tipo": "video"})


def test_tipo_desconhecido_recusa():
    recusa(creative={"tipo": "audio", "arquivo": "x.mp3"})


def test_hora_local_preservada():
    r = PredictionRequest(**base(planned_datetime="2026-10-22T22:00:00Z"))
    assert r.planned_datetime.astimezone(SP).hour == 19
    r2 = PredictionRequest(**base())
    assert r2.planned_datetime.hour == 19 and r2.planned_datetime.utcoffset() == timedelta(hours=-3)


def test_metrica_e_nivel_fixos():
    r = PredictionRequest(**base())
    assert r.metric == "likes" and r.interval_level == 0.8
    recusa(metric="views")
    recusa(interval_level=0.9)


def test_ida_e_volta_pelo_json():
    r = PredictionRequest(
        **base(
            channel_id=None,
            channel_stats={"subscribers": 5, "posts": posts(3)},
            cover="capa.png",
        )
    )
    assert PredictionRequest.model_validate_json(r.model_dump_json()) == r


# ---- resposta ----


def resposta(**mudancas):
    r = dict(
        predicted_metric=1000.0,
        interval_low=600.0,
        interval_high=1800.0,
        relative_to_channel=1.2,
        confidence_flag={"baixa_confianca": False, "motivos": []},
        top_drivers=[
            {
                "feature": "hora",
                "valor": 19,
                "contribuicao_log": 0.12,
                "multiplicador": 1.13,
                "direcao": "aumenta",
            }
        ],
        comparables=[
            {
                "post_id": "abc",
                "published_at": "2026-03-01T10:00:00-03:00",
                "likes": 900,
                "similaridade": 0.8,
            }
        ],
        explanation={"resumo": "r", "fatores": ["f"], "ressalvas": []},
        features_used=["hora"],
        model_version={
            "versao": "v1",
            "run_id": "abcdef123456",
            "ultimo_treino": "2025-12-31",
            "base": "v2",
            "hash_base": "96fd13a28156",
            "rotulo": "v1 · run abcdef12",
        },
    )
    r.update(mudancas)
    return r


def test_resposta_valida_e_ida_e_volta():
    r = PredictionResponse(**resposta())
    assert PredictionResponse.model_validate_json(r.model_dump_json()) == r


def test_resposta_com_intervalo_incoerente_recusa():
    with pytest.raises(ValidationError, match="interval_low"):
        PredictionResponse(**resposta(interval_low=1200.0))
    with pytest.raises(ValidationError, match="interval_high"):
        PredictionResponse(**resposta(interval_high=900.0))


def test_resposta_sem_model_version_recusa():
    d = resposta()
    del d["model_version"]
    with pytest.raises(ValidationError, match="model_version"):
        PredictionResponse(**d)


def test_resposta_model_version_incompleto_recusa():
    d = resposta(model_version={"versao": "v1"})
    with pytest.raises(ValidationError, match="run_id"):
        PredictionResponse(**d)


def test_flag_com_motivo_exige_baixa_confianca_e_vice_versa():
    m = [{"codigo": "canal_novo", "texto": "o canal tem só 2 posts anteriores"}]
    with pytest.raises(ValidationError):
        PredictionResponse(**resposta(confidence_flag={"baixa_confianca": False, "motivos": m}))
    with pytest.raises(ValidationError):
        PredictionResponse(**resposta(confidence_flag={"baixa_confianca": True, "motivos": []}))
    PredictionResponse(**resposta(confidence_flag={"baixa_confianca": True, "motivos": m}))


def test_codigo_de_motivo_fora_dos_6_recusa():
    m = [{"codigo": "inventado", "texto": "x"}]
    with pytest.raises(ValidationError):
        PredictionResponse(**resposta(confidence_flag={"baixa_confianca": True, "motivos": m}))


def test_mais_de_5_drivers_recusa():
    d = resposta()["top_drivers"] * 6
    with pytest.raises(ValidationError):
        PredictionResponse(**resposta(top_drivers=d))


def test_previsao_negativa_recusa_e_campo_extra_recusa():
    with pytest.raises(ValidationError):
        PredictionResponse(**resposta(predicted_metric=-1.0, interval_low=-2.0))
    with pytest.raises(ValidationError):
        PredictionResponse(**resposta(nota=4))
