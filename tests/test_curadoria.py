from datetime import datetime, timezone

import httplib2
import pandas as pd
import pytest

from previa.coletor import Cliente, abrir_base, parse_video, salvar
from previa.curadoria import (
    aplicar_criterios,
    atualizar_em_lote,
    avaliar_canal,
    avaliar_candidatos,
    canais_em_destaque,
    estratificar,
    expandir,
    handles_em_textos,
    normalizar_handle,
    novos_candidatos,
    resolver_handles,
)

from fakes import CANAL_ID, FakeYT, canal, erro, pagina, video

HOJE = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
UPLOADS = "UU" + CANAL_ID[2:]


def _candidato(handle, nome="", setor="beleza", tipo="marca"):
    return {
        "handle": handle,
        "nome": nome,
        "setor": setor,
        "tipo": tipo,
        "fonte": "lista manual",
        "observacao": "",
    }


def _canal_com_uploads(n_shorts, n_longos=10, publicado="2026-09-20T12:00:00Z"):
    ids_s = [f"s{i:03d}" for i in range(n_shorts)]
    ids_l = [f"l{i:03d}" for i in range(n_longos)]
    itens = {v: video(v, publicado, "PT30S") for v in ids_s}
    itens |= {v: video(v, publicado, "PT12M") for v in ids_l}
    todos = ids_s + ids_l
    blocos = [todos[i : i + 50] for i in range(0, len(todos), 50)]
    paginas = {}
    for k, bloco in enumerate(blocos):
        prox = f"p{k + 1}" if k + 1 < len(blocos) else None
        paginas["p0" if k == 0 else f"p{k}"] = pagina(bloco, prox)
    return FakeYT(paginas={UPLOADS: paginas}, itens=itens, canais={"@luminabeauty": canal()})


@pytest.mark.parametrize(
    "entrada, esperado",
    [
        ("@luminabeauty", "@luminabeauty"),
        ("LuminaBeauty", "@luminabeauty"),
        ("https://www.youtube.com/@LuminaBeauty/shorts", "@luminabeauty"),
        (" @Lumina.Beauty ", "@lumina.beauty"),
    ],
)
def test_normalizar_handle(entrada, esperado):
    assert normalizar_handle(entrada) == esperado


def test_resolver_handles_custa_uma_unidade_por_handle():
    yt = FakeYT(canais={"@luminabeauty": canal()})
    df, faltando = resolver_handles(Cliente(yt, arquivo=None), ["@luminabeauty", "@nao-existe"])
    assert faltando == ["@nao-existe"]
    assert df.loc[0, "inscritos"] == 180000 and df.loc[0, "uploads"].startswith("UU")
    assert yt.n("channels") == 2


def test_atualizar_em_lote_50_por_chamada():
    ids = [f"UC{i:022d}" for i in range(120)]
    yt = FakeYT(canais={i: canal(i) for i in ids})
    assert len(atualizar_em_lote(Cliente(yt, arquivo=None), ids)) == 120
    assert yt.n("channels") == 3


def test_handles_em_textos_conta_mencoes_e_ignora_email():
    descricoes = [
        "Look do dia com a @MariaModa e a @lumina.beauty! Contato: contato@marca.com.br",
        "Test drive do SUV novo com @garagem22 #shorts",
        "Tutorial feito em parceria com @lumina.beauty",
    ]
    assert handles_em_textos(descricoes).to_dict() == {
        "@lumina.beauty": 2,
        "@mariamoda": 1,
        "@garagem22": 1,
    }


def test_canais_em_destaque_so_secoes_multiple_channels():
    secoes = {
        "items": [
            {
                "snippet": {"type": "multipleChannels"},
                "contentDetails": {"channels": ["UCa", "UCb"]},
            },
            {"snippet": {"type": "singlePlaylist"}, "contentDetails": {"playlists": ["PL1"]}},
        ]
    }
    yt = FakeYT(secoes={CANAL_ID: secoes})
    assert canais_em_destaque(Cliente(yt, arquivo=None), CANAL_ID) == ["UCa", "UCb"]


def test_avaliar_canal_mede_shorts_atividade_e_curtidas_visiveis():
    yt = _canal_com_uploads(40)
    med = avaliar_canal(Cliente(yt, arquivo=None), {"channel_id": CANAL_ID}, HOJE)
    assert med["n_shorts_estimado"] == 40 and med["frac_curtidas_visiveis"] == 1.0
    assert med["ultimo_short"] == "2026-09-20" and med["leu_tudo"]


@pytest.mark.parametrize(
    "mudanca, esperado",
    [
        ({}, ("sim", "passa nos critérios automáticos")),
        ({"pais": "US"}, ("não", "país US")),
        ({"inscritos_verificacao": None}, ("não", "inscritos ocultos")),
        ({"n_shorts_estimado": 12}, ("não", "menos de 30 Shorts (12)")),
        (
            {"n_shorts_estimado": 12, "leu_tudo": False},
            ("revisar", "12 Shorts entre os 500 uploads mais recentes"),
        ),
        ({"ultimo_short": "2026-05-01"}, ("não", "sem Short há mais de 90 dias")),
        ({"frac_curtidas_visiveis": 0.2}, ("não", "curtidas ocultas")),
        ({"pais": None}, ("revisar", "país não informado: conferir idioma e país")),
    ],
)
def test_aplicar_criterios(mudanca, esperado):
    linha = {
        "pais": "BR",
        "inscritos_verificacao": 180000,
        "n_shorts_estimado": 40,
        "leu_tudo": True,
        "ultimo_short": "2026-09-20",
        "frac_curtidas_visiveis": 1.0,
    }
    assert aplicar_criterios(linha | mudanca, HOJE) == esperado


def test_avaliar_candidatos_registra_incluidos_e_excluidos():
    cand = pd.DataFrame(
        [_candidato("LuminaBeauty", "Lumina Beauty"), _candidato("@nao-existe", tipo="criador")]
    )
    planilha = avaliar_candidatos(Cliente(_canal_com_uploads(40), arquivo=None), cand, HOJE)
    por_handle = planilha.set_index("handle")
    assert por_handle.loc["@luminabeauty", "incluido"] == "sim"
    assert por_handle.loc["@luminabeauty", "faixa_inscritos"] == "100 mil–1 mi"
    assert por_handle.loc["@nao-existe", "motivo"] == "handle não encontrado"


def test_busca_de_reserva_acha_o_canal_mas_exige_revisao():
    yt = _canal_com_uploads(40)
    yt.canais = {CANAL_ID: canal()}  # o handle da lista está errado; o ID existe
    yt.buscas = {"Lumina Beauty": CANAL_ID}
    cand = pd.DataFrame([_candidato("@lumina-errado", "Lumina Beauty")])
    planilha = avaliar_candidatos(
        Cliente(yt, arquivo=None), cand, HOJE, cli_busca=Cliente(yt, arquivo=None)
    )
    linha = planilha.iloc[0]
    assert linha["incluido"] == "revisar" and linha["fonte"] == "lista manual+busca"
    assert "@luminabeauty" in linha["observacao"]


def test_novos_candidatos_nao_reavalia_o_que_ja_foi_revisado():
    planilha = pd.DataFrame({"handle": ["@luminabeauty"], "incluido": ["não"]})
    cand = pd.DataFrame({"handle": ["LuminaBeauty", "@garagem22"]})
    assert novos_candidatos(cand, planilha)["handle"].tolist() == ["@garagem22"]


def test_estratificar_por_setor_tipo_e_faixa():
    canais = pd.DataFrame(
        {
            "setor": ["beleza", "beleza", "automotivo"],
            "tipo": ["marca", "criador", "marca"],
            "inscritos": [180000, 5000, 2_000_000],
        }
    )
    assert estratificar(canais).loc[("beleza", "marca"), "100 mil–1 mi"] == 1


def test_expandir_acha_colabs_e_canais_em_destaque(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    item = video("s1", "2026-09-01T12:00:00Z", "PT30S")
    item["snippet"]["description"] = "Parceria com @Garagem22 e @luminabeauty"
    salvar(con, [parse_video(item, HOJE)])
    destaque = "UCdestaque000000000000000"
    secao = {"snippet": {"type": "multipleChannels"}, "contentDetails": {"channels": [destaque]}}
    yt = FakeYT(
        secoes={CANAL_ID: {"items": [secao]}},
        canais={destaque: canal(destaque, handle="@destaque")},
    )
    planilha = pd.DataFrame(
        {"channel_id": [CANAL_ID], "handle": ["@luminabeauty"], "incluido": ["sim"]}
    )
    candidatos = pd.DataFrame({"handle": ["@luminabeauty"]})
    novos = expandir(Cliente(yt, arquivo=None), con, planilha, candidatos)
    assert dict(zip(novos["handle"], novos["fonte"], strict=True)) == {
        "@garagem22": "colab",
        "@destaque": "destaque",
    }


URI_COM_CHAVE = (
    "https://youtube.googleapis.com/youtube/v3/playlistItems?part=x&key=AIzaSEGREDO&alt=json"
)


def test_erro_da_api_num_candidato_nao_derruba_a_avaliacao():
    yt = _canal_com_uploads(40)
    quebrado = "UCquebrado00000000000000"
    yt.canais["@quebrado"] = canal(quebrado, handle="@quebrado")
    yt.paginas["UU" + quebrado[2:]] = erro(403, "forbidden", uri=URI_COM_CHAVE)
    cand = pd.DataFrame(
        [_candidato("@quebrado", "Quebrado"), _candidato("@luminabeauty", "Lumina Beauty")]
    )
    planilha = avaliar_candidatos(Cliente(yt, arquivo=None), cand, HOJE).set_index("handle")
    assert planilha.loc["@luminabeauty", "incluido"] == "sim"
    assert planilha.loc["@quebrado", "incluido"] == "revisar"
    assert "AIzaSEGREDO" not in planilha.to_csv()


def test_cota_acabando_na_avaliacao_devolve_o_que_ja_foi_avaliado():
    yt = _canal_com_uploads(40)
    yt.canais["@outra"] = canal("UCoutra00000000000000000", handle="@outra")
    cand = pd.DataFrame(
        [_candidato("@luminabeauty", "Lumina Beauty"), _candidato("@outra", "Outra")]
    )
    # Lumina: resolver (1) + página (2) + lote (3); @outra: resolver (4); a página dela estoura
    planilha = avaliar_candidatos(Cliente(yt, limite=4, arquivo=None), cand, HOJE)
    assert planilha["handle"].tolist() == ["@luminabeauty"]


def test_balde_de_busca_esgotado_so_desliga_a_busca():
    yt = _canal_com_uploads(40)
    cand = pd.DataFrame(
        [_candidato("@errado1", "Nome 1"), _candidato("@luminabeauty", "Lumina Beauty")]
    )
    sem_busca = Cliente(yt, limite=0, arquivo=None)
    planilha = avaliar_candidatos(Cliente(yt, arquivo=None), cand, HOJE, cli_busca=sem_busca)
    por_handle = planilha.set_index("handle")
    assert por_handle.loc["@errado1", "motivo"] == "handle não encontrado"
    assert por_handle.loc["@luminabeauty", "incluido"] == "sim"


def test_main_da_curadoria_nao_vaza_a_chave(tmp_path, monkeypatch, capsys):
    import previa.curadoria as curadoria

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "candidatos.csv").write_text(
        "handle,nome,setor,tipo,fonte,observacao\n@luminabeauty,Lumina,beleza,marca,lista manual,\n"
    )
    monkeypatch.setenv("YOUTUBE_API_KEY", "AIzaSEGREDO")

    def explode(*args, **kwargs):
        raise erro(400, "badRequest", uri=URI_COM_CHAVE)

    monkeypatch.setattr(curadoria, "avaliar_candidatos", explode)
    assert curadoria.main(["avaliar"]) == 1
    saida = capsys.readouterr()
    assert "AIzaSEGREDO" not in saida.out + saida.err
    assert "key=***" in saida.err


@pytest.mark.parametrize(
    "falha", [TimeoutError("timed out"), httplib2.ServerNotFoundError("sem DNS")]
)
def test_falha_de_rede_no_meio_devolve_o_que_ja_foi_avaliado(falha):
    yt = _canal_com_uploads(40)
    outra = "UCoutra00000000000000000"
    yt.canais["@outra"] = canal(outra, handle="@outra")
    yt.paginas["UU" + outra[2:]] = falha
    cand = pd.DataFrame(
        [_candidato("@luminabeauty", "Lumina Beauty"), _candidato("@outra", "Outra")]
    )
    planilha = avaliar_candidatos(Cliente(yt, arquivo=None), cand, HOJE)
    assert planilha["handle"].tolist() == ["@luminabeauty"]


def test_candidato_sem_setor_nunca_entra_sozinho():
    expansao = {
        "handle": "@luminabeauty",
        "nome": "",
        "setor": "",
        "tipo": "",
        "fonte": "colab",
        "observacao": "citado 2x nas descrições da base",
    }
    planilha = avaliar_candidatos(
        Cliente(_canal_com_uploads(40), arquivo=None), pd.DataFrame([expansao]), HOJE
    )
    linha = planilha.iloc[0]
    assert linha["incluido"] == "revisar" and "setor" in linha["motivo"]


def test_expandir_aguenta_canal_em_destaque_sem_handle(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    com_handle, sem_handle = "UCdestaque000000000000000", "UCsemhandle00000000000000"
    secao = {
        "snippet": {"type": "multipleChannels"},
        "contentDetails": {"channels": [com_handle, sem_handle]},
    }
    yt = FakeYT(
        secoes={CANAL_ID: {"items": [secao]}},
        canais={
            com_handle: canal(com_handle, handle="@destaque"),
            sem_handle: canal(sem_handle, handle=None),
        },
    )
    planilha = pd.DataFrame(
        {"channel_id": [CANAL_ID], "handle": ["@luminabeauty"], "incluido": ["sim"]}
    )
    novos = expandir(Cliente(yt, arquivo=None), con, planilha, pd.DataFrame({"handle": []}))
    assert novos["handle"].tolist() == ["@destaque"]
