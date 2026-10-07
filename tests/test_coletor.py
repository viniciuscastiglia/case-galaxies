import json
from datetime import date, datetime, timedelta, timezone

import httplib2
import pandas as pd
import pytest
from tenacity import wait_none

from previa.coletor import (
    Cliente,
    CotaEsgotada,
    abrir_base,
    atualizar_snapshots,
    buscar_canais,
    buscar_videos,
    coletar,
    coletar_canal,
    consultar_rotulados,
    exportar_rotulados,
    ler_canais,
    listar_ids,
    parse_video,
    playlist_uploads,
    renovar_vencidos,
    salvar,
    salvar_canais,
    sem_chave,
)

from fakes import CANAL_ID, FakeYT, Req, canal, erro, pagina, video

AGORA = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


def test_retry_em_erro_transitorio_conta_a_tentativa_que_falhou():
    respostas = [erro(503, "backendError"), {"items": []}]

    def fn():
        r = respostas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    cli = Cliente(None, arquivo=None, espera=wait_none())
    assert cli.chamar(Req(fn)) == {"items": []}
    assert sum(cli.gasto.values()) == 2  # a tentativa que falhou também custou


@pytest.mark.parametrize(
    "falha",
    [
        TimeoutError("The read operation timed out"),  # a falha da coleta de 06/10
        ConnectionResetError(54, "Connection reset by peer"),
        httplib2.ServerNotFoundError("Unable to find the server at www.googleapis.com"),
    ],
)
def test_falha_de_rede_e_transitoria(falha):
    respostas = [falha, {"items": []}]

    def fn():
        r = respostas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    cli = Cliente(None, arquivo=None, espera=wait_none())
    assert cli.chamar(Req(fn)) == {"items": []}


def test_quota_exceeded_para_sem_retry():
    tentativas = []

    def estoura():
        tentativas.append(1)
        raise erro(403, "quotaExceeded")

    with pytest.raises(CotaEsgotada):
        Cliente(None, arquivo=None, espera=wait_none()).chamar(Req(estoura))
    assert len(tentativas) == 1


def test_limite_local_para_antes_de_chamar():
    with pytest.raises(CotaEsgotada):
        Cliente(None, limite=0, arquivo=None).chamar(Req(lambda: {}))


def test_dia_de_cota_e_o_do_pacifico_e_sobrevive_entre_execucoes(tmp_path):
    # 06:00 UTC de 01/10 ainda é 30/09 no Pacífico (23:00 PDT)
    cli = Cliente(
        None,
        arquivo=tmp_path / "cota.json",
        agora=lambda: datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc),
    )
    cli.chamar(Req(lambda: {}))
    assert cli.gasto == {"2026-09-30": 1}
    assert Cliente(None, arquivo=tmp_path / "cota.json").gasto == {"2026-09-30": 1}


def test_parse_short_vertical():
    r = parse_video(video("v1", "2026-08-01T22:00:05Z", "PT22S"), AGORA)
    assert r["is_short"] and r["duration_s"] == 22 and r["likes"] == 2410
    assert round(r["age_days"]) == 57


def test_curtidas_ocultas_viram_nulo():
    item = video("v2", "2026-08-01T22:00:05Z", "PT55S", likes=None)
    assert parse_video(item, AGORA)["likes"] is None


def test_horizontal_ou_longo_nao_e_short():
    horizontal = video("v3", "2026-08-01T22:00:05Z", "PT50S", w="1920", h="1080")
    assert not parse_video(horizontal, AGORA)["is_short"]
    assert not parse_video(video("v4", "2026-08-01T22:00:05Z", "PT15M51S"), AGORA)["is_short"]
    # antes de 15/10/2024 o limite era 60 s
    assert not parse_video(video("v5", "2024-05-01T10:00:00Z", "PT1M30S"), AGORA)["is_short"]


def test_live_agendada_nao_e_short():
    assert not parse_video(video("v6", "2026-08-01T22:00:05Z", "P0D"), AGORA)["is_short"]


def test_sem_player_proporcao_desconhecida_continua_short_provavel():
    item = video("v7", "2026-08-01T22:00:05Z", "PT40S")
    del item["player"]
    r = parse_video(item, AGORA)
    assert r["vertical"] is None and r["is_short"]


def test_playlist_uploads():
    assert playlist_uploads("UCabc123LuminaBeauty0000") == "UUabc123LuminaBeauty0000"
    with pytest.raises(ValueError):
        playlist_uploads("HCtopico000000000000000")


def test_listar_ids_pagina_e_para_quando_a_pagina_ja_e_conhecida():
    ids = [f"v{i:02d}" for i in range(60)]
    yt = FakeYT(paginas={"UUx": {"p0": pagina(ids[:50], "p1"), "p1": pagina(ids[50:])}})
    cli = Cliente(yt, arquivo=None)
    assert listar_ids(cli, "UUx") == ids
    assert yt.n("playlistItems") == 2
    yt.chamadas.clear()
    assert listar_ids(cli, "UUx", frozenset(ids[:50])) == []
    assert yt.n("playlistItems") == 1


def test_playlist_vazia_termina_com_uma_chamada():
    yt = FakeYT(paginas={"UUx": {"p0": {"kind": "youtube#playlistItemListResponse", "items": []}}})
    assert listar_ids(Cliente(yt, arquivo=None), "UUx") == []
    assert yt.n("playlistItems") == 1


def test_buscar_videos_em_lotes_de_50_e_ignora_removidos():
    itens = {f"v{i:03d}": video(f"v{i:03d}", "2026-08-01T22:00:05Z", "PT22S") for i in range(120)}
    yt = FakeYT(itens=itens)
    assert len(buscar_videos(Cliente(yt, arquivo=None), [*itens, "removido"])) == 120
    assert yt.n("videos") == 3  # ceil(121 / 50)


def _coletar_em(con, ids_e_datas, dia_coleta):
    coletado = datetime.fromisoformat(dia_coleta).replace(tzinfo=timezone.utc)
    salvar(con, [parse_video(video(v, pub, "PT22S"), coletado) for v, pub in ids_e_datas])


def test_salvar_duas_vezes_no_mesmo_dia_nao_duplica(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    lote = [(f"v{i}", "2026-08-01T22:00:05Z") for i in range(3)]
    _coletar_em(con, lote, "2026-09-27T12:00:00")
    _coletar_em(con, lote, "2026-09-27T18:00:00")
    assert con.execute("SELECT count(*) FROM videos").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 3


def test_dias_diferentes_viram_serie(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    lote = [(f"v{i}", "2026-08-01T22:00:05Z") for i in range(3)]
    _coletar_em(con, lote, "2026-09-20T12:00:00")
    _coletar_em(con, lote, "2026-09-27T12:00:00")
    assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 6


def test_curtidas_ocultas_ficam_nulas_na_base(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    salvar(
        con,
        [
            parse_video(video("visivel", "2026-08-01T22:00:05Z", "PT22S"), AGORA),
            parse_video(video("oculto", "2026-08-01T22:00:05Z", "PT22S", likes=None), AGORA),
        ],
    )
    got = dict(con.execute("SELECT video_id, likes FROM snapshots").fetchall())
    assert got == {"visivel": 2410, "oculto": None}


def test_video_sem_tags_e_sem_player_e_salvo(tmp_path):
    item = video("semtag", "2026-08-01T22:00:05Z", "PT22S")
    del item["snippet"]["tags"]
    del item["player"]
    con = abrir_base(tmp_path / "t.duckdb")
    salvar(con, [parse_video(item, AGORA)])
    assert con.execute("SELECT tags, vertical FROM videos").fetchone() == ([], None)


def test_video_que_nao_e_short_nao_ganha_snapshot(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    salvar(con, [parse_video(video("longo", "2026-08-01T22:00:05Z", "PT15M"), AGORA)])
    assert con.execute("SELECT count(*) FROM videos").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 0


def test_rotulo_e_o_primeiro_snapshot_com_30_dias_e_respeita_o_teto(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    lote = [(f"v{d}", f"2026-08-0{d}T12:00:00Z") for d in range(1, 6)]
    lote.append(("jovem", "2026-09-01T12:00:00Z"))  # nunca chega a 30 dias nas coletas abaixo
    for dia in ["2026-08-20T12:00:00", "2026-09-10T12:00:00", "2026-09-25T12:00:00"]:
        _coletar_em(con, lote, dia)
    df = consultar_rotulados(con, teto_por_canal=3)
    assert set(df["video_id"]) == {"v3", "v4", "v5"}  # os 3 rotulados mais recentes
    assert (df["age_days"] >= 30).all()
    assert (df["collected_at"].dt.date == date(2026, 9, 10)).all()
    assert exportar_rotulados(con, tmp_path / "r.parquet", teto_por_canal=3) == 3
    assert len(pd.read_parquet(tmp_path / "r.parquet")) == 3


def test_foto_dos_canais_uma_por_dia_e_inscritos_ocultos_nulos(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    outro = "UCdef456Garagem2200000000"
    itens = [canal(), canal(outro, handle="@garagem22", oculto=True)]
    salvar_canais(con, itens, AGORA)
    salvar_canais(con, itens, AGORA)
    linhas = dict(con.execute("SELECT channel_id, subscribers FROM canais_snapshots").fetchall())
    assert linhas == {CANAL_ID: 180000, outro: None}


def test_buscar_canais_50_por_chamada():
    ids = [f"UC{i:022d}" for i in range(120)]
    yt = FakeYT(canais={i: canal(i) for i in ids})
    assert len(buscar_canais(Cliente(yt, arquivo=None), ids)) == 120
    assert yt.n("channels") == 3


UPLOADS = "UU" + CANAL_ID[2:]


def _canal_lumina(n=60):
    ids = [f"v{i:02d}" for i in range(n)]
    itens = {v: video(v, "2026-08-01T22:00:05Z", "PT22S") for v in ids}
    paginas = {"p0": pagina(ids[:50], "p1"), "p1": pagina(ids[50:])}
    return FakeYT(paginas={UPLOADS: paginas}, itens=itens, canais={CANAL_ID: canal()})


def test_coleta_idempotente_e_incremental(tmp_path):
    yt = _canal_lumina()
    cli, con = Cliente(yt, arquivo=None), abrir_base(tmp_path / "t.duckdb")
    assert coletar_canal(cli, con, CANAL_ID) == 60
    assert len(yt.chamadas) == 2 + 2  # 2 páginas + 2 lotes de videos.list
    assert coletar_canal(cli, con, CANAL_ID) == 0  # nada novo
    assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 60


def test_canal_sem_playlist_de_uploads_devolve_zero(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    assert coletar_canal(Cliente(FakeYT(), arquivo=None), con, CANAL_ID) == 0


def test_falha_num_canal_nao_derruba_os_outros(tmp_path):
    cli, con = Cliente(_canal_lumina(), arquivo=None), abrir_base(tmp_path / "t.duckdb")
    resumo = coletar(cli, con, ["HCtopico000000000000000", CANAL_ID])
    assert resumo["canais_ok"] == 1 and resumo["videos_novos"] == 60
    assert "HCtopico000000000000000" in resumo["falhas"]


def test_timeout_persistente_num_canal_nao_derruba_os_outros(tmp_path):
    lento = "UCcanalLento000000000000"
    yt = _canal_lumina()
    yt.paginas["UU" + lento[2:]] = TimeoutError("The read operation timed out")
    executou, original = [], yt._playlistItems  # o retry repete o execute(), não o list()
    yt._playlistItems = lambda kw: executou.append(kw["playlistId"]) or original(kw)
    cli = Cliente(yt, arquivo=None, espera=wait_none())
    resumo = coletar(cli, abrir_base(tmp_path / "t.duckdb"), [lento, CANAL_ID])
    assert resumo["canais_ok"] == 1 and resumo["videos_novos"] == 60
    assert "TimeoutError" in resumo["falhas"][lento]
    assert executou.count("UU" + lento[2:]) == 5  # refez antes de desistir do canal


def test_cota_no_meio_para_sem_corromper_e_retoma_sem_duplicar(tmp_path):
    yt, con = _canal_lumina(), abrir_base(tmp_path / "t.duckdb")
    # 1 (canais) + 2 (páginas) cabem; o 1º lote de videos.list estoura o limite
    resumo = coletar(Cliente(yt, limite=3, arquivo=None), con, [CANAL_ID])
    assert resumo["parou_por_cota"]
    assert con.execute("SELECT count(*) FROM videos").fetchone()[0] == 0
    resumo = coletar(Cliente(yt, arquivo=None), con, [CANAL_ID])  # "no dia seguinte"
    assert resumo["parou_por_cota"] is None and resumo["videos_novos"] == 60
    assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 60


def test_renovacao_dos_30_dias_renova_e_apaga_os_que_sumiram(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    antigo = datetime.now(timezone.utc) - timedelta(days=40)
    pub = "2025-06-01T12:00:00Z"
    salvar(
        con,
        [
            parse_video(video("fica", pub, "PT22S"), antigo),
            parse_video(video("sumiu", pub, "PT22S"), antigo),
        ],
    )
    yt = FakeYT(itens={"fica": video("fica", pub, "PT22S")})
    assert renovar_vencidos(Cliente(yt, arquivo=None), con) == {"renovados": 1, "apagados": 1}
    n_fica = "SELECT count(*) FROM snapshots WHERE video_id = 'fica'"
    assert con.execute(n_fica).fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM videos WHERE video_id = 'sumiu'").fetchone()[0] == 0


def test_atualizar_snapshots_so_pega_jovens_sem_foto_de_hoje(tmp_path):
    con = abrir_base(tmp_path / "t.duckdb")
    agora = datetime.now(timezone.utc)
    ontem = agora - timedelta(days=1)
    jovem = (agora - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    velho = (agora - timedelta(days=200)).strftime("%Y-%m-%dT%H:%M:%SZ")
    salvar(
        con,
        [
            parse_video(video("jovem", jovem, "PT22S"), ontem),
            parse_video(video("velho", velho, "PT22S"), ontem),
        ],
    )
    yt = FakeYT(itens={"jovem": video("jovem", jovem, "PT22S")})
    assert atualizar_snapshots(Cliente(yt, arquivo=None), con) == 1
    assert atualizar_snapshots(Cliente(yt, arquivo=None), con) == 0  # já tem foto de hoje


def test_ler_canais_so_os_incluidos(tmp_path):
    caminho = tmp_path / "canais.csv"
    caminho.write_text(
        "channel_id,handle,incluido\nUCa,@a,sim\nUCb,@b,não\nUCc,@c,revisar\nUCd,@d, Sim \n"
    )
    assert ler_canais(caminho) == ["UCa", "UCd"]


URI_COM_CHAVE = (
    "https://youtube.googleapis.com/youtube/v3/playlistItems?part=x&key=AIzaSEGREDO&alt=json"
)


def test_sem_chave_apaga_a_chave_da_url():
    assert sem_chave(URI_COM_CHAVE).endswith("part=x&key=***&alt=json")


def test_falha_de_canal_nao_vaza_a_chave_da_api(tmp_path):
    yt = _canal_lumina()
    yt.paginas[UPLOADS] = erro(403, "forbidden", uri=URI_COM_CHAVE)
    resumo = coletar(Cliente(yt, arquivo=None), abrir_base(tmp_path / "t.duckdb"), [CANAL_ID])
    assert "AIzaSEGREDO" not in json.dumps(resumo)
    assert "key=***" in resumo["falhas"][CANAL_ID]


def test_main_nao_vaza_a_chave_em_erro_inesperado(tmp_path, monkeypatch, capsys):
    import previa.coletor as coletor

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "canais.csv").write_text(f"channel_id,incluido\n{CANAL_ID},sim\n")
    monkeypatch.setenv("YOUTUBE_API_KEY", "AIzaSEGREDO")

    def explode(*args, **kwargs):
        raise erro(400, "badRequest", uri=URI_COM_CHAVE)

    monkeypatch.setattr(coletor, "coletar", explode)
    assert coletor.main() == 1
    saida = capsys.readouterr()
    assert "AIzaSEGREDO" not in saida.out + saida.err
    assert "key=***" in saida.err
