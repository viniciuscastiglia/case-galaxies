"""Coletor de Shorts pela YouTube Data API (aula 1.5 da trilha da liga).

Caminho barato: canal -> playlist de uploads (UU...) -> playlistItems.list (50 por página)
-> videos.list (50 IDs por chamada), sempre a 1 unidade por chamada. Idempotente (upsert),
incremental (para quando a página já é conhecida), resiliente (retry só em erro transitório,
para em quotaExceeded) e testável sem chave (tests/fakes.py).

Acréscimos em relação à aula: foto diária dos canais (Monte sua base, passo 5), renovação
dos dados com mais de 25 dias (aula 1.8, regra dos 30 dias) e teto de Shorts por canal na
exportação (aula 1.4).
"""

import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import isodate
import pandas as pd
from googleapiclient.errors import HttpError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

PT = ZoneInfo("America/Los_Angeles")  # a cota zera à meia-noite do horário do Pacífico
MUDOU_PARA_3_MIN = datetime(2024, 10, 15, tzinfo=timezone.utc)  # antes: Shorts até 60 s
PARTES = "snippet,contentDetails,statistics,player"
PARTES_CANAL = "snippet,contentDetails,statistics"
DADOS = Path("data")


def sem_chave(texto: str) -> str:
    """Apaga a chave da API de um texto. As mensagens de HttpError trazem a URL da requisição,
    e a chave vai nela como `key=...`: nada que vai para log ou tela pode sair sem passar aqui."""
    return re.sub(r"(key=)[^&\s\"'<>]+", r"\1***", texto)


# --- etapa 1: cota e retries ------------------------------------------------------------


class CotaEsgotada(RuntimeError):
    """Pare a coleta e retome no próximo dia de cota."""


def motivo(e: HttpError) -> str:
    """Extrai o 'reason' do erro da YouTube API (ex.: quotaExceeded)."""
    try:
        return json.loads(e.content)["error"]["errors"][0]["reason"]
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def transitorio(e: BaseException) -> bool:
    """Vale tentar de novo? Erros 5xx, 429 e rate limit, sim; cota esgotada, não."""
    if not isinstance(e, HttpError):
        return False
    return e.status_code >= 500 or e.status_code == 429 or motivo(e) == "rateLimitExceeded"


class Cliente:
    """Embrulha o cliente oficial: conta unidades por dia (PT) e refaz erros transitórios."""

    def __init__(self, yt, limite=9_000, arquivo=DADOS / "cota.json", espera=None, agora=None):
        self.yt, self.limite = yt, limite
        self.arquivo = Path(arquivo) if arquivo else None
        existe = self.arquivo is not None and self.arquivo.exists()
        self.gasto = json.loads(self.arquivo.read_text()) if existe else {}
        self.espera = espera or wait_exponential_jitter(initial=1, max=60)
        self.agora = agora or (lambda: datetime.now(timezone.utc))

    def _debitar(self, custo: int) -> None:
        dia = self.agora().astimezone(PT).date().isoformat()
        if self.gasto.get(dia, 0) + custo > self.limite:
            raise CotaEsgotada(f"limite local de {self.limite} unidades atingido em {dia} (PT)")
        self.gasto[dia] = self.gasto.get(dia, 0) + custo
        if self.arquivo:
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
            self.arquivo.write_text(json.dumps(self.gasto))

    def chamar(self, req, custo: int = 1) -> dict:
        @retry(
            retry=retry_if_exception(transitorio),
            wait=self.espera,
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def uma_vez():
            self._debitar(custo)  # toda requisição custa, inclusive as que falham
            return req.execute()

        try:
            return uma_vez()
        except HttpError as e:
            if motivo(e) == "quotaExceeded":
                raise CotaEsgotada("a API respondeu quotaExceeded") from e
            raise


# --- etapa 2: da playlist de uploads aos detalhes ----------------------------------------


def playlist_uploads(channel_id: str) -> str:
    """UC... (canal) -> UU... (playlist de uploads). O oficial é ler
    contentDetails.relatedPlaylists.uploads; este atalho dá o mesmo valor."""
    if not channel_id.startswith("UC"):
        raise ValueError(f"channel_id inesperado: {channel_id}")
    return "UU" + channel_id[2:]


def listar_ids(cli, playlist_id: str, conhecidos=frozenset(), max_paginas=None) -> list[str]:
    """Pagina a playlist (50 itens por página, 1 unidade por página).

    Coleta incremental: para quando uma página inteira já é conhecida.
    A ordem (mais novo primeiro) é o comportamento observado, não um contrato:
    faça uma varredura completa de vez em quando (conhecidos=frozenset()).
    """
    ids, token, paginas = [], None, 0
    while True:
        req = cli.yt.playlistItems().list(
            part="contentDetails", playlistId=playlist_id, maxResults=50, pageToken=token
        )
        resp = cli.chamar(req)
        paginas += 1
        pagina = [it["contentDetails"]["videoId"] for it in resp.get("items", [])]
        ids += [v for v in pagina if v not in conhecidos]
        token = resp.get("nextPageToken")
        tudo_conhecido = bool(pagina) and all(v in conhecidos for v in pagina)
        if not token or tudo_conhecido or (max_paginas and paginas >= max_paginas):
            return ids


def buscar_videos(cli, ids: list[str]) -> list[dict]:
    """videos.list em lotes de 50 IDs: 1 unidade por lote.
    `player` + maxHeight faz a API devolver embedWidth/embedHeight (proporção)."""
    itens = []
    for i in range(0, len(ids), 50):
        req = cli.yt.videos().list(part=PARTES, id=",".join(ids[i : i + 50]), maxHeight=1080)
        itens += cli.chamar(req).get("items", [])  # removidos/privados simplesmente não voltam
    return itens


# --- etapa 3: um item da API vira uma linha -----------------------------------------------


def _int(valor):
    return None if valor is None else int(valor)  # a API manda números como string


def parse_video(item: dict, coletado_em: datetime) -> dict:
    sn = item["snippet"]
    cd = item.get("contentDetails", {})
    st = item.get("statistics", {})
    pl = item.get("player", {})
    publicado = datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00"))
    duracao = isodate.parse_duration(cd.get("duration", "P0D")).total_seconds()

    w, h = int(pl.get("embedWidth") or 0), int(pl.get("embedHeight") or 0)
    vertical = (h >= w) if (w and h) else None  # quadrado conta; None = desconhecido
    # +1 s de folga: a doc avisa que a duração pode diferir em ~1 s da real
    limite = (180 if publicado >= MUDOU_PARA_3_MIN else 60) + 1

    return {
        "video_id": item["id"],
        "channel_id": sn["channelId"],
        "published_at": publicado,
        "title": sn.get("title", ""),
        "description": sn.get("description", ""),
        "tags": sn.get("tags", []),
        "category_id": sn.get("categoryId"),
        "audio_lang": sn.get("defaultAudioLanguage"),
        "duration_s": duracao,
        "vertical": vertical,
        "is_short": 0 < duracao <= limite and vertical is not False,
        "views": _int(st.get("viewCount")),
        "likes": _int(st.get("likeCount")),  # ausente = curtidas ocultas -> None, nunca 0
        "comments": _int(st.get("commentCount")),
        "age_days": (coletado_em - publicado).total_seconds() / 86_400,
        "collected_at": coletado_em,
    }


# --- etapa 4: persistência idempotente (DuckDB) e exportação (Parquet) ------------------

DDL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id VARCHAR PRIMARY KEY, channel_id VARCHAR, published_at TIMESTAMPTZ,
    title VARCHAR, description VARCHAR, tags VARCHAR[], category_id VARCHAR,
    audio_lang VARCHAR, duration_s DOUBLE, vertical BOOLEAN, is_short BOOLEAN,
    first_seen_at TIMESTAMPTZ, updated_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS snapshots (
    video_id VARCHAR, collected_date DATE, collected_at TIMESTAMPTZ,
    views BIGINT, likes BIGINT, comments BIGINT, age_days DOUBLE,
    PRIMARY KEY (video_id, collected_date));
CREATE TABLE IF NOT EXISTS canais_snapshots (
    channel_id VARCHAR, collected_date DATE, collected_at TIMESTAMPTZ,
    subscribers BIGINT, hidden_subscribers BOOLEAN, video_count BIGINT, view_count BIGINT,
    PRIMARY KEY (channel_id, collected_date));
"""

SQL_ROTULADOS = """
WITH rotulos AS (
    SELECT v.*, s.likes, s.views, s.comments, s.age_days, s.collected_at
    FROM videos v JOIN snapshots s USING (video_id)
    WHERE v.is_short AND s.age_days >= $idade_min
    QUALIFY row_number() OVER (PARTITION BY video_id ORDER BY s.age_days) = 1
)
SELECT * FROM rotulos
QUALIFY row_number() OVER (PARTITION BY channel_id ORDER BY published_at DESC) <= $teto
ORDER BY channel_id, published_at
"""


def abrir_base(caminho=DADOS / "shorts.duckdb") -> duckdb.DuckDBPyConnection:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(caminho))
    con.execute("SET TimeZone = 'UTC'")  # datas de snapshot sempre em UTC
    con.execute(DDL)
    return con


def salvar(con, linhas: list[dict]) -> None:
    """Upsert por video_id; um snapshot de estatísticas por vídeo por dia.
    Rodar duas vezes no mesmo dia não duplica nada."""
    if not linhas:
        return
    # tipos anuláveis explícitos: curtida oculta vira NULL, não NaN nem 0
    novos = pd.DataFrame(linhas).astype(
        {"views": "Int64", "likes": "Int64", "comments": "Int64", "vertical": "boolean"}
    )
    con.register("novos", novos)
    con.execute("""
        INSERT INTO videos
        SELECT video_id, channel_id, published_at, title, description, tags, category_id,
               audio_lang, duration_s, vertical, is_short, collected_at, collected_at
        FROM novos
        ON CONFLICT (video_id) DO UPDATE SET
            title = excluded.title, description = excluded.description,
            tags = excluded.tags, updated_at = excluded.updated_at""")
    con.execute("""
        INSERT INTO snapshots
        SELECT video_id, CAST(collected_at AS DATE), collected_at, views, likes, comments,
               age_days
        FROM novos WHERE is_short
        ON CONFLICT (video_id, collected_date) DO UPDATE SET
            collected_at = excluded.collected_at, views = excluded.views,
            likes = excluded.likes, comments = excluded.comments, age_days = excluded.age_days""")
    con.unregister("novos")


def buscar_canais(cli, channel_ids: list[str]) -> list[dict]:
    """channels.list por ID: até 50 canais por chamada (1 unidade)."""
    itens = []
    for i in range(0, len(channel_ids), 50):
        req = cli.yt.channels().list(part=PARTES_CANAL, id=",".join(channel_ids[i : i + 50]))
        itens += cli.chamar(req).get("items", [])
    return itens


def salvar_canais(con, itens: list[dict], coletado_em: datetime) -> None:
    """Foto do dia dos canais: uma linha por canal por dia. É a channel_stats_at_collection
    do esquema mínimo (Monte sua base, passo 5; aula 1.7: inscritos são foto da coleta)."""
    if not itens:
        return
    linhas = []
    for c in itens:
        st = c.get("statistics", {})
        oculto = bool(st.get("hiddenSubscriberCount"))
        linhas.append(
            {
                "channel_id": c["id"],
                "collected_at": coletado_em,
                "subscribers": None if oculto else _int(st.get("subscriberCount")),
                "hidden_subscribers": oculto,
                "video_count": _int(st.get("videoCount")),
                "view_count": _int(st.get("viewCount")),
            }
        )
    novos = pd.DataFrame(linhas).astype(
        {"subscribers": "Int64", "video_count": "Int64", "view_count": "Int64"}
    )
    con.register("canais_novos", novos)
    con.execute("""
        INSERT INTO canais_snapshots
        SELECT channel_id, CAST(collected_at AS DATE), collected_at, subscribers,
               hidden_subscribers, video_count, view_count
        FROM canais_novos
        ON CONFLICT (channel_id, collected_date) DO UPDATE SET
            collected_at = excluded.collected_at, subscribers = excluded.subscribers,
            hidden_subscribers = excluded.hidden_subscribers,
            video_count = excluded.video_count, view_count = excluded.view_count""")
    con.unregister("canais_novos")


def consultar_rotulados(con, idade_min: int = 30, teto_por_canal: int = 400) -> pd.DataFrame:
    """Um rótulo por Short: o primeiro snapshot com pelo menos `idade_min` dias.
    Teto: os `teto_por_canal` Shorts rotulados mais recentes de cada canal (aula 1.4)."""
    return con.execute(SQL_ROTULADOS, {"idade_min": idade_min, "teto": teto_por_canal}).df()


def exportar_rotulados(
    con, destino=DADOS / "shorts_rotulados.parquet", idade_min: int = 30, teto_por_canal: int = 400
) -> int:
    rotulados = consultar_rotulados(con, idade_min, teto_por_canal)
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    rotulados.to_parquet(destino, index=False)
    return len(rotulados)


# --- etapa 5: orquestração, renovação dos 30 dias e execução diária -----------------------


def coletar_canal(cli, con, channel_id: str) -> int:
    """Descobre vídeos novos do canal e grava metadados + snapshot de hoje."""
    conhecidos = {
        r[0]
        for r in con.execute(
            "SELECT video_id FROM videos WHERE channel_id = ?", [channel_id]
        ).fetchall()
    }
    try:
        novos = listar_ids(cli, playlist_uploads(channel_id), frozenset(conhecidos))
    except HttpError as e:
        if e.status_code == 404:  # playlist de uploads inexistente (canal sem vídeos)
            return 0
        raise
    agora = datetime.now(timezone.utc)
    salvar(con, [parse_video(it, agora) for it in buscar_videos(cli, novos)])
    return len(novos)


def atualizar_snapshots(cli, con, idade_max: int = 60) -> int:
    """Re-coleta estatísticas dos Shorts ainda 'jovens' (1 unidade a cada 50)."""
    ids = [
        r[0]
        for r in con.execute(
            """
        SELECT video_id FROM videos
        WHERE is_short AND published_at >= now() - to_days(?)
          AND video_id NOT IN (SELECT video_id FROM snapshots WHERE collected_date = current_date)
    """,
            [idade_max],
        ).fetchall()
    ]
    agora = datetime.now(timezone.utc)
    salvar(con, [parse_video(it, agora) for it in buscar_videos(cli, ids)])
    return len(ids)


def apagar(con, video_ids: list[str]) -> None:
    """Para vídeos que sumiram da API (removidos/privados) ou que não serão renovados."""
    if video_ids:
        con.execute("DELETE FROM snapshots WHERE list_contains(?, video_id)", [video_ids])
        con.execute("DELETE FROM videos WHERE list_contains(?, video_id)", [video_ids])


def renovar_vencidos(cli, con, prazo_dias: int = 25) -> dict:
    """Regra dos 30 dias (Developer Policies III.E.4.d, aula 1.8): renova as estatísticas de
    todo Short cujo snapshot mais recente passou de `prazo_dias` (5 dias de margem) e apaga
    da base os que sumiram da API."""
    vencidos = [
        r[0]
        for r in con.execute(
            """
        SELECT video_id FROM snapshots GROUP BY video_id
        HAVING max(collected_at) < now() - to_days(?)""",
            [prazo_dias],
        ).fetchall()
    ]
    if not vencidos:
        return {"renovados": 0, "apagados": 0}
    agora = datetime.now(timezone.utc)
    itens = buscar_videos(cli, vencidos)
    salvar(con, [parse_video(it, agora) for it in itens])
    voltaram = {it["id"] for it in itens}
    sumiram = [v for v in vencidos if v not in voltaram]
    apagar(con, sumiram)
    return {"renovados": len(voltaram), "apagados": len(sumiram)}


def ler_canais(caminho=DADOS / "canais.csv") -> list[str]:
    """IDs dos canais com incluido == 'sim' na planilha de curadoria (aula 1.4)."""
    planilha = pd.read_csv(caminho, dtype=str)
    incluido = planilha["incluido"].fillna("").str.strip().str.lower() == "sim"
    return planilha.loc[incluido, "channel_id"].dropna().tolist()


def coletar(cli, con, channel_ids: list[str]) -> dict:
    """Uma execução diária. Para em CotaEsgotada (idempotente: amanhã continua); a falha de
    um canal não derruba os outros, mas fica registrada no resumo."""
    resumo = {
        "canais_ok": 0,
        "falhas": {},
        "videos_novos": 0,
        "snapshots_jovens": 0,
        "renovacao": {},
        "parou_por_cota": None,
    }
    try:
        salvar_canais(con, buscar_canais(cli, channel_ids), datetime.now(timezone.utc))
        for cid in channel_ids:
            try:
                resumo["videos_novos"] += coletar_canal(cli, con, cid)
                resumo["canais_ok"] += 1
            except (HttpError, ValueError) as e:
                resumo["falhas"][cid] = sem_chave(repr(e))
        resumo["snapshots_jovens"] = atualizar_snapshots(cli, con)
        resumo["renovacao"] = renovar_vencidos(cli, con)
    except CotaEsgotada as e:
        resumo["parou_por_cota"] = str(e)
    return resumo


def main() -> int:
    from googleapiclient.discovery import build

    chave = os.environ.get("YOUTUBE_API_KEY")
    if not chave:
        print(
            "YOUTUBE_API_KEY não definida: rode pelo scripts/coletar.sh, que carrega o .env.",
            file=sys.stderr,
        )
        return 2
    yt = build("youtube", "v3", developerKey=chave, cache_discovery=False)
    cli, con = Cliente(yt), abrir_base()
    inicio = datetime.now(timezone.utc)
    try:
        resumo = coletar(cli, con, ler_canais())
        resumo["rotulados"] = exportar_rotulados(con)
    except Exception:  # erro inesperado: registra sem a chave e sai com falha
        print(inicio.isoformat(), sem_chave(traceback.format_exc()), file=sys.stderr)
        return 1
    resumo["unidades_hoje_pt"] = cli.gasto.get(datetime.now(PT).date().isoformat(), 0)
    print(inicio.isoformat(), json.dumps(resumo, ensure_ascii=False))
    if resumo["parou_por_cota"]:
        return 3
    return 1 if resumo["falhas"] else 0


if __name__ == "__main__":
    sys.exit(main())
