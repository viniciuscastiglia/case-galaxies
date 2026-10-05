# Base de Shorts — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** deixar a base de Shorts de beleza e automotivo coletando todo dia, com curadoria de canais auditável, seguindo
o módulo 1 da trilha da liga.

**Architecture:** um pacote `previa` com dois módulos que espelham as aulas: `coletor.py` (aula 1.5, com os acréscimos
das aulas 1.4, 1.7 e 1.8) e `curadoria.py` (aula 1.4). A base mora num arquivo DuckDB com três tabelas (`videos`,
`snapshots` e `canais_snapshots`); a exportação gera Parquet. Um agendamento do macOS (launchd) roda a coleta às 05h15.

**Tech Stack:** Python 3.12 (uv), google-api-python-client, tenacity, isodate, DuckDB, pandas, pyarrow, Pydantic,
pandera, pytest e ruff.

**Spec:** o material da liga (clone em `autoestudos/`): `docs/projetos/galaxies/monte-sua-base.mdx` e as aulas
`docs/aulas/ml-preditivo-e-agentes/1-formulacao-e-dados/` 3, 4, 5, 7 e 8. Regras de trabalho em `CLAUDE.md`.

## Global Constraints

- Python 3.12 via uv. Versões fixadas (Stack 2026 da liga): `google-api-python-client==2.200.0`, `isodate==0.7.2`,
  `tenacity==9.1.4`, `duckdb==1.5.5`, `pandas==3.0.6`, `numpy==2.5.3`, `pyarrow==25.0.1`, `pydantic==2.13.5`,
  `pandera==0.33.1`; grupo dev: `pytest==9.1.1`, `ruff==0.16.9`.
- Vídeos vêm da playlist de uploads (`UU…`); `search.list` nunca lista vídeos e só serve de reserva para achar um canal
  cujo handle não resolveu (no máximo 90 chamadas por dia, no balde próprio de 100).
- Limite local de 9.000 unidades por dia de cota, contado no dia do Pacífico (`America/Los_Angeles`).
- Short = duração ≤ 180 s (+1 s de folga) para vídeos a partir de 15/10/2024 (≤ 60 s antes) e orientação não
  horizontal. `UUSH` só como conferência numa amostra.
- `likeCount` ausente vira `None`, nunca 0. Rótulo = primeiro snapshot com `age_days` ≥ 30. Teto de 400 Shorts
  rotulados por canal (os mais recentes).
- `collected_at` em toda linha; snapshots únicos por `(video_id, collected_date)` e `(channel_id, collected_date)`.
- Dados só em `data/`, fora do git, exceto `data/candidatos.csv` e `data/canais.csv`. Chave só no `.env`.
- Código no estilo das aulas (nomes em português onde a aula usa português). Commits só quando o Vinicius pedir.

## Review Focus

1. **Handle escrito de outro jeito** (URL, sem `@`, maiúsculas, espaços): é normalizado antes de resolver
   (teste na Task 5).
2. **A cota acaba no meio da execução:** a coleta para sem corromper nada e, no dia seguinte, retoma sem duplicar
   (teste na Task 4).
3. **Vídeo removido ou privado entre coletas:** sai da base na renovação dos 30 dias (teste na Task 4).
4. **`channel_id` fora do padrão** (ex.: canal de tópico `HC…`) ou canal sem uploads: a falha fica isolada no resumo e os
   outros canais seguem (teste na Task 4).
5. **Duas execuções no mesmo dia** (o Mac acorda e o launchd roda, ou execução manual): nenhum snapshot duplica, e o dia
   da cota é o do Pacífico (testes nas Tasks 1 e 3).

---

### Task 1: Ambiente e cliente com cota e retries (aula 1.5, etapa 1)

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.python-version` (via `uv init`), `src/previa/__init__.py`
- Create: `src/previa/coletor.py`
- Create: `tests/fakes.py`, `tests/test_coletor.py`

**Interfaces:**
- Produces: `CotaEsgotada(RuntimeError)`; `motivo(e: HttpError) -> str`; `transitorio(e: BaseException) -> bool`;
  `Cliente(yt, limite=9000, arquivo=DADOS/"cota.json", espera=None, agora=None)` com `.chamar(req, custo=1) -> dict`
  e `.gasto: dict[str, int]` (dia PT → unidades); constantes `PT`, `DADOS = Path("data")`.
- Produces (testes): `tests/fakes.py` com `CANAL_ID`, `video()`, `pagina()`, `canal()`, `erro()`, `Req`, `FakeYT`.

- [ ] **Step 1: Instalar o uv e criar o projeto**

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh   # instala em ~/.local/bin
cd ~/case-galaxies
uv init --package --name previa --python 3.12 --vcs none --no-readme --author-from none
uv add google-api-python-client==2.200.0 isodate==0.7.2 tenacity==9.1.4 duckdb==1.5.5 \
       pandas==3.0.6 numpy==2.5.3 pyarrow==25.0.1 pydantic==2.13.5 pandera==0.33.1
uv add --dev pytest==9.1.1 ruff==0.16.9
```

Depois, no `pyproject.toml`: trocar `requires-python` por `">=3.12,<3.13"`, apagar a seção `[project.scripts]` que o
`uv init` cria e acrescentar ao fim:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.ruff.lint.isort]
known-first-party = ["previa"]
known-local-folder = ["fakes"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
```

E `src/previa/__init__.py` passa a ter só:

```python
"""Prévia: previsão de curtidas de YouTube Shorts antes da publicação (case Academy x Galaxies)."""
```

- [ ] **Step 2: Escrever as respostas falsas no formato real da API**

`tests/fakes.py`:

```python
"""Respostas falsas no formato real da YouTube Data API (aulas 1.4 e 1.5).

Os testes rodam sem chave: FakeYT imita yt.<recurso>().list(...).execute().
"""

import httplib2
from googleapiclient.errors import HttpError

CANAL_ID = "UCabc123LuminaBeauty0000"


def video(vid, publicado, dur, likes="2410", w="608", h="1080", canal_id=CANAL_ID):
    item = {
        "kind": "youtube#video",
        "id": vid,
        "snippet": {
            "publishedAt": publicado,
            "channelId": canal_id,
            "title": "1 produto, 20 segundos",
            "description": "Base nova com a @lumina.beauty #shorts",
            "tags": ["maquiagem"],
            "categoryId": "26",
            "defaultAudioLanguage": "pt-BR",
        },
        "contentDetails": {"duration": dur, "definition": "hd", "caption": "false"},
        "statistics": {
            "viewCount": "81234",
            "likeCount": likes,
            "favoriteCount": "0",
            "commentCount": "57",
        },
        "player": {"embedHtml": "<iframe ...></iframe>", "embedWidth": w, "embedHeight": h},
    }
    if likes is None:
        del item["statistics"]["likeCount"]  # canal escondeu as curtidas
    return item


def pagina(ids, prox=None, publicado="2026-08-01T22:00:05Z"):
    resp = {
        "kind": "youtube#playlistItemListResponse",
        "items": [
            {
                "kind": "youtube#playlistItem",
                "contentDetails": {"videoId": v, "videoPublishedAt": publicado},
            }
            for v in ids
        ],
    }
    if prox:
        resp["nextPageToken"] = prox
    return resp


def canal(
    cid=CANAL_ID,
    handle="@luminabeauty",
    pais="BR",
    inscritos="180000",
    oculto=False,
    n_videos="912",
):
    sn = {"title": "Lumina Beauty", "customUrl": handle, "publishedAt": "2019-03-02T14:00:00Z"}
    if pais is not None:
        sn["country"] = pais
    st = {"viewCount": "48200000", "hiddenSubscriberCount": oculto, "videoCount": n_videos}
    if not oculto:
        st["subscriberCount"] = inscritos
    return {
        "kind": "youtube#channel",
        "id": cid,
        "snippet": sn,
        "contentDetails": {"relatedPlaylists": {"likes": "", "uploads": "UU" + cid[2:]}},
        "statistics": st,
    }


def erro(status, reason):
    corpo = f'{{"error": {{"code": {status}, "errors": [{{"reason": "{reason}"}}]}}}}'
    return HttpError(httplib2.Response({"status": status}), corpo.encode())


class Req:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class _Recurso:
    def __init__(self, fake, nome):
        self.fake, self.nome = fake, nome

    def list(self, **kw):
        self.fake.chamadas.append((self.nome, kw))
        return Req(lambda: getattr(self.fake, f"_{self.nome}")(kw))


class FakeYT:
    """paginas: {playlist_id: {"p0" | page_token: resposta}}; itens: {video_id: vídeo};
    canais: {handle ou channel_id: canal}; secoes: {channel_id: resposta de channelSections};
    buscas: {texto buscado: channel_id}."""

    def __init__(self, paginas=None, itens=None, canais=None, secoes=None, buscas=None):
        self.paginas = paginas or {}
        self.itens = itens or {}
        self.canais = canais or {}
        self.secoes = secoes or {}
        self.buscas = buscas or {}
        self.chamadas = []

    def playlistItems(self):
        return _Recurso(self, "playlistItems")

    def videos(self):
        return _Recurso(self, "videos")

    def channels(self):
        return _Recurso(self, "channels")

    def channelSections(self):
        return _Recurso(self, "channelSections")

    def search(self):
        return _Recurso(self, "search")

    def n(self, recurso):
        return sum(1 for r, _ in self.chamadas if r == recurso)

    def _playlistItems(self, kw):
        playlist = self.paginas.get(kw["playlistId"])
        if playlist is None:
            raise erro(404, "playlistNotFound")
        return playlist[kw.get("pageToken") or "p0"]

    def _videos(self, kw):
        return {"items": [self.itens[i] for i in kw["id"].split(",") if i in self.itens]}

    def _channels(self, kw):
        if "forHandle" in kw:
            item = self.canais.get(kw["forHandle"])
            if item is None:
                return {"kind": "youtube#channelListResponse", "pageInfo": {"totalResults": 0}}
            return {"kind": "youtube#channelListResponse", "items": [item]}
        return {"items": [self.canais[i] for i in kw["id"].split(",") if i in self.canais]}

    def _channelSections(self, kw):
        return self.secoes.get(kw["channelId"], {"items": []})

    def _search(self, kw):
        cid = self.buscas.get(kw["q"])
        return {"items": [{"id": {"kind": "youtube#channel", "channelId": cid}}] if cid else []}
```

- [ ] **Step 3: Escrever os testes do cliente (vão falhar)**

`tests/test_coletor.py`:

```python
from datetime import datetime, timezone

import pytest
from tenacity import wait_none

from previa.coletor import Cliente, CotaEsgotada
from fakes import Req, erro


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
```

- [ ] **Step 4: Rodar e ver falhar**

Run: `uv run pytest tests/test_coletor.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'previa.coletor'`.

- [ ] **Step 5: Implementar a etapa 1**

`src/previa/coletor.py`:

```python
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
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from googleapiclient.errors import HttpError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

PT = ZoneInfo("America/Los_Angeles")  # a cota zera à meia-noite do horário do Pacífico
DADOS = Path("data")


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
```

- [ ] **Step 6: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_coletor.py -q && uv run ruff check . && uv run ruff format --check .`
Expected: `4 passed`; ruff sem erros (se o format apontar diferenças, rodar `uv run ruff format .`).

- [ ] **Step 7: Ponto de commit (só quando o Vinicius pedir)**

Mensagem sugerida: `feat: add project env and YouTube API client with quota and retries`.

---

### Task 2: Da playlist aos vídeos e o parse de cada vídeo (aula 1.5, etapas 2 e 3)

**Files:**
- Modify: `src/previa/coletor.py` (acrescentar as etapas 2 e 3)
- Modify: `tests/test_coletor.py`

**Interfaces:**
- Consumes: `Cliente.chamar` (Task 1).
- Produces: `playlist_uploads(channel_id: str) -> str`; `listar_ids(cli, playlist_id, conhecidos=frozenset(),
  max_paginas=None) -> list[str]`; `buscar_videos(cli, ids: list[str]) -> list[dict]`;
  `parse_video(item: dict, coletado_em: datetime) -> dict` com as chaves `video_id, channel_id, published_at, title,
  description, tags, category_id, audio_lang, duration_s, vertical, is_short, views, likes, comments, age_days,
  collected_at`; constantes `PARTES`, `MUDOU_PARA_3_MIN`.

- [ ] **Step 1: Escrever os testes (vão falhar)**

Acrescentar a `tests/test_coletor.py` (os imports sobem para o topo do arquivo):

```python
from previa.coletor import buscar_videos, listar_ids, parse_video, playlist_uploads
from fakes import FakeYT, pagina, video

AGORA = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_coletor.py -q`
Expected: FAIL com `ImportError: cannot import name 'buscar_videos'`.

- [ ] **Step 3: Implementar as etapas 2 e 3**

Em `src/previa/coletor.py`, acrescentar `import isodate` aos imports, as constantes junto de `PT` e as funções depois
da etapa 1:

```python
MUDOU_PARA_3_MIN = datetime(2024, 10, 15, tzinfo=timezone.utc)  # antes: Shorts até 60 s
PARTES = "snippet,contentDetails,statistics,player"


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
```

- [ ] **Step 4: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_coletor.py -q && uv run ruff check . && uv run ruff format --check .`
Expected: `13 passed`; ruff sem erros.

- [ ] **Step 5: Ponto de commit (só quando o Vinicius pedir)**

Mensagem sugerida: `feat: list uploads, fetch videos and parse Shorts`.

---

### Task 3: Persistência idempotente, foto dos canais e exportação (aula 1.5, etapa 4, + aula 1.4)

**Files:**
- Modify: `src/previa/coletor.py` (etapa 4)
- Modify: `tests/test_coletor.py`

**Interfaces:**
- Consumes: `parse_video`, `_int`, `Cliente.chamar` (Tasks 1 e 2).
- Produces: `abrir_base(caminho=DADOS/"shorts.duckdb") -> duckdb.DuckDBPyConnection`; `salvar(con, linhas: list[dict])
  -> None`; `buscar_canais(cli, channel_ids: list[str]) -> list[dict]`; `salvar_canais(con, itens: list[dict],
  coletado_em: datetime) -> None`; `consultar_rotulados(con, idade_min=30, teto_por_canal=400) -> pd.DataFrame`;
  `exportar_rotulados(con, destino=DADOS/"shorts_rotulados.parquet", idade_min=30, teto_por_canal=400) -> int`;
  constante `PARTES_CANAL`.

- [ ] **Step 1: Escrever os testes (vão falhar)**

Acrescentar a `tests/test_coletor.py`:

```python
from datetime import date

import pandas as pd

from previa.coletor import (
    abrir_base,
    buscar_canais,
    consultar_rotulados,
    exportar_rotulados,
    salvar,
    salvar_canais,
)
from fakes import CANAL_ID, canal


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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_coletor.py -q`
Expected: FAIL com `ImportError: cannot import name 'abrir_base'`.

- [ ] **Step 3: Implementar a etapa 4**

Em `src/previa/coletor.py`, acrescentar `import duckdb` e `import pandas as pd` aos imports, a constante
`PARTES_CANAL = "snippet,contentDetails,statistics"` junto de `PARTES`, e as funções depois da etapa 3:

```python
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
```

- [ ] **Step 4: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_coletor.py -q && uv run ruff check . && uv run ruff format --check .`
Expected: `21 passed`; ruff sem erros.

- [ ] **Step 5: Ponto de commit (só quando o Vinicius pedir)**

Mensagem sugerida: `feat: persist videos, snapshots and channel stats idempotently in DuckDB`.

---

### Task 4: Orquestração diária, renovação dos 30 dias e agendamento (aula 1.5, etapa 5, + aula 1.8)

**Files:**
- Modify: `src/previa/coletor.py` (etapa 5 e `main`)
- Modify: `tests/test_coletor.py`
- Create: `scripts/coletar.sh`, `scripts/com.previa.coletor.plist`

**Interfaces:**
- Consumes: tudo das Tasks 1 a 3.
- Produces: `coletar_canal(cli, con, channel_id) -> int`; `atualizar_snapshots(cli, con, idade_max=60) -> int`;
  `renovar_vencidos(cli, con, prazo_dias=25) -> dict` (`{"renovados": int, "apagados": int}`);
  `apagar(con, video_ids: list[str]) -> None`; `ler_canais(caminho=DADOS/"canais.csv") -> list[str]`;
  `coletar(cli, con, channel_ids: list[str]) -> dict` (resumo com `canais_ok`, `falhas`, `videos_novos`,
  `snapshots_jovens`, `renovacao`, `parou_por_cota`); `main() -> int` (0 ok, 1 falha de canal, 2 sem chave,
  3 parou por cota); execução por `python -m previa.coletor`.

- [ ] **Step 1: Escrever os testes (vão falhar)**

Acrescentar a `tests/test_coletor.py`:

```python
from datetime import timedelta

from previa.coletor import (
    atualizar_snapshots,
    coletar,
    coletar_canal,
    ler_canais,
    renovar_vencidos,
)

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
    assert (
        coletar_canal(Cliente(FakeYT(), arquivo=None), abrir_base(tmp_path / "t.duckdb"), CANAL_ID)
        == 0
    )


def test_falha_num_canal_nao_derruba_os_outros(tmp_path):
    cli, con = Cliente(_canal_lumina(), arquivo=None), abrir_base(tmp_path / "t.duckdb")
    resumo = coletar(cli, con, ["HCtopico000000000000000", CANAL_ID])
    assert resumo["canais_ok"] == 1 and resumo["videos_novos"] == 60
    assert "HCtopico000000000000000" in resumo["falhas"]


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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_coletor.py -q`
Expected: FAIL com `ImportError: cannot import name 'atualizar_snapshots'`.

- [ ] **Step 3: Implementar a etapa 5**

Em `src/previa/coletor.py`, acrescentar `import os` e `import sys` aos imports e, ao fim do arquivo:

```python
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
                resumo["falhas"][cid] = repr(e)
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
    resumo = coletar(cli, con, ler_canais())
    resumo["rotulados"] = exportar_rotulados(con)
    resumo["unidades_hoje_pt"] = cli.gasto.get(datetime.now(PT).date().isoformat(), 0)
    print(inicio.isoformat(), json.dumps(resumo, ensure_ascii=False))
    if resumo["parou_por_cota"]:
        return 3
    return 1 if resumo["falhas"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Rodar e ver passar; lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `28 passed`; ruff sem erros.

- [ ] **Step 5: Script de execução e agendamento**

`scripts/coletar.sh` (e `chmod +x scripts/coletar.sh`):

```bash
#!/bin/zsh
# Coleta diária (aula 1.5): carrega o .env e roda o coletor. Usado pelo launchd às 05h15.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
set -a; source ./.env; set +a
# o Mac pode acordar sem rede: espera até ~10 min pela API antes de desistir
for _ in {1..20}; do
  curl -s --head --max-time 10 https://www.googleapis.com > /dev/null && break
  sleep 30
done
"$HOME/.local/bin/uv" run python -m previa.coletor >> data/logs/coleta.log 2>&1
```

`scripts/com.previa.coletor.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.previa.coletor</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>/Users/viniciustavarescastiglia/case-galaxies/scripts/coletar.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>5</integer>
    <key>Minute</key>
    <integer>15</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/Users/viniciustavarescastiglia/case-galaxies/data/logs/launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/viniciustavarescastiglia/case-galaxies/data/logs/launchd.err.log</string>
</dict>
</plist>
```

Por que launchd e não cron: se o Mac estiver dormindo às 05h15, o launchd roda a tarefa quando ele acordar; o cron
simplesmente pula o dia. A instalação acontece na Task 7, depois da primeira coleta manual.

Run: `plutil -lint scripts/com.previa.coletor.plist && zsh -n scripts/coletar.sh`
Expected: `scripts/com.previa.coletor.plist: OK` e nenhuma saída do `zsh -n`.

- [ ] **Step 6: Ponto de commit (só quando o Vinicius pedir)**

Mensagem sugerida: `feat: daily collection run with 30-day refresh and launchd schedule`.

---

### Task 5: Curadoria de canais (aula 1.4)

**Files:**
- Create: `src/previa/curadoria.py`
- Create: `tests/test_curadoria.py`

**Interfaces:**
- Consumes: `Cliente`, `DADOS`, `PARTES_CANAL`, `abrir_base`, `buscar_canais`, `buscar_videos`, `listar_ids`,
  `parse_video`, `playlist_uploads` (coletor).
- Produces: `normalizar_handle(texto) -> str`; `achatar_canal(c, handle=None) -> dict`; `resolver_handles(cli, handles)
  -> tuple[pd.DataFrame, list[str]]`; `atualizar_em_lote(cli, channel_ids) -> pd.DataFrame`;
  `handles_em_textos(textos) -> pd.Series`; `canais_em_destaque(cli, channel_id) -> list[str]`;
  `buscar_por_nome(cli_busca, nome) -> str | None`; `faixa(inscritos) -> str | None`; `estratificar(canais) ->
  pd.DataFrame`; `avaliar_canal(cli, canal, hoje, max_paginas=10) -> dict`; `aplicar_criterios(linha, hoje) ->
  tuple[str, str]`; `avaliar_candidatos(cli, candidatos, hoje, cli_busca=None) -> pd.DataFrame` (colunas `COLUNAS`);
  `novos_candidatos(candidatos, planilha) -> pd.DataFrame`; `expandir(cli, con, planilha, candidatos, top=30) ->
  pd.DataFrame`; `main(argv=None) -> int` com as ações `avaliar [--buscar-faltantes]` e `expandir`.

- [ ] **Step 1: Escrever os testes (vão falhar)**

`tests/test_curadoria.py`:

```python
from datetime import datetime, timezone

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
from fakes import CANAL_ID, FakeYT, canal, pagina, video

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
    yt = FakeYT(
        secoes={
            CANAL_ID: {
                "items": [
                    {
                        "snippet": {"type": "multipleChannels"},
                        "contentDetails": {"channels": [destaque]},
                    }
                ]
            }
        },
        canais={destaque: canal(destaque, handle="@destaque")},
    )
    planilha = pd.DataFrame(
        {"channel_id": [CANAL_ID], "handle": ["@luminabeauty"], "incluido": ["sim"]}
    )
    candidatos = pd.DataFrame({"handle": ["@luminabeauty"]})
    novos = expandir(Cliente(yt, arquivo=None), con, planilha, candidatos)
    assert dict(zip(novos["handle"], novos["fonte"])) == {
        "@garagem22": "colab",
        "@destaque": "destaque",
    }
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_curadoria.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'previa.curadoria'`.

- [ ] **Step 3: Implementar**

`src/previa/curadoria.py`:

```python
"""Curadoria de canais (aula 1.4 da trilha da liga).

Funil: candidatos (data/candidatos.csv) -> resolver handles (channels.list forHandle, 1 unidade
cada) -> medir critérios escritos ANTES de olhar resultados (docs/PLANO_DA_BASE.md) -> planilha
data/canais.csv com incluídos E excluídos, cada um com o motivo. A decisão final é humana:
a coluna `incluido` só vira 'sim' de vez depois da revisão do Vinicius.
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from googleapiclient.errors import HttpError

from previa.coletor import (
    DADOS,
    PARTES_CANAL,
    Cliente,
    abrir_base,
    buscar_canais,
    buscar_videos,
    listar_ids,
    parse_video,
    playlist_uploads,
)

MIN_SHORTS = 30  # histórico: pelo menos 30 Shorts
ATIVIDADE_DIAS = 90  # atividade: pelo menos 1 Short nos últimos 90 dias
MIN_CURTIDAS_VISIVEIS = 0.5  # likeCount presente na maioria dos Shorts
MAX_PAGINAS_AVALIACAO = 10  # lê até os 500 uploads mais recentes de cada candidato

# handle: 3 a 30 caracteres; letras, números, "_", "-" e "."; não pode ser e-mail
RE_HANDLE = re.compile(r"(?<![\w.@])@([\w][\w.-]{1,28}[\w])")
FAIXAS = [0, 10_000, 100_000, 1_000_000, np.inf]
ROTULOS = ["até 10 mil", "10–100 mil", "100 mil–1 mi", "mais de 1 mi"]
COLUNAS = [
    "channel_id",
    "handle",
    "nome",
    "titulo",
    "setor",
    "tipo",
    "faixa_inscritos",
    "inscritos_verificacao",
    "pais",
    "n_shorts_estimado",
    "ultimo_short",
    "frac_curtidas_visiveis",
    "incluido",
    "motivo",
    "fonte",
    "observacao",
    "verificado_em",
]


def _vazio(valor) -> bool:
    if isinstance(valor, str):
        return valor.strip() == ""
    return valor is None or bool(pd.isna(valor))


def normalizar_handle(texto: str) -> str:
    """'youtube.com/@Marca/shorts', 'Marca' ou ' @marca ' -> '@marca'."""
    h = texto.strip().split("youtube.com/")[-1].split("/")[0].split("?")[0]
    return "@" + h.lstrip("@").lower()


def achatar_canal(c: dict, handle: str | None = None) -> dict:
    sn, st = c["snippet"], c.get("statistics", {})
    return {
        "channel_id": c["id"],
        "handle": handle or sn.get("customUrl"),
        "titulo": sn["title"],
        "pais": sn.get("country"),  # opcional: muitos canais não preenchem
        "criado_em": sn["publishedAt"],
        "inscritos": None if st.get("hiddenSubscriberCount") else int(st["subscriberCount"]),
        "n_videos": int(st.get("videoCount", 0)),
        "uploads": c["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def resolver_handles(cli, handles: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """forHandle aceita UM handle por chamada: custa 1 unidade por handle."""
    linhas, nao_achados = [], []
    for h in handles:
        resp = cli.chamar(cli.yt.channels().list(part=PARTES_CANAL, forHandle=h))
        itens = resp.get("items", [])  # handle inexistente: sem "items"
        if itens:
            linhas.append(achatar_canal(itens[0], handle=h))
        else:
            nao_achados.append(h)
    return pd.DataFrame(linhas), nao_achados


def atualizar_em_lote(cli, channel_ids: list[str]) -> pd.DataFrame:
    """Com os IDs em mãos, 1 unidade a cada 50 canais."""
    return pd.DataFrame([achatar_canal(c) for c in buscar_canais(cli, channel_ids)])


def handles_em_textos(textos: list[str]) -> pd.Series:
    """Conta menções a @handles em descrições e títulos (colabs, parcerias)."""
    achados = [f"@{h.lower()}" for t in textos for h in RE_HANDLE.findall(t or "")]
    return pd.Series(achados, dtype="str").value_counts()


def canais_em_destaque(cli, channel_id: str) -> list[str]:
    """Seções 'multipleChannels' da página do canal (1 unidade por canal)."""
    req = cli.yt.channelSections().list(part="snippet,contentDetails", channelId=channel_id)
    ids = []
    for s in cli.chamar(req).get("items", []):
        if s["snippet"]["type"] == "multipleChannels":
            ids += s.get("contentDetails", {}).get("channels", [])
    return ids


def buscar_por_nome(cli_busca, nome: str) -> str | None:
    """Reserva para handle não encontrado (aula 1.4): search.list com type=channel, no balde
    próprio de 100 chamadas por dia. Devolve o channel_id do primeiro resultado."""
    req = cli_busca.yt.search().list(
        part="snippet",
        q=nome,
        type="channel",
        regionCode="BR",
        relevanceLanguage="pt",
        maxResults=1,
    )
    itens = cli_busca.chamar(req).get("items", [])
    return itens[0]["id"]["channelId"] if itens else None


def faixa(inscritos) -> str | None:
    if _vazio(inscritos):
        return None
    return str(pd.cut([float(inscritos)], FAIXAS, labels=ROTULOS, right=False)[0])


def estratificar(canais: pd.DataFrame) -> pd.DataFrame:
    """Tabela cruzada setor × tipo × faixa de inscritos: mostra onde a seleção está rala."""
    faixas = pd.cut(canais["inscritos"].astype(float), FAIXAS, labels=ROTULOS, right=False)
    # object em vez de categoria: as margens ('total') não cabem numa coluna categórica
    tabela = pd.crosstab(
        [canais["setor"], canais["tipo"]],
        faixas.astype(object).rename("faixa"),
        margins=True,
        margins_name="total",
    )
    return tabela[[r for r in ROTULOS if r in tabela.columns] + ["total"]]


def avaliar_canal(
    cli, canal: dict, hoje: datetime, max_paginas: int = MAX_PAGINAS_AVALIACAO
) -> dict:
    """Mede os critérios lendo os uploads mais recentes. Só elegibilidade (quantos Shorts,
    atividade, curtidas visíveis): quantas curtidas os vídeos têm não entra na decisão."""
    try:
        ids = listar_ids(cli, playlist_uploads(canal["channel_id"]), max_paginas=max_paginas)
    except HttpError as e:
        if e.status_code != 404:
            raise
        ids = []
    shorts = [v for v in (parse_video(it, hoje) for it in buscar_videos(cli, ids)) if v["is_short"]]
    ultimo = max((v["published_at"] for v in shorts), default=None)
    visiveis = sum(v["likes"] is not None for v in shorts)
    return {
        "n_shorts_estimado": len(shorts),
        "ultimo_short": ultimo.date().isoformat() if ultimo else None,
        "frac_curtidas_visiveis": visiveis / len(shorts) if shorts else None,
        "leu_tudo": len(ids) < max_paginas * 50,
    }


def aplicar_criterios(linha: dict, hoje: datetime) -> tuple[str, str]:
    """Critérios da aula 1.4, escritos antes de olhar resultados. Devolve (incluido, motivo)
    com incluido em {'sim', 'não', 'revisar'}; o olho humano confirma país, idioma e setor."""
    pais = linha.get("pais")
    if not _vazio(pais) and pais != "BR":
        return "não", f"país {pais}"
    if _vazio(linha.get("inscritos_verificacao")):
        return "não", "inscritos ocultos"
    n = int(linha.get("n_shorts_estimado") or 0)
    if n < MIN_SHORTS:
        if linha.get("leu_tudo", True):
            return "não", f"menos de {MIN_SHORTS} Shorts ({n})"
        return "revisar", f"{n} Shorts entre os {MAX_PAGINAS_AVALIACAO * 50} uploads mais recentes"
    ultimo = linha.get("ultimo_short")
    corte = hoje - timedelta(days=ATIVIDADE_DIAS)
    if _vazio(ultimo) or datetime.fromisoformat(ultimo).replace(tzinfo=timezone.utc) < corte:
        return "não", f"sem Short há mais de {ATIVIDADE_DIAS} dias"
    frac = linha.get("frac_curtidas_visiveis")
    if not _vazio(frac) and float(frac) <= MIN_CURTIDAS_VISIVEIS:
        return "não", "curtidas ocultas"
    if _vazio(pais):
        return "revisar", "país não informado: conferir idioma e país"
    return "sim", "passa nos critérios automáticos"


def avaliar_candidatos(
    cli, candidatos: pd.DataFrame, hoje: datetime, cli_busca=None
) -> pd.DataFrame:
    """candidatos: colunas handle, nome, setor, tipo, fonte, observacao. Devolve as linhas da
    planilha de curadoria (COLUNAS), incluindo os excluídos e o motivo de cada um."""
    cand = candidatos.fillna("").assign(handle=lambda d: d["handle"].map(normalizar_handle))
    cand = cand.drop_duplicates("handle")
    resolvidos, _ = resolver_handles(cli, cand["handle"].tolist())
    por_handle = {r["handle"]: r for r in resolvidos.to_dict("records")}
    linhas = []
    for c in cand.to_dict("records"):
        base = {k: c.get(k, "") for k in ("handle", "nome", "setor", "tipo", "fonte", "observacao")}
        base["verificado_em"] = hoje.date().isoformat()
        r, por_busca = por_handle.get(c["handle"]), False
        if r is None and cli_busca is not None and c.get("nome"):
            cid = buscar_por_nome(cli_busca, c["nome"])
            achados = atualizar_em_lote(cli, [cid]) if cid else pd.DataFrame()
            if len(achados):
                r, por_busca = achados.iloc[0].to_dict(), True
        if r is None:
            linhas.append({**base, "incluido": "não", "motivo": "handle não encontrado"})
            continue
        linha = {
            **base,
            "channel_id": r["channel_id"],
            "titulo": r["titulo"],
            "pais": r["pais"],
            "inscritos_verificacao": r["inscritos"],
            "faixa_inscritos": faixa(r["inscritos"]),
            **avaliar_canal(cli, r, hoje),
        }
        linha["incluido"], linha["motivo"] = aplicar_criterios(linha, hoje)
        if por_busca:
            linha["fonte"] = f"{base['fonte']}+busca" if base["fonte"] else "busca"
            achado = f"handle não encontrado; a busca achou {r['handle']} ({r['titulo']})"
            linha["observacao"] = "; ".join(x for x in (base["observacao"], achado) if x)
            if linha["incluido"] == "sim":
                linha["incluido"], linha["motivo"] = (
                    "revisar",
                    "achado por busca: confirmar o canal",
                )
        linhas.append(linha)
    planilha = pd.DataFrame(linhas).reindex(columns=COLUNAS)
    duplicado = planilha["channel_id"].notna() & planilha.duplicated("channel_id")
    if duplicado.any():
        planilha.loc[duplicado, ["incluido", "motivo"]] = ["não", "canal repetido na lista"]
    return planilha


def novos_candidatos(candidatos: pd.DataFrame, planilha: pd.DataFrame) -> pd.DataFrame:
    """Só os candidatos que ainda não estão na planilha: a revisão humana nunca é sobrescrita."""
    ja = set(planilha["handle"].dropna().map(normalizar_handle)) if len(planilha) else set()
    return candidatos[~candidatos["handle"].map(normalizar_handle).isin(ja)]


def expandir(
    cli, con, planilha: pd.DataFrame, candidatos: pd.DataFrame, top: int = 30
) -> pd.DataFrame:
    """Candidatos novos contra o viés de sobrevivência (aula 1.4): handles citados nas
    descrições dos canais incluídos (fonte 'colab') e canais em destaque na página deles
    (fonte 'destaque'). Setor e tipo ficam para a revisão humana."""
    incluidos = planilha[planilha["incluido"].fillna("").str.strip().str.lower() == "sim"]
    ids = incluidos["channel_id"].dropna().tolist()
    ja = set(candidatos["handle"].map(normalizar_handle)) | set(planilha["handle"].dropna())
    textos = [
        r[0]
        for r in con.execute(
            "SELECT description FROM videos WHERE list_contains(?, channel_id)", [ids]
        ).fetchall()
    ]
    novos = [
        {
            "handle": h,
            "nome": "",
            "setor": "",
            "tipo": "",
            "fonte": "colab",
            "observacao": f"citado {n}x nas descrições da base",
        }
        for h, n in handles_em_textos(textos).items()
        if h not in ja
    ][:top]
    conhecidos = set(planilha["channel_id"].dropna())
    em_destaque = [c for cid in ids for c in canais_em_destaque(cli, cid) if c not in conhecidos]
    for r in atualizar_em_lote(cli, list(dict.fromkeys(em_destaque))).to_dict("records"):
        h = normalizar_handle(r["handle"]) if r.get("handle") else None
        if h and h not in ja:
            novos.append(
                {
                    "handle": h,
                    "nome": r["titulo"],
                    "setor": "",
                    "tipo": "",
                    "fonte": "destaque",
                    "observacao": "em destaque na página de um incluído",
                }
            )
    colunas = ["handle", "nome", "setor", "tipo", "fonte", "observacao"]
    return pd.DataFrame(novos, columns=colunas).drop_duplicates("handle")


def main(argv=None) -> int:
    from googleapiclient.discovery import build

    parser = argparse.ArgumentParser(prog="python -m previa.curadoria")
    parser.add_argument("acao", choices=["avaliar", "expandir"])
    parser.add_argument(
        "--buscar-faltantes",
        action="store_true",
        help="usa search.list (balde de 100/dia) para handles não encontrados",
    )
    args = parser.parse_args(argv)
    chave = os.environ.get("YOUTUBE_API_KEY")
    if not chave:
        print(
            "YOUTUBE_API_KEY não definida: carregue o .env (set -a; source .env; set +a).",
            file=sys.stderr,
        )
        return 2
    yt = build("youtube", "v3", developerKey=chave, cache_discovery=False)
    cli, hoje = Cliente(yt), datetime.now(timezone.utc)
    caminho_cand, caminho_plan = DADOS / "candidatos.csv", DADOS / "canais.csv"
    candidatos = pd.read_csv(caminho_cand, dtype=str).fillna("")
    planilha = (
        pd.read_csv(caminho_plan, dtype=str)
        if caminho_plan.exists()
        else pd.DataFrame(columns=COLUNAS)
    )
    if args.acao == "expandir":
        novos = expandir(cli, abrir_base(), planilha, candidatos)
        pd.concat([candidatos, novos], ignore_index=True).to_csv(caminho_cand, index=False)
        print(f"{len(novos)} candidatos novos anexados a {caminho_cand}")
    else:
        cli_busca = None
        if args.buscar_faltantes:
            cli_busca = Cliente(yt, limite=90, arquivo=DADOS / "cota_busca.json")
        avaliados = avaliar_candidatos(cli, novos_candidatos(candidatos, planilha), hoje, cli_busca)
        planilha = pd.concat([planilha, avaliados], ignore_index=True).reindex(columns=COLUNAS)
        planilha.to_csv(caminho_plan, index=False)
        print(planilha["incluido"].value_counts().to_string())
        incluidos = planilha[planilha["incluido"] == "sim"]
        if len(incluidos):
            tabela = incluidos.assign(inscritos=incluidos["inscritos_verificacao"])
            print(estratificar(tabela).to_string())
    print("unidades gastas hoje (PT):", cli.gasto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Rodar e ver passar; lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `50 passed` (28 do coletor + 22 da curadoria, contando cada caso parametrizado); ruff sem erros.

- [ ] **Step 5: Ponto de commit (só quando o Vinicius pedir)**

Mensagem sugerida: `feat: channel curation funnel with criteria, stratification and expansion`.

---

### Task 6: Lista de candidatos e plano da base (aulas 1.3 e 1.4)

**Files:**
- Create: `data/candidatos.csv`
- Create: `docs/PLANO_DA_BASE.md`
- Create: `tests/test_candidatos.py`
- Modify: `.gitignore` (acrescentar `!data/candidatos.csv` logo abaixo de `!data/canais.csv`)

**Interfaces:**
- Consumes: o formato de `avaliar_candidatos` (Task 5): colunas `handle, nome, setor, tipo, fonte, observacao`.
- Produces: `data/candidatos.csv`, lido por `python -m previa.curadoria avaliar`.

- [ ] **Step 1: Escrever o teste do formato da lista (vai falhar)**

`tests/test_candidatos.py`:

```python
from pathlib import Path

import pandas as pd

CAMINHO = Path(__file__).parents[1] / "data" / "candidatos.csv"


def test_lista_de_candidatos_tem_o_formato_e_o_mix_do_plano():
    df = pd.read_csv(CAMINHO, dtype=str).fillna("")
    assert list(df.columns) == ["handle", "nome", "setor", "tipo", "fonte", "observacao"]
    assert df["handle"].str.startswith("@").all()
    assert df["handle"].str.lower().is_unique
    assert set(df["setor"]) == {"beleza", "automotivo"}
    assert set(df["tipo"]) == {"marca", "criador"}
    assert len(df) >= 90  # aula 1.4: pelo menos 90 candidatos
    por_celula = df.groupby(["setor", "tipo"]).size()
    assert len(por_celula) == 4 and (por_celula >= 15).all()
    assert (df["nome"] != "").all()  # o nome alimenta a busca de reserva
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_candidatos.py -q`
Expected: FAIL com `FileNotFoundError` para `data/candidatos.csv`.

- [ ] **Step 3: Escrever a lista**

`data/candidatos.csv` com pelo menos 90 linhas e as colunas do teste. Regras de composição:
- Quatro células (beleza × marca, beleza × criador, automotivo × marca, automotivo × criador), com pelo menos 15
  candidatos cada, ~110 no total, porque parte dos handles não vai resolver ou não vai passar nos critérios.
- Tamanhos variados: grandes, médios e pequenos, de propósito (aula 1.4, viés de sobrevivência).
- `fonte = "lista manual"` em todos: a expansão por colabs e destaques (Task 7) traz o terço de outras fontes que a
  aula pede.
- Automotivo: carros, picapes e SUVs (montadoras, varejo e serviços, revistas, criadores); motos ficam fora, como
  proposta a confirmar no plano da base.
- `nome` = o nome do canal como aparece no YouTube (usado pela busca de reserva); `observacao` = nota curta quando
  útil (ex.: "revista", "varejo").

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_candidatos.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Escrever o plano da base (exercício da aula 1.3)**

`docs/PLANO_DA_BASE.md`, em PT-BR, com as cinco partes da aula 1.3 e os números **propostos**, marcando com
**[decisão do Vinicius]** o que ele confirma na revisão:

1. **Metas:** 60 a 70 canais incluídos, cerca de metade em cada setor; pelo menos 10 marcas por setor
   [decisão do Vinicius]; nenhuma faixa de inscritos vazia entre 10 mil e 1 milhão em cada setor (a Lumina tem 180 mil);
   5 a 15 mil Shorts rotulados; teto de 400 Shorts rotulados por canal [decisão do Vinicius]; rótulo = primeiro
   snapshot com 30 dias ou mais; automotivo = carros, picapes e SUVs, sem motos [decisão do Vinicius].
2. **Critérios de inclusão** (os da aula 1.4, escritos antes de olhar resultados): maioria dos Shorts no setor;
   português e público brasileiro (conferência humana); pelo menos 1 Short nos últimos 90 dias; pelo menos 30 Shorts;
   curtidas visíveis na maioria dos Shorts; inscritos visíveis; fora cortes e reuploads, compilações, agregadores e
   conteúdo infantil.
3. **Fontes:** YouTube Data API como base principal (treino e teste; não tem retenção, CTR nem impressões);
   SMTPD e MicroLens-100K só como apoio opcional (não são Shorts de marcas brasileiras).
4. **Custo:** avaliação única de ~110 candidatos, até ~20 unidades cada (até ~2.200); primeira coleta completa de
   65 canais com 300 uploads cada = 845 unidades (`custo_coleta` da aula 1.3); coletas diárias de ~100 a 300 unidades.
5. **Cronograma:** chave e avaliação automática em 30/09 ou 01/10; revisão dos canais e primeira coleta até 02/10
   (corte de publicação 02/09); coleta diária às 05h15; congelar a base de treino em 12/10 (corte 12/09); última coleta
   de rótulos em 19/10 (corte 19/09); apagar os dados brutos em 30/10.
6. **Riscos:** poucas marcas brasileiras com 30 ou mais Shorts (mitigação: expansão por colabs e destaques; se faltar,
   documentar, sem afrouxar critério depois de ver curtidas); handles errados na lista (busca de reserva e correção
   manual); viés de sobrevivência da lista de memória (expansão e coluna `fonte`); Mac dormindo às 05h15 (o launchd
   roda ao acordar; a coleta é idempotente).

- [ ] **Step 6: Ponto de commit (só quando o Vinicius pedir)**

Mensagem sugerida: `docs: add candidate channel list and base plan draft`.

---

### Task 7: Execução com a chave: avaliar, revisar, coletar e agendar

Pré-requisito (Vinicius, ~15 min): criar a chave seguindo o passo 1 do "Monte sua base" (projeto no Google Cloud,
ativar a YouTube Data API v3, criar a chave e restringir à YouTube Data API v3), depois `cp .env.example .env` e colar a
chave no `.env`. A chave nunca vai para o chat.

- [ ] **Step 1: Testar a chave (1 unidade)**

```bash
set -a; source .env; set +a
uv run python -c "import os; from googleapiclient.discovery import build; \
yt = build('youtube', 'v3', developerKey=os.environ['YOUTUBE_API_KEY'], cache_discovery=False); \
c = yt.channels().list(part='snippet', forHandle='@YouTube').execute()['items'][0]; \
print(c['id'], c['snippet']['title'])"
```

Expected: `UCBR8-60-B28hp2BmDPdntcQ YouTube`.

- [ ] **Step 2: Avaliar os candidatos**

Run: `set -a; source .env; set +a; uv run python -m previa.curadoria avaliar --buscar-faltantes`
Expected: `data/canais.csv` com uma linha por candidato, a contagem de `sim`/`não`/`revisar`, a tabela
setor × tipo × faixa e o gasto do dia abaixo de ~2.500 unidades.

- [ ] **Step 3: Revisão humana (Vinicius, 30 a 45 min)**

Claude monta no chat um resumo por célula setor × tipo: todos os `sim` e `revisar` com título e link
(`https://www.youtube.com/channel/<channel_id>`), os excluídos agrupados por motivo e a tabela de estratificação.
O Vinicius confirma ou corrige setor, tipo e `incluido` (reuploads, conteúdo infantil, setor ambíguo, país) e decide os
itens [decisão do Vinicius] do plano da base. Claude aplica as correções em `data/canais.csv`, com o motivo de cada uma.

- [ ] **Step 4: Primeira coleta completa**

Run: `./scripts/coletar.sh && tail -n 1 data/logs/coleta.log`
Expected: uma linha JSON com `canais_ok` igual ao número de incluídos, `falhas` vazio, `parou_por_cota` nulo,
`rotulados` entre 5 e 15 mil (se ficar abaixo, anotar no data card) e `unidades_hoje_pt` somando menos de ~3.500 com a
avaliação.

- [ ] **Step 5: Conferências de sanidade**

```bash
uv run python - <<'EOF'
import duckdb
con = duckdb.connect("data/shorts.duckdb", read_only=True)
print(con.execute("""
    SELECT count(*) AS videos, sum(is_short::INT) AS shorts,
           round(avg((vertical IS NULL)::INT), 3) AS frac_sem_orientacao
    FROM videos""").df())
print(con.execute("""
    SELECT round(avg((likes IS NULL)::INT), 3) AS frac_curtidas_ocultas
    FROM snapshots""").df())
EOF
```

Expected: `frac_curtidas_ocultas` abaixo de 0,10 (senão, rever os canais que puxam esse número) e
`frac_sem_orientacao` anotada para o data card. Depois do exame, o Vinicius faz o exercício da aula 1.5: comparar os
Shorts prováveis de um canal pequeno com a aba "Shorts" dele.

- [ ] **Step 6: Expansão contra o viés de sobrevivência**

Run: `set -a; source .env; set +a; uv run python -m previa.curadoria expandir && uv run python -m previa.curadoria avaliar`
Expected: novos candidatos com `fonte` `colab` ou `destaque` em `data/candidatos.csv` e avaliados em
`data/canais.csv`; eles entram na próxima revisão (Step 3) e na coleta seguinte.

- [ ] **Step 7: Agendar a coleta diária**

```bash
cp scripts/com.previa.coletor.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.previa.coletor.plist
launchctl kickstart -k gui/$(id -u)/com.previa.coletor   # roda uma vez agora, para testar
sleep 60 && tail -n 1 data/logs/coleta.log
```

Expected: uma linha JSON nova no log (com `videos_novos` perto de zero, porque a coleta completa já rodou).
Para desfazer: `launchctl bootout gui/$(id -u)/com.previa.coletor`.

---

### Task 8: Esquema mínimo validado e data card (aula 1.8)

**Files:**
- Create: `src/previa/esquema.py`
- Create: `tests/test_esquema.py`
- Create: `docs/DATA_CARD.md`

**Interfaces:**
- Consumes: `consultar_rotulados`, `abrir_base`, `DADOS` (coletor); `data/canais.csv` (colunas `channel_id`,
  `setor`).
- Produces: `ChannelStats`, `Post` (Pydantic, esquema mínimo por registro); `BaseShorts` (pandera, tabela);
  `montar_base_minima(rotulados: pd.DataFrame, setores: dict[str, str], inscritos: pd.DataFrame) -> pd.DataFrame`;
  `exportar_base_minima(con, caminho_canais=DADOS/"canais.csv", destino=DADOS/"base_minima.parquet") -> int`.

- [ ] **Step 1: Escrever os testes (vão falhar)**

`tests/test_esquema.py`:

```python
import pandas as pd
import pandera.pandas as pa
import pytest

from previa.esquema import BaseShorts, Post, montar_base_minima


def _rotulados():
    return pd.DataFrame(
        {
            "video_id": ["Lm9Qk2VbT0s", "aB3_x-Z9k0Q"],
            "channel_id": ["UCabc123LuminaBeauty0000"] * 2,
            "published_at": pd.to_datetime(["2026-08-01T22:00:05Z", "2026-08-03T19:00:00Z"]),
            "title": ["1 produto, 20 segundos", "Pele natural em 5 passos"],
            "description": ["Base nova #shorts", ""],
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


def test_base_minima_tem_o_esquema_do_briefing_e_valida():
    base = montar_base_minima(_rotulados(), {"UCabc123LuminaBeauty0000": "beleza"}, _inscritos())
    BaseShorts.validate(base, lazy=True)
    assert base.loc[0, "media_ref"] == "https://www.youtube.com/shorts/Lm9Qk2VbT0s"
    assert base.loc[0, "caption"] == "1 produto, 20 segundos\nBase nova #shorts"
    assert base.loc[0, "subscribers_at_collection"] == 180000


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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_esquema.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'previa.esquema'`.

- [ ] **Step 3: Implementar**

`src/previa/esquema.py`:

```python
"""Esquema mínimo do briefing, em código (aula 1.8): Pydantic valida cada registro e pandera
valida a tabela de treino. Rode a validação a cada exportação."""

from typing import Literal

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from previa.coletor import DADOS, consultar_rotulados

SETORES = ["moda", "beleza", "automotivo"]


class ChannelStats(BaseModel):
    """Estatísticas do canal NO MOMENTO DA COLETA (não da publicação)."""

    subscribers: int | None = Field(default=None, ge=0)  # None = canal oculta inscritos
    video_count: int = Field(ge=0)
    collected_at: AwareDatetime


class Post(BaseModel):
    """Esquema mínimo do briefing, um registro por Short."""

    post_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")  # formato observado dos IDs
    channel_id: str = Field(pattern=r"^UC[A-Za-z0-9_-]{22}$")
    published_at: AwareDatetime
    caption: str  # título + descrição
    media_ref: str = Field(pattern=r"^https://www\.youtube\.com/shorts/")  # link, não arquivo
    metric_target: int | None = Field(ge=0)  # likeCount; None = oculto
    target_age_days: float = Field(ge=30)  # idade no snapshot do rótulo
    channel_stats_at_collection: ChannelStats
    sector: Literal["moda", "beleza", "automotivo"]

    @model_validator(mode="after")
    def coleta_depois_da_publicacao(self):
        if self.channel_stats_at_collection.collected_at <= self.published_at:
            raise ValueError("coleta anterior à publicação: confira fusos e datas")
        return self


class BaseShorts(pa.DataFrameModel):
    """A tabela achatada que vai para o treino (um Short por linha)."""

    post_id: Series[str] = pa.Field(unique=True, str_matches=r"^[A-Za-z0-9_-]{11}$")
    channel_id: Series[str] = pa.Field(str_startswith="UC")
    published_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "us", "tz": "UTC"})
    caption: Series[str]
    media_ref: Series[str] = pa.Field(str_startswith="https://www.youtube.com/shorts/")
    metric_target: Series[pd.Int64Dtype] = pa.Field(ge=0, nullable=True)
    target_age_days: Series[float] = pa.Field(ge=30)
    subscribers_at_collection: Series[pd.Int64Dtype] = pa.Field(ge=0, nullable=True)
    collected_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "us", "tz": "UTC"})
    sector: Series[str] = pa.Field(isin=SETORES)

    @pa.dataframe_check
    def coleta_depois_da_publicacao(cls, df: pd.DataFrame) -> Series[bool]:
        return df["collected_at"] > df["published_at"]

    @pa.dataframe_check
    def no_maximo_10pct_sem_curtidas(cls, df: pd.DataFrame) -> bool:
        # checagem no DataFrame: checks de coluna ignoram os nulos quando nullable=True
        return df["metric_target"].isna().mean() <= 0.10

    class Config:
        strict = True  # coluna extra ou faltando = erro
        coerce = True  # converte tipos quando possível (ex.: "2410" -> 2410)


def montar_base_minima(
    rotulados: pd.DataFrame, setores: dict[str, str], inscritos: pd.DataFrame
) -> pd.DataFrame:
    """Shorts rotulados (consultar_rotulados) no esquema mínimo, com o setor da planilha e os
    inscritos do dia do rótulo (canais_snapshots). Os inscritos ficam para estratificação e
    análise de erros, fora das features de treino (aula 1.7)."""
    base = rotulados.assign(collected_date=rotulados["collected_at"].dt.date)
    base = base.merge(inscritos, on=["channel_id", "collected_date"], how="left")
    return pd.DataFrame(
        {
            "post_id": base["video_id"],
            "channel_id": base["channel_id"],
            "published_at": base["published_at"],
            "caption": base["title"].fillna("") + "\n" + base["description"].fillna(""),
            "media_ref": "https://www.youtube.com/shorts/" + base["video_id"],
            "metric_target": base["likes"].astype("Int64"),
            "target_age_days": base["age_days"].astype(float),
            "subscribers_at_collection": base["subscribers"].astype("Int64"),
            "collected_at": base["collected_at"],
            "sector": base["channel_id"].map(setores),
        }
    )


def exportar_base_minima(
    con, caminho_canais=DADOS / "canais.csv", destino=DADOS / "base_minima.parquet"
) -> int:
    planilha = pd.read_csv(caminho_canais, dtype=str)
    setores = dict(zip(planilha["channel_id"], planilha["setor"], strict=False))
    inscritos = con.execute(
        "SELECT channel_id, collected_date, subscribers FROM canais_snapshots"
    ).df()
    inscritos["collected_date"] = pd.to_datetime(inscritos["collected_date"]).dt.date
    base = montar_base_minima(consultar_rotulados(con), setores, inscritos)
    BaseShorts.validate(base, lazy=True)  # erro aqui: corrija na coleta, não na validação
    base.to_parquet(destino, index=False)
    return len(base)
```

- [ ] **Step 4: Rodar e ver passar; lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `54 passed` (50 + 1 da lista de candidatos + 3 do esquema); ruff sem erros.

- [ ] **Step 5: Exportar a base mínima de verdade (depois da primeira coleta)**

```bash
uv run python -c "from previa.coletor import abrir_base; \
from previa.esquema import exportar_base_minima; print(exportar_base_minima(abrir_base()))"
```

Expected: o número de Shorts rotulados, sem `SchemaErrors`.

- [ ] **Step 6: Rascunho do data card**

`docs/DATA_CARD.md` a partir do template da liga (`docs/projetos/galaxies/templates.mdx`, seção Data card), com os
números tirados da base (período dos vídeos, datas de coleta, volume por setor × tipo × faixa, heurística de Shorts,
filtro de idade, taxa de curtidas ocultas, teto por canal, `frac_sem_orientacao`). As seções "O que esta base NÃO
representa" e "Vieses conhecidos" ficam como `TODO(human)` com perguntas-guia da aula 1.8, porque são o exercício de
reflexão do Vinicius.

- [ ] **Step 7: Ponto de commit (só quando o Vinicius pedir)**

Mensagem sugerida: `feat: minimal schema with pydantic and pandera; data card draft`.

---

## Portão de aprendizado (antes do módulo 2)

Antes de a modelagem usar a base, o Vinicius explica em 3 a 5 linhas o que o coletor faz e por quê (teach-back do
`CLAUDE.md`), com base nas aulas 1.4, 1.5 e 1.7. Claude atualiza as seções "Estado" e "Comandos" do `CLAUDE.md` com os
comandos que passaram a existir.

---

## Apêndice: execução em 30/09/2026

Tasks 1 a 6 completas; Task 8, Steps 1 a 4. A Task 7 e a Task 8, Steps 5 e 6, esperam a chave da API e a primeira
coleta. Suíte final: 64 testes passando; ruff limpo. Sem commits, porque ainda não há repositório git.

**Decisões tomadas durante a execução**

| Onde | Decisão | Por quê | Custo se errado |
| :--- | :--- | :--- | :--- |
| Setup | executar sem worktree e sem commits; progresso registrado à mão em `.superpowers/` | não há git, e o `CLAUDE.md` proíbe commits sem pedido | não há histórico por tarefa |
| Task 1 | ruff ignora a regra UP017 (`datetime.UTC`) | as aulas usam `timezone.utc`, e o código segue as aulas | nenhum funcional |
| Task 1 | ruff exclui `*.md` | o ruff 0.16 formata blocos de código dentro de Markdown e reformatou este plano | nenhum |
| Task 5 | `zip(..., strict=True)` no teste de `expandir` | a regra B905 do ruff exige | nenhum |
| Task 8 | teste novo para `exportar_base_minima` (lacuna do plano), com RED antes do GREEN | toda função precisa de um teste que falhou primeiro | nenhum |

**Revisão final** (feita pelo próprio executor, sem subagente): quatro achados corrigidos, cada um com um teste que
falhou antes.

1. **Crítico:** a chave da API vazava em mensagens de erro, porque o `HttpError` do Google imprime a URL com `key=`.
   Correção: `sem_chave()` antes de qualquer log, mais proteção geral nos dois `main`.
2. **Importante:** a curadoria perdia todo o trabalho se a cota acabasse no meio, e um erro num canal derrubava a
   avaliação inteira. Correção: resolução canal a canal, erro isolado (vira `revisar`), parada limpa por cota e busca
   desligada quando o balde de busca acaba.
3. **Importante:** o teste da lista de candidatos quebraria depois de `expandir`. Correção: as regras de composição
   valem para a lista manual; as linhas da expansão chegam com setor e tipo em branco.
4. **Importante:** `exportar_base_minima` exportava canais excluídos na revisão. Correção: só canais com
   `incluido == 'sim'`.

**Pequenos, adiados**

- A primeira coleta lê a playlist de uploads inteira; canais de mídia grandes custam mais cota (`max_paginas`
  resolveria).
- Metadados de vídeos que não são Shorts ficam na tabela `videos` sem renovação; bastaria guardar só o ID deles.
- Dados de canais excluídos continuam no DuckDB.
- O DuckDB trava o arquivo se `curadoria expandir` rodar junto com a coleta agendada.
- `normalizar_handle` não entende URLs do tipo `/channel/UC…`.
