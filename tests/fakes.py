"""Respostas falsas no formato real da YouTube Data API (aulas 1.4 e 1.5).

Os testes rodam sem chave: FakeYT imita yt.<recurso>().list(...).execute().
"""

import io
import zlib

import httplib2
import numpy as np
from googleapiclient.errors import HttpError
from PIL import Image

CANAL_ID = "UCabc123LuminaBeauty0000"


def thumbnail(semente, brilho=0):
    """JPEG 480x360 como o hqdefault de um Short: faixas pretas dos lados e a arte vertical no
    meio. A mesma semente dá a mesma arte; `brilho` imita a mesma arte com outra exposição."""
    rng = np.random.default_rng(semente)
    blocos = (rng.integers(30, 200, size=(8, 9)) + brilho).astype(np.uint8)
    arte = Image.fromarray(blocos).resize((202, 360), Image.NEAREST)
    quadro = Image.new("L", (480, 360), 0)
    quadro.paste(arte, (139, 0))
    buf = io.BytesIO()
    quadro.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def thumbnail_por_id(video_id):
    """Uma arte diferente por vídeo: o `baixar` falso dos testes que não tratam de duplicatas."""
    return thumbnail(zlib.crc32(video_id.encode()))


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


def erro(status, reason, uri=None):
    corpo = f'{{"error": {{"code": {status}, "errors": [{{"reason": "{reason}"}}]}}}}'
    return HttpError(httplib2.Response({"status": status}), corpo.encode(), uri=uri)


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
        if isinstance(playlist, Exception):  # simula um erro da API nessa playlist
            raise playlist
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
        """buscas[q] é um channel_id ou uma lista deles (um item por resultado, como a API:
        com type=video, cada item é um vídeo e o canal vem em snippet.channelId)."""
        achados = self.buscas.get(kw["q"]) or []
        if isinstance(achados, str):
            achados = [achados]
        tipo = kw.get("type", "channel")
        return {
            "items": [
                {
                    "id": {"kind": "youtube#video", "videoId": f"v{i}"}
                    if tipo == "video"
                    else {"kind": "youtube#channel", "channelId": cid},
                    "snippet": {"channelId": cid},
                }
                for i, cid in enumerate(achados)
            ]
        }
