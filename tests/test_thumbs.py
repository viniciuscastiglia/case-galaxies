"""Testes do download das thumbnails (decisão #12; aula 4.2), sem rede: o `httpx.MockTransport`
faz o papel do i.ytimg.com, como na aula."""

import httpx
import numpy as np
import pytest

from previa.thumbs import (
    BloqueioDoYouTube,
    baixar_thumbnail,
    baixar_todas,
    recortar_9_16,
    remover_barras,
)

VID = "aB3dE5fG7hI"


def _cliente(responder):
    pedidos = []

    def servidor(req):
        pedidos.append(req.url.path)
        return responder(req, pedidos)

    return httpx.Client(transport=httpx.MockTransport(servidor)), pedidos


def test_tenta_do_maior_para_o_menor_e_repete_o_transitorio(tmp_path):
    def responder(req, pedidos):
        if "hqdefault" in req.url.path and pedidos.count(req.url.path) == 1:
            return httpx.Response(404)  # listada mas inexistente: vai para a próxima
        if "mqdefault" in req.url.path and pedidos.count(req.url.path) == 1:
            return httpx.Response(503)  # transitório: tenta de novo
        return httpx.Response(200, content=b"\xff\xd8\xff jpeg falso")

    cliente, pedidos = _cliente(responder)
    caminho = baixar_thumbnail(VID, tmp_path, cliente, espera=0)
    assert caminho == tmp_path / f"{VID}.jpg" and caminho.read_bytes().startswith(b"\xff\xd8")
    hq, mq = f"/vi/{VID}/hqdefault.jpg", f"/vi/{VID}/mqdefault.jpg"
    assert pedidos == [hq, mq, mq]


def test_cache_nao_baixa_duas_vezes(tmp_path):
    cliente, pedidos = _cliente(lambda req, p: httpx.Response(200, content=b"\xff\xd8\xff"))
    baixar_thumbnail(VID, tmp_path, cliente, espera=0)
    baixar_thumbnail(VID, tmp_path, cliente, espera=0)
    assert len(pedidos) == 1


def test_sem_nenhum_tamanho_devolve_none(tmp_path):
    cliente, _ = _cliente(lambda req, p: httpx.Response(404))
    assert baixar_thumbnail(VID, tmp_path, cliente, espera=0) is None
    assert not list(tmp_path.glob("*.jpg"))


def test_lote_registra_falha_e_para_no_bloqueio(tmp_path):
    def responder(req, pedidos):
        if "bbbbbbbbbbb" in req.url.path:
            return httpx.Response(404)
        if "ccccccccccc" in req.url.path:
            return httpx.Response(429)
        return httpx.Response(200, content=b"\xff\xd8\xff")

    cliente, _ = _cliente(responder)
    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd"]
    with pytest.raises(BloqueioDoYouTube):
        baixar_todas(ids, tmp_path, cliente, pausa_s=0, espera=0)
    assert (tmp_path / "aaaaaaaaaaa.jpg").exists() and not (tmp_path / "ddddddddddd.jpg").exists()
    assert "bbbbbbbbbbb" in (tmp_path / "_falhas.txt").read_text()


def test_remover_barras_volta_a_9_16():
    rng = np.random.default_rng(0)
    thumb = rng.integers(0, 12, (360, 480, 3), dtype=np.uint8)  # faixas "pretas" com ruído
    thumb[:, 139:341] = rng.integers(30, 255, (360, 202, 3))
    assert remover_barras(thumb).shape == (360, 202, 3)
    preta = np.zeros((90, 120, 3), dtype=np.uint8)
    assert remover_barras(preta).shape == preta.shape  # toda preta: devolve como veio


def test_recortar_9_16_pega_a_faixa_central():
    # a thumbnail de um Short (480 × 360) traz o quadro vertical no centro e, nas laterais, uma
    # cópia desfocada dele (não faixas pretas): o recorte fixo fica com as colunas 139 a 340
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:, 139:341] = 200
    r = recortar_9_16(img)
    assert r.shape == (360, 202, 3) and (r == 200).all()
