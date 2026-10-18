"""Duplicatas (aula 2.2, família 5; decisão #1 de 14/10): mesmo canal, título normalizado e
duração iguais E a mesma thumbnail (dHash a até 10 bits da cópia mais antiga)."""

from pathlib import Path

import httpx
import pandas as pd
import pytest
from tenacity import wait_none

from previa import duplicatas
from previa.duplicatas import (
    baixar_thumbnail,
    dhash,
    distancia,
    grupo_de_duplicatas,
    hashes_das_thumbnails,
    ler_limiar,
    linhas_do_modelo,
)

from fakes import CANAL_ID, thumbnail

OUTRO = "UCdef456Garagem2200000000"


def _posts(*linhas):
    """linhas: (video_id, channel_id, published_at, title, duration_s)."""
    df = pd.DataFrame(
        linhas, columns=["video_id", "channel_id", "published_at", "title", "duration_s"]
    )
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def _grupos(df, hashes):
    return dict(zip(df["video_id"], grupo_de_duplicatas(df, hashes), strict=True))


# --- dHash ---


def test_dhash_junta_a_mesma_arte_e_separa_artes_diferentes():
    a = dhash(thumbnail(1))
    assert distancia(a, dhash(thumbnail(1, brilho=15))) <= 10  # a mesma arte, outra exposição
    assert distancia(a, dhash(thumbnail(2))) > 20  # outra arte (controle de 14/10: mediana 32)


# --- a regra ---


def test_reupload_do_mesmo_criativo_ganha_o_id_da_primeira_copia():
    df = _posts(
        ("bbbbbbbbbbb", CANAL_ID, "2023-10-24T12:00:00Z", "  natura  essencial ATO ", 16.0),
        ("kkkkkkkkkkk", CANAL_ID, "2023-09-04T12:00:00Z", "Natura Essencial Ato", 16.0),
        ("ddddddddddd", CANAL_ID, "2023-09-05T12:00:00Z", "Natura Essencial Ato", 30.0),
    )
    h = dhash(thumbnail(1))
    hashes = {"bbbbbbbbbbb": h, "kkkkkkkkkkk": h, "ddddddddddd": h}
    assert _grupos(df, hashes) == {
        "kkkkkkkkkkk": "kkkkkkkkkkk",  # a primeira cópia publicada
        "bbbbbbbbbbb": "kkkkkkkkkkk",  # mesmo título com outra caixa e espaços
        "ddddddddddd": "ddddddddddd",  # outra duração: outro vídeo
    }


def test_titulo_generico_com_outra_thumbnail_nao_e_duplicata():
    """O caso que derrubou a regra antiga: "🧡" com a mesma duração e artes diferentes. Um post
    junta-se à cópia mais antiga com a mesma arte, não só à primeira do título."""
    df = _posts(
        ("aaaaaaaaaaa", CANAL_ID, "2025-01-01T12:00:00Z", "🧡", 15.0),
        ("bbbbbbbbbbb", CANAL_ID, "2025-02-01T12:00:00Z", "🧡", 15.0),
        ("ccccccccccc", CANAL_ID, "2025-03-01T12:00:00Z", "🧡", 15.0),
    )
    hashes = {
        "aaaaaaaaaaa": dhash(thumbnail(1)),
        "bbbbbbbbbbb": dhash(thumbnail(2)),
        "ccccccccccc": dhash(thumbnail(2, brilho=15)),
    }
    assert _grupos(df, hashes) == {
        "aaaaaaaaaaa": "aaaaaaaaaaa",
        "bbbbbbbbbbb": "bbbbbbbbbbb",
        "ccccccccccc": "bbbbbbbbbbb",
    }


def test_mesmo_criativo_em_outro_canal_e_duplicata():
    """A marca e a influenciadora postam o mesmo vídeo (aula 2.2, família 5; decisão de 17/10):
    fica a primeira cópia publicada. Com outra arte, são vídeos diferentes."""
    df = _posts(
        ("aaaaaaaaaaa", CANAL_ID, "2025-01-01T12:00:00Z", "Natura Essencial Ato", 16.0),
        ("bbbbbbbbbbb", OUTRO, "2025-01-02T12:00:00Z", "Natura Essencial Ato", 16.0),
        ("ccccccccccc", OUTRO, "2025-01-03T12:00:00Z", "Natura Essencial Ato", 16.0),
    )
    hashes = {
        "aaaaaaaaaaa": dhash(thumbnail(1)),
        "bbbbbbbbbbb": dhash(thumbnail(1)),
        "ccccccccccc": dhash(thumbnail(2)),
    }
    assert _grupos(df, hashes) == {
        "aaaaaaaaaaa": "aaaaaaaaaaa",
        "bbbbbbbbbbb": "aaaaaaaaaaa",
        "ccccccccccc": "ccccccccccc",
    }


def test_sem_thumbnail_fica_pela_regra_de_titulo_e_duracao_e_avisa():
    """Conservador: sem a imagem, a cópia sai das linhas do modelo, como na regra antiga."""
    df = _posts(
        ("aaaaaaaaaaa", CANAL_ID, "2025-01-01T12:00:00Z", "CH 01", 20.0),
        ("bbbbbbbbbbb", CANAL_ID, "2025-01-02T12:00:00Z", "CH 01", 20.0),
    )
    with pytest.warns(UserWarning, match="sem thumbnail"):
        grupos = _grupos(df, {"aaaaaaaaaaa": dhash(thumbnail(1)), "bbbbbbbbbbb": None})
    assert grupos["bbbbbbbbbbb"] == "aaaaaaaaaaa"


def test_sem_hash_na_copia_mais_antiga_nao_junta_as_outras_artes():
    """Só o post sem imagem cai na regra de título e duração: os outros continuam comparando
    as artes entre si (revisão de 17/10: antes, a chave inteira virava um grupo só)."""
    df = _posts(
        ("aaaaaaaaaaa", CANAL_ID, "2025-01-01T12:00:00Z", "CH 01", 6.0),
        ("bbbbbbbbbbb", CANAL_ID, "2025-02-01T12:00:00Z", "CH 01", 6.0),
        ("ccccccccccc", CANAL_ID, "2025-03-01T12:00:00Z", "CH 01", 6.0),
        ("ddddddddddd", CANAL_ID, "2025-04-01T12:00:00Z", "CH 01", 6.0),
    )
    hashes = {
        "aaaaaaaaaaa": None,
        "bbbbbbbbbbb": dhash(thumbnail(1)),
        "ccccccccccc": dhash(thumbnail(2)),
        "ddddddddddd": dhash(thumbnail(1, brilho=15)),
    }
    with pytest.warns(UserWarning, match="1 post"):
        grupos = _grupos(df, hashes)
    assert grupos == {
        "aaaaaaaaaaa": "aaaaaaaaaaa",
        "bbbbbbbbbbb": "bbbbbbbbbbb",
        "ccccccccccc": "ccccccccccc",
        "ddddddddddd": "bbbbbbbbbbb",
    }


def test_limiar_vem_do_config_e_muda_a_regra():
    assert ler_limiar() == 10  # decisão #1, configs/duplicatas.yaml
    df = _posts(
        ("aaaaaaaaaaa", CANAL_ID, "2025-01-01T12:00:00Z", "🧡", 15.0),
        ("bbbbbbbbbbb", CANAL_ID, "2025-02-01T12:00:00Z", "🧡", 15.0),
    )
    hashes = {"aaaaaaaaaaa": dhash(thumbnail(1)), "bbbbbbbbbbb": dhash(thumbnail(2))}
    assert set(grupo_de_duplicatas(df, hashes)) == {"aaaaaaaaaaa", "bbbbbbbbbbb"}
    assert set(grupo_de_duplicatas(df, hashes, limiar=64)) == {"aaaaaaaaaaa"}  # tudo é cópia


def test_post_sem_copia_nao_precisa_de_thumbnail():
    df = _posts(("aaaaaaaaaaa", CANAL_ID, "2025-01-01T12:00:00Z", "Único", 20.0))
    assert _grupos(df, {}) == {"aaaaaaaaaaa": "aaaaaaaaaaa"}


def test_linhas_do_modelo_ficam_so_com_a_primeira_copia():
    base = pd.DataFrame(
        {
            "post_id": ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"],
            "duplicate_group": ["aaaaaaaaaaa", "aaaaaaaaaaa", "ccccccccccc"],
        }
    )
    assert list(linhas_do_modelo(base)["post_id"]) == ["aaaaaaaaaaa", "ccccccccccc"]


# --- cache das thumbnails ---


class Baixador:
    """baixar falso: devolve uma arte por vídeo e conta as chamadas; `falham` levanta erro."""

    def __init__(self, falham=()):
        self.falham, self.pedidos = set(falham), []

    def __call__(self, video_id):
        self.pedidos.append(video_id)
        if video_id in self.falham:
            raise OSError("404 Not Found")
        return thumbnail(len(self.pedidos))


def test_cache_guarda_os_hashes_e_so_baixa_o_que_falta(tmp_path):
    cache = tmp_path / "thumb_hash.parquet"
    baixar = Baixador()
    primeira = hashes_das_thumbnails(["aaaaaaaaaaa", "bbbbbbbbbbb"], cache, baixar)
    segunda = hashes_das_thumbnails(["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"], cache, baixar)
    assert sorted(baixar.pedidos) == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    assert segunda["aaaaaaaaaaa"] == primeira["aaaaaaaaaaa"]
    assert len(pd.read_parquet(cache)) == 3


def test_thumbnail_que_falha_avisa_fica_fora_do_cache_e_e_tentada_de_novo(tmp_path):
    cache = tmp_path / "thumb_hash.parquet"
    baixar = Baixador(falham={"bbbbbbbbbbb"})
    with pytest.warns(UserWarning, match="1 thumbnail"):
        hashes = hashes_das_thumbnails(["aaaaaaaaaaa", "bbbbbbbbbbb"], cache, baixar)
    assert hashes["bbbbbbbbbbb"] is None
    assert set(pd.read_parquet(cache)["video_id"]) == {"aaaaaaaaaaa"}
    with pytest.warns(UserWarning):
        hashes_das_thumbnails(["aaaaaaaaaaa", "bbbbbbbbbbb"], cache, baixar)
    assert baixar.pedidos.count("bbbbbbbbbbb") == 2


def test_cache_aceita_caminho_em_texto(tmp_path):
    cache = str(tmp_path / "thumb_hash.parquet")
    hashes_das_thumbnails(["aaaaaaaaaaa"], cache, Baixador())
    assert len(pd.read_parquet(cache)) == 1


def test_escrita_interrompida_nao_corrompe_o_cache(tmp_path, monkeypatch):
    """O cache é gravado num .tmp e trocado de uma vez (os.replace, aula 4.2)."""
    cache = tmp_path / "thumb_hash.parquet"
    hashes_das_thumbnails(["aaaaaaaaaaa"], cache, Baixador())

    def morre_no_meio(self, caminho, **kwargs):
        Path(caminho).write_bytes(b"meio parquet")
        raise RuntimeError("o Mac dormiu no meio da escrita")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", morre_no_meio)
    with pytest.raises(RuntimeError, match="dormiu"):
        hashes_das_thumbnails(["aaaaaaaaaaa", "bbbbbbbbbbb"], cache, Baixador())
    monkeypatch.undo()
    assert list(pd.read_parquet(cache)["video_id"]) == ["aaaaaaaaaaa"]


def test_aviso_de_falha_nunca_mostra_a_chave(tmp_path):
    def baixar(video_id):
        raise OSError("falhou em https://exemplo/v?key=SEGREDO123&id=1")

    with pytest.warns(UserWarning) as avisos:
        hashes_das_thumbnails(["aaaaaaaaaaa"], tmp_path / "c.parquet", baixar)
    assert "SEGREDO123" not in str(avisos[0].message)


# --- download (aula 4.2: retry só em erro transitório) ---


def _cliente(*status):
    """httpx com um servidor falso que responde `status` em ordem; conta os pedidos."""
    fila, pedidos = list(status), []

    def responder(pedido):
        pedidos.append(str(pedido.url))
        codigo = fila.pop(0)
        return httpx.Response(codigo, content=thumbnail(1) if codigo == 200 else b"")

    return httpx.Client(transport=httpx.MockTransport(responder)), pedidos


@pytest.fixture
def sem_espera(monkeypatch):
    monkeypatch.setattr(duplicatas._baixar.retry, "wait", wait_none())


def test_download_refaz_erro_transitorio(sem_espera):
    cliente, pedidos = _cliente(503, 429, 200)
    assert dhash(baixar_thumbnail("aaaaaaaaaaa", cliente)) == dhash(thumbnail(1))
    assert len(pedidos) == 3
    assert pedidos[0] == "https://i.ytimg.com/vi/aaaaaaaaaaa/hqdefault.jpg"


def test_download_nao_insiste_no_404(sem_espera):
    cliente, pedidos = _cliente(404, 200)
    with pytest.raises(httpx.HTTPStatusError):
        baixar_thumbnail("aaaaaaaaaaa", cliente)
    assert len(pedidos) == 1
