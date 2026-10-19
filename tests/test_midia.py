"""Testes do pipeline de mídia da amostra (decisão #12; aulas 4.1, 4.2, 4.3 e 4.5), sem YouTube.

O vídeo de teste é sintético, gerado com o ffmpeg: 2 s vermelhos, um corte, 2 s azuis, com um
tom de 440 Hz. O Whisper não roda aqui (o modelo `small` tem ~500 MB).
"""

import shutil
import subprocess

import numpy as np
import pytest
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from previa import midia

TEM_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(midia, "RAIZ_CACHE", tmp_path / "cache")
    return tmp_path / "cache"


@pytest.fixture
def video(tmp_path):
    if not TEM_FFMPEG:
        pytest.skip("ffmpeg não instalado")
    destino = tmp_path / "sint.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=360x640:d=2:r=30",
            "-f", "lavfi", "-i", "color=c=blue:s=360x640:d=2:r=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(destino),
        ],
        check=True,
    )  # fmt: skip
    return destino


def test_cache_por_video_nao_repete_e_erros_ficam_fora(cache):
    chamadas = []

    @midia.cache_por_video("teste", versao=1)
    def etapa(video_id):
        chamadas.append(video_id)
        if video_id == "quebrado":
            raise ValueError("arquivo corrompido")
        return {"x": 1}

    for _ in range(2):
        df = midia.processar_lote(etapa, ["aaa", "quebrado"])
    assert chamadas == ["aaa", "quebrado", "quebrado"]  # o que deu certo vem do cache
    assert df["video_id"].tolist() == ["aaa"]
    assert (cache / "teste_v1" / "_erros" / "quebrado.txt").exists()


def test_instantes_da_aula():
    assert midia.instantes_amostragem(12.0) == [
        ("gancho_0", 0.5), ("gancho_1", 1.5), ("gancho_2", 2.5),
        ("uniforme_0", 3.9), ("uniforme_1", 5.7), ("uniforme_2", 7.5),
        ("uniforme_3", 9.3), ("uniforme_4", 11.1),
    ]  # fmt: skip


def test_aparencia_com_gabarito():
    cinza = np.full((640, 360, 3), 128, np.uint8)
    vermelho = np.zeros((640, 360, 3), np.uint8)
    vermelho[..., 0] = 220
    a, b = midia.aparencia(cinza), midia.aparencia(vermelho)
    assert a["saturacao"] == 0.0 and a["contraste"] == 0.0 and a["colorfulness"] == 0.0
    assert b["saturacao"] == 1.0 and b["colorfulness"] == pytest.approx(73.8, abs=0.1)


@pytest.mark.skipif(not TEM_FFMPEG, reason="ffmpeg não instalado")
def test_metadados_e_sinais_no_video_sintetico(video, cache):
    meta = midia.metadados_arquivo(str(video))
    assert meta["vertical"] and meta["tem_audio"] and meta["duracao_s"] == pytest.approx(4, abs=0.1)
    s = midia.sinais_baixo_nivel("sint", str(video))
    assert s["n_cortes"] == 1 and s["s_ate_primeiro_corte"] == pytest.approx(2.0, abs=0.1)
    assert s["saturacao_gancho"] > 0.9  # o primeiro frame é vermelho puro (BGR convertido)
    assert "cor_dominante_gancho" not in s  # string: não vira coluna


@pytest.mark.skipif(not TEM_FFMPEG, reason="ffmpeg não instalado")
def test_energia_do_audio(video, tmp_path):
    wav = midia.extrair_audio(video)
    e = midia.features_energia(str(wav))
    assert e["aud_rms_media"] > 0.05 and e["aud_rms_cv"] < 0.2  # um tom constante (1/8 da escala)


def test_baixar_video_cache_falha_permanente_e_bloqueio(tmp_path, monkeypatch):
    monkeypatch.setattr(midia, "PASTA_VIDEOS", tmp_path / "videos")
    respostas = {
        "okokokokoko": {"requested_downloads": [{"filepath": str(tmp_path / "videos/ok.mp4")}]},
        "sumiusumiuu": DownloadError("ERROR: [youtube] sumiu: Video unavailable"),
        "botbotbotbo": DownloadError("ERROR: [youtube] bot: Sign in to confirm you're not a bot."),
    }

    def extract_info_falso(self, url, download=True):
        r = respostas[url.rsplit("/", 1)[-1]]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(YoutubeDL, "extract_info", extract_info_falso)
    assert midia.baixar_video("okokokokoko").name == "ok.mp4"
    assert midia.baixar_video("sumiusumiuu") is None
    assert "sumiusumiuu" in midia.falhas_permanentes()
    with pytest.raises(midia.BloqueadoPeloYouTube):
        midia.baixar_video("botbotbotbo")


def test_extrair_e_apagar_so_apaga_quando_tudo_deu_certo(tmp_path, monkeypatch):
    monkeypatch.setattr(midia, "PASTA_VIDEOS", tmp_path / "videos")
    arquivo = tmp_path / "v.mp4"
    monkeypatch.setattr(midia, "baixar_video", lambda vid: arquivo)

    def ok(video_id, caminho):
        return {"ok": 1}

    def falha(video_id, caminho):
        raise RuntimeError("etapa quebrou")

    arquivo.write_bytes(b"x")
    assert not midia.extrair_e_apagar("v", [ok, falha]) and arquivo.exists()
    assert midia.extrair_e_apagar("v", [ok]) and not arquivo.exists()


def test_falha_repetida_apaga_o_mp4_e_registra(tmp_path, monkeypatch):
    # retenção (princípio 5): uma etapa que sempre falha não pode manter o MP4 para sempre
    monkeypatch.setattr(midia, "PASTA_VIDEOS", tmp_path / "videos")
    (tmp_path / "videos").mkdir()
    arquivo = tmp_path / "videos" / "v.mp4"
    monkeypatch.setattr(midia, "baixar_video", lambda vid: arquivo)

    def falha(video_id, caminho):
        raise StopIteration("nenhum frame lido")

    for _ in range(midia.MAX_FALHAS_ETAPA):
        arquivo.write_bytes(b"x")
        assert not midia.extrair_e_apagar("v", [falha])
    assert not arquivo.exists() and "v" in midia.falhas_permanentes()
