"""Frames, cortes e áudio da amostra multimodal (decisão #12; aulas 4.1, 4.2, 4.3 e 4.5).

Para cada Short da amostra (`data/amostra_multimodal.csv`, na ordem sorteada): baixa em 360p pelo
lado menor com o yt-dlp (4.2), roda as etapas cacheadas por `video_id` e só então apaga o MP4
(o mínimo necessário, apagar depois de extrair). As etapas:

- `etapa_metadados` (4.1): duração, proporção, rotação e trilhas, pelo ffprobe;
- `sinais_baixo_nivel` (4.3): 8 frames (3 no gancho, 5 uniformes), aparência, cortes com o
  `AdaptiveDetector`, movimento em larguras de tela por segundo e variância entre frames;
- `etapa_audio` (4.5): WAV 16 kHz mono, faster-whisper `small` com VAD e tempos por palavra,
  energia e pulso com librosa; a transcrição é guardada (erro comum 5).

O cache segue a 4.1: uma pasta por etapa e versão, um JSON por vídeo, escrita atômica, falhas em
`_erros/`. O yt-dlp roda só do IP de casa, com pausas de 5 a 10 s; "not a bot" para o lote (4.2).
Na previsão, o MP4 da marca passa pelas mesmas etapas, depois de convertido para o lado menor 360.

`uv run python -m previa.midia amostra [--limite N]` processa a amostra.
"""

import argparse
import functools
import json
import os
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from previa.amostra import ler_config
from previa.coletor import DADOS

RAIZ_CACHE = DADOS / "cache"
PASTA_VIDEOS = DADOS / "videos"
GANCHO_S = 3.0


# --- cache por vídeo (4.1) ---


def cache_por_video(etapa: str, versao: int = 1):
    """Guarda o resultado da etapa em <RAIZ_CACHE>/<etapa>_v<versao>/<video_id>.json."""

    def pasta() -> Path:
        return RAIZ_CACHE / f"{etapa}_v{versao}"

    def decorador(func):
        @functools.wraps(func)
        def wrapper(video_id: str, *args, **kwargs) -> dict:
            arquivo = pasta() / f"{video_id}.json"
            if arquivo.exists():
                return json.loads(arquivo.read_text(encoding="utf-8"))
            resultado = func(video_id, *args, **kwargs)
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            tmp = arquivo.with_suffix(".tmp")
            tmp.write_text(json.dumps(resultado, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, arquivo)  # atômico: nunca fica um JSON pela metade
            return resultado

        wrapper.pasta = pasta
        return wrapper

    return decorador


def processar_lote(func, video_ids, **kwargs) -> pd.DataFrame:
    """Roda uma etapa cacheada em vários vídeos; falhas vão para <pasta>/_erros."""
    linhas = []
    for vid in video_ids:
        try:
            linhas.append({"video_id": vid, **func(vid, **kwargs)})
        except Exception as e:  # um vídeo quebrado não derruba o lote
            erros = func.pasta() / "_erros"
            erros.mkdir(parents=True, exist_ok=True)
            (erros / f"{vid}.txt").write_text(f"{e!r}\n{traceback.format_exc()}")
    return pd.DataFrame(linhas)


def consolidar(func) -> pd.DataFrame:
    """Junta os JSON de uma etapa num parquet ao lado da pasta (o que o treino lê)."""
    linhas = [
        {"video_id": p.stem, **json.loads(p.read_text(encoding="utf-8"))}
        for p in sorted(func.pasta().glob("*.json"))
    ]
    df = pd.DataFrame(linhas)
    df.to_parquet(func.pasta().with_suffix(".parquet"), index=False)
    return df


# --- arquivo (4.1) ---


def metadados_arquivo(caminho: str) -> dict:
    """Container, trilhas e dimensões pelo ffprobe, com a rotação corrigida."""
    saida = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", caminho],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    info = json.loads(saida)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    largura, altura = int(video["width"]), int(video["height"])
    rot = next((abs(int(sd.get("rotation", 0))) for sd in video.get("side_data_list", [])), 0)
    if rot in (90, 270):
        largura, altura = altura, largura
    return {
        "duracao_s": round(float(info["format"]["duration"]), 2),
        "fps": round(float(Fraction(video["avg_frame_rate"])), 2),
        "largura": largura,
        "altura": altura,
        "proporcao": round(largura / altura, 4),
        "vertical": altura > largura,
        "tem_audio": audio is not None,
        "bitrate_kbps": round(int(info["format"].get("bit_rate", 0)) / 1000),
    }


@cache_por_video("metadados", versao=1)
def etapa_metadados(video_id: str, caminho: str) -> dict:
    return metadados_arquivo(caminho)


# --- frames, aparência, cortes e movimento (4.3) ---


def instantes_amostragem(duracao, n_gancho=3, n_resto=5, gancho_s=GANCHO_S):
    """n_gancho instantes dentro do gancho + n_resto uniformes no resto (centro das fatias)."""
    fim_gancho = min(gancho_s, duracao)
    passo_g = fim_gancho / n_gancho
    ts = [(f"gancho_{i}", round(passo_g * (i + 0.5), 2)) for i in range(n_gancho)]
    if duracao > gancho_s:
        passo_r = (duracao - gancho_s) / n_resto
        ts += [(f"uniforme_{i}", round(gancho_s + passo_r * (i + 0.5), 2)) for i in range(n_resto)]
    return ts


def extrair_frames(caminho: str, instantes, largura: int = 360) -> dict[str, np.ndarray]:
    """Um frame por instante, com seek por tempo; RGB (o OpenCV lê em BGR) e largura fixa."""
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise OSError(f"não abriu: {caminho}")
    frames = {}
    for rotulo, t in instantes:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (largura, round(h * largura / w)), interpolation=cv2.INTER_AREA)
        frames[rotulo] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return frames


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003): ~0 = cinza; >100 = muito colorido."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def aparencia(rgb: np.ndarray) -> dict:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return {
        "brilho": round(float(hsv[..., 2].mean() / 255), 3),
        "contraste": round(float(lab[..., 0].std() / 255), 3),
        "saturacao": round(float(hsv[..., 1].mean() / 255), 3),
        "colorfulness": round(colorfulness(rgb), 1),
    }


def features_de_corte(caminho: str, gancho_s: float = GANCHO_S) -> dict:
    """Cortes secos com o AdaptiveDetector (o ContentDetector(27) perde cortes sutis)."""
    from scenedetect import AdaptiveDetector, detect

    cenas = detect(caminho, AdaptiveDetector(), start_in_scene=True)
    cortes = [inicio.seconds for inicio, _ in cenas[1:]]  # 0.7: .seconds
    duracao = cenas[-1][1].seconds
    planos = [fim.seconds - inicio.seconds for inicio, fim in cenas]
    return {
        "n_cortes": len(cortes),
        "cortes_por_s": round(len(cortes) / duracao, 3),
        "duracao_media_plano_s": round(sum(planos) / len(planos), 2),
        "cortes_no_gancho": sum(c < gancho_s for c in cortes),
        "s_ate_primeiro_corte": round(cortes[0], 2) if cortes else round(duracao, 2),
    }


def _cinza(bgr: np.ndarray, largura: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    pequeno = cv2.resize(bgr, (largura, round(h * largura / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)


def movimento(caminho: str, instantes, largura: int = 180) -> dict:
    """Farnebäck entre o frame em t e o seguinte, em larguras de tela por segundo."""
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS)
    por_rotulo = {}
    for rotulo, t in instantes:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            continue
        g1, g2 = _cinza(f1, largura), _cinza(f2, largura)
        if np.abs(g1.astype(int) - g2).mean() > 40:  # o par atravessou um corte
            continue
        fluxo = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vel = np.linalg.norm(fluxo, axis=2) * fps / largura
        por_rotulo[rotulo] = (float(vel.mean()), float((vel > 0.15).mean()))
    cap.release()
    if not por_rotulo:
        return dict.fromkeys(
            ["movimento_medio", "movimento_max", "movimento_gancho", "fracao_px_em_movimento"]
        )
    medias = [m for m, _ in por_rotulo.values()]
    gancho = [m for r, (m, _) in por_rotulo.items() if r.startswith("gancho")]
    return {
        "movimento_medio": round(float(np.mean(medias)), 4),
        "movimento_max": round(float(np.max(medias)), 4),
        "movimento_gancho": round(float(np.mean(gancho)), 4) if gancho else None,
        "fracao_px_em_movimento": round(float(np.mean([f for _, f in por_rotulo.values()])), 3),
    }


def variancia_entre_frames(frames: dict[str, np.ndarray]) -> float:
    pilha = np.stack([cv2.resize(f, (90, 160)) for f in frames.values()]).astype(np.float32) / 255
    return round(float(pilha.std(axis=0).mean()), 3)


@cache_por_video("sinais_baixo_nivel", versao=1)
def sinais_baixo_nivel(video_id: str, caminho: str) -> dict:
    inst = instantes_amostragem(metadados_arquivo(caminho)["duracao_s"])
    frames = extrair_frames(caminho, inst)
    por_frame = {r: aparencia(f) for r, f in frames.items()}
    primeiro = next(iter(por_frame.values()))  # gancho_0: a primeira coisa que o usuário vê
    feats = {}
    for col in ("brilho", "contraste", "saturacao", "colorfulness"):
        feats[f"{col}_medio"] = round(float(np.mean([a[col] for a in por_frame.values()])), 3)
        feats[f"{col}_gancho"] = primeiro[col]
    return {
        **feats,
        **features_de_corte(caminho),
        **movimento(caminho, inst),
        "var_entre_frames": variancia_entre_frames(frames),
    }


# --- áudio e fala (4.5) ---


def extrair_audio(video: Path, sr: int = 16_000) -> Path:
    """A trilha de áudio em WAV mono 16 kHz (a librosa 1.0 não abre .mp4)."""
    wav = Path(video).with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr)]
        + [str(wav)],
        check=True,
    )
    return wav


@functools.lru_cache(maxsize=1)
def _whisper(nome: str):
    from faster_whisper import WhisperModel

    return WhisperModel(nome, device="cpu", compute_type="int8")


def features_fala(caminho: str, modelo: str = "small", gancho_s: float = GANCHO_S):
    import librosa

    # o áudio vai como array: o decode_audio do faster-whisper 1.2.1 chama o PyAV com um
    # parâmetro (metadata_errors) que a versão instalada não tem mais (19/10)
    audio, _ = librosa.load(caminho, sr=16_000)
    segs, info = _whisper(modelo).transcribe(audio, vad_filter=True, word_timestamps=True)
    palavras = [w for s in segs for w in (s.words or [])]
    dur, tem_fala = info.duration, len(palavras) > 0
    tempo_fala = float(sum(w.end - w.start for w in palavras))
    feats = {
        # idioma só vale com voz e confiança >= 0,7 (4.5, erro comum 1)
        "aud_idioma": info.language
        if tem_fala and info.language_probability >= 0.7
        else "sem_fala",
        "aud_n_palavras": len(palavras),
        "aud_palavras_por_s_video": len(palavras) / dur if dur else 0.0,
        "aud_palavras_por_s_fala": len(palavras) / tempo_fala if tem_fala and tempo_fala else 0.0,
        "aud_pct_com_fala": tempo_fala / dur if dur else 0.0,
        "aud_fala_no_gancho": any(w.start < gancho_s for w in palavras),
        "aud_inicio_fala_s": float(palavras[0].start) if tem_fala else None,
    }
    return feats, palavras


def features_energia(caminho: str, gancho_s: float = GANCHO_S, hop: int = 512) -> dict:
    import librosa

    y, sr = librosa.load(caminho, sr=16_000)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
    x = onset_env - onset_env.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac = ac / (ac[0] + 1e-9)
    fps = sr / hop
    lags = np.arange(int(fps * 60 / 200), int(fps * 60 / 60))
    lags = lags[lags < len(ac)]
    return {
        "aud_rms_media": float(rms.mean()),
        "aud_rms_gancho": float(rms[t < gancho_s].mean()),
        "aud_rms_cv": float(rms.std() / (rms.mean() + 1e-9)),
        "aud_bpm": float(np.atleast_1d(tempo)[0]),  # só tem sentido com pulso alto (erro comum 2)
        "aud_onsets_por_s": len(onsets) / (len(y) / sr),
        "aud_pulso": float(ac[lags].max()) if len(lags) else 0.0,
        "aud_planura": float(librosa.feature.spectral_flatness(y=y).mean()),
    }


def energia_fora_da_fala(caminho: str, palavras, pad: float = 0.2, hop: int = 512) -> float:
    import librosa

    y, sr = librosa.load(caminho, sr=16_000)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    com_fala = np.zeros(len(t), dtype=bool)
    for w in palavras:
        com_fala |= (t >= w.start - pad) & (t <= w.end + pad)
    return float(rms[~com_fala].mean()) if (~com_fala).any() else 0.0


@cache_por_video("audio", versao=1)
def etapa_audio(video_id: str, caminho: str, modelo: str = "small") -> dict:
    wav = extrair_audio(Path(caminho))
    try:
        fala, palavras = features_fala(str(wav), modelo)
        energia = features_energia(str(wav))
        fora = energia_fora_da_fala(str(wav), palavras)
        return {
            **fala,
            **energia,
            "aud_rms_fora_fala": fora,
            "aud_musica_provavel": bool(energia["aud_pulso"] > 0.5 or fora > 0.03),
            "txt_transcricao": "".join(w.word for w in palavras).strip(),  # vai para o texto
            "_whisper": modelo,
        }
    finally:
        wav.unlink(missing_ok=True)  # o WAV é intermediário


# --- download e retenção (4.2) ---

BLOQUEIO = ("not a bot", "Sign in to confirm", "HTTP Error 429")
PERMANENTE = ("Video unavailable", "Private video", "removed", "members-only")


MAX_FALHAS_ETAPA = 3  # depois disso, o MP4 é apagado e o vídeo vira falha permanente


class BloqueadoPeloYouTube(RuntimeError):
    """Pare o lote: insistir só piora o bloqueio do seu IP."""


def _arquivo_falhas() -> Path:
    return PASTA_VIDEOS / "_falhas.json"


def falhas_permanentes() -> dict:
    return json.loads(_arquivo_falhas().read_text()) if _arquivo_falhas().exists() else {}


def _registrar_falha(video_id: str, motivo: str) -> None:
    falhas = falhas_permanentes()
    falhas[video_id] = motivo[:200]
    _arquivo_falhas().write_text(json.dumps(falhas, indent=1))


def _opcoes(pausa_min=5, pausa_max=10) -> dict:
    return {
        "format": "bv*+ba/b",
        "format_sort": ["res:360"],  # lado MENOR <= 360: 360×640 num Short vertical (erro comum 2)
        "merge_output_format": "mp4",
        "outtmpl": "%(id)s.%(ext)s",
        "paths": {"home": str(PASTA_VIDEOS)},
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "sleep_interval": pausa_min,
        "max_sleep_interval": pausa_max,
        "sleep_interval_requests": 1,
        "retries": 3,
        "js_runtimes": {"deno": {}},
    }


def baixar_video(video_id: str, tentativas: int = 2, pausa_min=5, pausa_max=10) -> Path | None:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    PASTA_VIDEOS.mkdir(parents=True, exist_ok=True)
    if (PASTA_VIDEOS / f"{video_id}.mp4").exists():
        return PASTA_VIDEOS / f"{video_id}.mp4"
    url = f"https://www.youtube.com/shorts/{video_id}"
    for tentativa in range(1, tentativas + 1):
        try:
            with YoutubeDL(_opcoes(pausa_min, pausa_max)) as ydl:
                info = ydl.extract_info(url, download=True)
            return Path(info["requested_downloads"][0]["filepath"])
        except DownloadError as e:
            msg = str(e)
            if any(s in msg for s in BLOQUEIO):
                raise BloqueadoPeloYouTube(msg) from e
            if any(s in msg for s in PERMANENTE) or tentativa == tentativas:
                _registrar_falha(video_id, msg)
                return None
            time.sleep(60 * tentativa)  # transitório: espere e tente de novo
    return None


def extrair_e_apagar(video_id: str, etapas: list) -> bool:
    """Baixa, roda todas as etapas cacheadas e só então apaga o MP4 (4.2)."""
    caminho = baixar_video(video_id)
    if caminho is None:
        return False
    try:
        for etapa in etapas:
            etapa(video_id, str(caminho))
    except Exception as e:
        PASTA_VIDEOS.mkdir(parents=True, exist_ok=True)
        contagem = PASTA_VIDEOS / "_falhas_etapa.json"
        falhas = json.loads(contagem.read_text()) if contagem.exists() else {}
        falhas[video_id] = falhas.get(video_id, 0) + 1
        contagem.write_text(json.dumps(falhas, indent=1))
        if falhas[video_id] >= MAX_FALHAS_ETAPA:  # retenção (princípio 5): não guarda para sempre
            caminho.unlink(missing_ok=True)
            _registrar_falha(video_id, f"etapa falhou {falhas[video_id]} vezes: {e!r}")
            print(f"{video_id}: {e!r}; desisti e apaguei o MP4", flush=True)
        else:
            print(f"{video_id}: {e!r}; mantenho o arquivo para tentar de novo", flush=True)
        return False
    caminho.unlink()  # sobram só as features e o video_id
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m previa.midia")
    parser.add_argument("acao", choices=["amostra"])
    parser.add_argument("--limite", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = ler_config()["midia"]
    amostra = pd.read_csv(DADOS / "amostra_multimodal.csv").sort_values("ordem")

    def audio(video_id, caminho):
        return etapa_audio(video_id, caminho, modelo=cfg["whisper"])

    audio.pasta = etapa_audio.pasta
    etapas = [etapa_metadados, sinais_baixo_nivel, audio]
    pular = set(falhas_permanentes())

    def feito(v):
        return all((e.pasta() / f"{v}.json").exists() for e in etapas)

    fila = [v for v in amostra["post_id"] if v not in pular and not feito(v)]
    if args.limite:
        fila = fila[: args.limite]
    print(f"{len(fila)} Shorts na fila ({len(amostra) - len(fila)} feitos ou pulados)", flush=True)
    ok = 0
    for i, vid in enumerate(fila, start=1):
        t0 = time.perf_counter()
        try:
            ok += extrair_e_apagar(vid, etapas)
        except BloqueadoPeloYouTube as e:
            print(f"parou o lote em {vid}: {str(e)[:200]}", file=sys.stderr, flush=True)
            return 1
        print(f"{i}/{len(fila)} {vid} {time.perf_counter() - t0:.0f} s (ok: {ok})", flush=True)
    for e in etapas:
        consolidar(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
