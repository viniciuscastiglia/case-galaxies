"""Testes da rubrica do VLM (decisão #13 de 18/10; aula 4.7), sem chave e sem rede.

O `httpx.MockTransport` faz o papel da API do Gemini, como na aula: o SDK monta o pedido de
verdade, e o servidor falso responde.
"""

import json

import httpx
import numpy as np
import pandas as pd
import pytest
from google import genai
from google.genai import errors, types
from pydantic import ValidationError
from tenacity import wait_none

from previa.vlm import Rubrica, extrair_rubrica, processar

URL = "https://www.youtube.com/watch?v=XXXXXXXXXXX"
OK = {
    "forca_hook_0a10": 8,
    "energia_ritmo_0a10": 7,
    "tem_rosto": True,
    "close_up_produto": True,
    "before_after_ou_transformacao": True,
    "texto_na_tela": False,
}


def _resposta(corpo):
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": json.dumps(corpo)}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 3100,
                "candidatesTokenCount": 60,
                "totalTokenCount": 3160,
            },
        },
    )


def _erro(codigo, status):
    return httpx.Response(
        codigo, json={"error": {"code": codigo, "message": "falso", "status": status}}
    )


def _cliente(respostas):
    """Cliente do Gemini que recebe as respostas da lista, na ordem; guarda os pedidos."""
    pedidos = []

    def servidor(req: httpx.Request) -> httpx.Response:
        pedidos.append(json.loads(req.content))
        return respostas[len(pedidos) - 1]

    transporte = httpx.Client(transport=httpx.MockTransport(servidor))
    cliente = genai.Client(api_key="falsa", http_options=types.HttpOptions(httpx_client=transporte))
    return cliente, pedidos


def _extrair(cliente, setor="beleza"):
    sem_espera = extrair_rubrica.retry_with(wait=wait_none())  # o backoff real vai até 2 min
    return sem_espera(cliente, URL, setor, "gemini-3.5-flash-lite")


def test_rubrica_tem_6_campos_com_ancoras():
    props = Rubrica.model_json_schema()["properties"]
    assert set(props) == set(OK)
    for campo in ("forca_hook_0a10", "energia_ritmo_0a10"):
        assert props[campo]["minimum"] == 0 and props[campo]["maximum"] == 10
        assert all(a in props[campo]["description"] for a in ("0 =", "5 =", "10 ="))


def test_pedido_vai_com_url_fps_e_resolucao_baixa():
    cliente, pedidos = _cliente([_resposta(OK)])
    _extrair(cliente, "automotivo")
    parte = pedidos[0]["contents"][0]["parts"][0]
    assert parte["fileData"]["file_uri"] == URL  # a chave sai assim no corpo (saída da 4.7)
    assert parte["videoMetadata"]["fps"] == 1
    config = pedidos[0]["generationConfig"]
    assert config["temperature"] == 0
    assert config["responseMimeType"] == "application/json"
    assert config["mediaResolution"] == "MEDIA_RESOLUTION_LOW"
    assert "AUTOMOTIVO" in pedidos[0]["contents"][0]["parts"][1]["text"]


def test_resposta_valida_vira_rubrica():
    cliente, _ = _cliente([_resposta(OK)])
    rubrica, tokens = _extrair(cliente)
    assert rubrica == Rubrica(**OK)
    assert tokens == 3160


def test_nota_fora_da_escala_levanta():
    cliente, pedidos = _cliente([_resposta({**OK, "forca_hook_0a10": 12})])
    with pytest.raises(ValidationError):
        _extrair(cliente)
    assert len(pedidos) == 1  # erro de validação não se repete (erro comum 4)


def test_400_nao_repete():
    cliente, pedidos = _cliente([_erro(400, "INVALID_ARGUMENT")] * 6)
    with pytest.raises(errors.APIError):
        _extrair(cliente)
    assert len(pedidos) == 1


def test_timeout_repete_e_depois_passa():
    def servidor(req):
        servidor.n += 1
        if servidor.n == 1:
            raise httpx.ReadTimeout("conexão parada", request=req)
        return _resposta(OK)

    servidor.n = 0
    transporte = httpx.Client(transport=httpx.MockTransport(servidor))
    cliente = genai.Client(api_key="falsa", http_options=types.HttpOptions(httpx_client=transporte))
    rubrica, _ = _extrair(cliente)
    assert rubrica.forca_hook_0a10 == 8 and servidor.n == 2


def test_429_repete_e_depois_passa():
    cliente, pedidos = _cliente(
        [_erro(429, "RESOURCE_EXHAUSTED"), _erro(503, "UNAVAILABLE"), _resposta(OK)]
    )
    rubrica, _ = _extrair(cliente)
    assert rubrica.forca_hook_0a10 == 8
    assert len(pedidos) == 3


# --- o laço cacheado (cota diária, erros e ordem) ---

HOJE = "2026-10-18"  # dia do Pacífico


def _amostra(*linhas):
    """(post_id, ordem, duration_s) -> amostra mínima, toda de beleza."""
    return pd.DataFrame(
        [(p, "beleza", o, d) for p, o, d in linhas],
        columns=["post_id", "sector", "ordem", "duration_s"],
    )


def _extrator(falhas=None):
    """extrair falso: guarda a ordem das chamadas; `falhas[post_id]` é a exceção a levantar."""
    chamadas = []

    def extrair(url, setor):
        post_id = url.rsplit("=", 1)[1]
        chamadas.append(post_id)
        if falhas and post_id in falhas:
            raise falhas[post_id]
        return Rubrica(**OK), 3000

    return extrair, chamadas


def _rodar(amostra, extrair, pasta, limite_s=28_800, n=None):
    return processar(
        amostra,
        extrair,
        pasta,
        versao=1,
        modelo="falso",
        limite_s=limite_s,
        n=n,
        hoje=lambda: HOJE,
        dormir=lambda s: None,
    )


def _linhas(caminho):
    return [json.loads(x) for x in caminho.read_text().splitlines()] if caminho.exists() else []


def test_pula_o_que_ja_esta_no_cache(tmp_path):
    amostra = _amostra(("aaaaaaaaaaa", 0, 30), ("bbbbbbbbbbb", 1, 30))
    extrair, chamadas = _extrator()
    _rodar(amostra, extrair, tmp_path)
    resumo = _rodar(amostra, extrair, tmp_path)  # 2ª rodada: nada novo
    assert chamadas == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    assert resumo["pulados"] == 2 and resumo["ok"] == 0
    assert [x["post_id"] for x in _linhas(tmp_path / "rubrica_v1.jsonl")] == chamadas


def test_erro_vai_para_o_arquivo_e_segue(tmp_path):
    invalida = None
    try:
        Rubrica.model_validate({**OK, "forca_hook_0a10": 12})
    except ValidationError as e:
        invalida = e
    privado = errors.ClientError(403, {"error": {"code": 403, "message": "privado"}})
    amostra = _amostra(("aaaaaaaaaaa", 0, 30), ("bbbbbbbbbbb", 1, 30), ("ccccccccccc", 2, 30))
    extrair, _ = _extrator({"aaaaaaaaaaa": invalida, "bbbbbbbbbbb": privado})
    resumo = _rodar(amostra, extrair, tmp_path)
    assert resumo["ok"] == 1 and resumo["erros"] == 2
    erros = _linhas(tmp_path / "erros_v1.jsonl")
    assert [x["post_id"] for x in erros] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    assert "ValidationError" in erros[0]["erro"] and "403" in erros[1]["erro"]
    # na rodada seguinte, os erros são tentados de novo (só o cache de sucesso é pulado)
    extrair2, chamadas2 = _extrator()
    _rodar(amostra, extrair2, tmp_path)
    assert chamadas2 == ["aaaaaaaaaaa", "bbbbbbbbbbb"]


def test_timeout_persistente_vira_erro_do_video_e_segue(tmp_path):
    parado = httpx.ReadTimeout("conexão parada")
    amostra = _amostra(("aaaaaaaaaaa", 0, 30), ("bbbbbbbbbbb", 1, 30))
    extrair, chamadas = _extrator({"aaaaaaaaaaa": parado})
    resumo = _rodar(amostra, extrair, tmp_path)
    assert chamadas == ["aaaaaaaaaaa", "bbbbbbbbbbb"] and resumo["ok"] == 1
    erro = _linhas(tmp_path / "erros_v1.jsonl")[0]
    assert "Timeout" in erro["erro"] and erro["_segundos"] == 0  # não gastou a cota do dia


def test_tres_falhas_de_rede_seguidas_param_a_rodada(tmp_path):
    rede = httpx.ConnectError("sem rede")
    ids = [c * 11 for c in "abcde"]
    amostra = _amostra(*[(v, k, 30) for k, v in enumerate(ids)])
    extrair, chamadas = _extrator(dict.fromkeys(ids, rede))
    resumo = _rodar(amostra, extrair, tmp_path)
    assert chamadas == ids[:3] and resumo["parou_por"] == "rede"


def test_para_no_limite_de_segundos_do_dia(tmp_path):
    ontem = {"post_id": "zzzzzzzzzzz", **OK, "_dia_pt": "2026-10-17", "_segundos": 90}
    hoje = {"post_id": "yyyyyyyyyyy", **OK, "_dia_pt": HOJE, "_segundos": 40}
    (tmp_path / "rubrica_v1.jsonl").write_text(json.dumps(ontem) + "\n" + json.dumps(hoje) + "\n")
    amostra = _amostra(("aaaaaaaaaaa", 0, 50), ("bbbbbbbbbbb", 1, 50))
    extrair, chamadas = _extrator()
    resumo = _rodar(amostra, extrair, tmp_path, limite_s=100)  # 40 de hoje + 50 cabem; +50 não
    assert chamadas == ["aaaaaaaaaaa"]
    assert resumo["parou_por"] == "limite_do_dia" and resumo["segundos_hoje"] == 90


def test_para_quando_a_cota_acaba(tmp_path):
    cota = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
    amostra = _amostra(("aaaaaaaaaaa", 0, 30), ("bbbbbbbbbbb", 1, 30), ("ccccccccccc", 2, 30))
    extrair, chamadas = _extrator({"bbbbbbbbbbb": cota})
    resumo = _rodar(amostra, extrair, tmp_path)
    assert chamadas == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    assert resumo["parou_por"] == "cota" and resumo["ok"] == 1
    assert _linhas(tmp_path / "erros_v1.jsonl") == []  # cota não é erro do vídeo


def test_processa_na_ordem_sorteada_e_respeita_n(tmp_path):
    amostra = _amostra(("aaaaaaaaaaa", 2, 30), ("bbbbbbbbbbb", 0, 30), ("ccccccccccc", 1, 30))
    extrair, chamadas = _extrator()
    resumo = _rodar(amostra, extrair, tmp_path, n=2)
    assert chamadas == ["bbbbbbbbbbb", "ccccccccccc"]
    assert resumo["parou_por"] == "limite_n"
    linha = _linhas(tmp_path / "rubrica_v1.jsonl")[0]
    assert linha["_dia_pt"] == HOJE and linha["_segundos"] == 30 and linha["_tokens"] == 3000


# --- teste-reteste (4.7): a simulação da aula, com o gabarito impresso nela ---


def test_teste_reteste_reproduz_a_aula():
    from previa.vlm import teste_reteste

    rng = np.random.default_rng(7)
    n = 30
    verdade = rng.integers(2, 9, n)

    def rodada(ruido_hook, ruido_viral, erro_rosto):
        return pd.DataFrame(
            {
                "forca_hook_0a10": np.clip(
                    verdade + rng.integers(-ruido_hook, ruido_hook + 1, n), 0, 10
                ),
                "potencial_viral_0a10": np.clip(
                    6 + rng.integers(-ruido_viral, ruido_viral + 1, n), 0, 10
                ),
                "tem_rosto": (verdade > 4) ^ (rng.random(n) < erro_rosto),
            },
            index=[f"v{i:02d}" for i in range(n)],
        )

    r1, r2 = rodada(1, 2, 0.05), rodada(1, 2, 0.05)
    t = teste_reteste(r1, r2)
    assert t.loc["forca_hook_0a10"].tolist() == [0.53, 0.91, 0.92]
    assert t.loc["potencial_viral_0a10", "kappa"] == -0.18
    assert t.loc["tem_rosto", "kappa"] == 0.86
