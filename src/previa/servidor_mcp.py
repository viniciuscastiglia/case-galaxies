"""Servidor MCP do Prévia (aula 6.5; A2, #17, #37, F18): duas tools finas sobre a função pura.

As tools só chamam `previsao.prever_criativo` e `previsao.comparar_criativos` e convertem os erros
em `ToolError` com a mensagem legível de `configs/mcp.yaml` (6.2 EC2, 6.3 EC1: no nível do pedido,
nenhuma feature é calculada aqui). Nenhum LLM no servidor (#17, 6.5 EC5): a explicação é o template
da função, e o LLM é o do host. Stdio é o padrão e o stdout é o canal JSON-RPC: nada de `print()`,
logs no stderr e `show_banner=False` (6.5 EC1). O HTTP Streamable (`--http`) só em 127.0.0.1 (EC6).

O pacote vem de `GALAXIES_ARTEFATOS` (F18, #37). Quem sobe este processo junta torch (SigLIP) e
LightGBM, então `um_runtime()` roda antes dos imports pesados (CLAUDE.md, armadilha do OpenMP).
"""

import logging
import sys

from previa.openmp import um_runtime

if __name__ == "__main__":
    um_runtime()

import yaml  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from fastmcp.exceptions import ValidationError as ErroDeEntrada  # noqa: E402
from fastmcp.server.middleware import Middleware  # noqa: E402

from previa import previsao  # noqa: E402
from previa.caminhos import CONFIGS, RAIZ  # noqa: E402
from previa.coletor import sem_chave  # noqa: E402
from previa.schemas import PredictionRequest, PredictionResponse  # noqa: E402

log = logging.getLogger("previa.servidor_mcp")
TEXTOS = yaml.safe_load((CONFIGS / "mcp.yaml").read_text(encoding="utf-8"))
_ERROS = {k: " ".join(v.split()) for k, v in TEXTOS["erros"].items()}
_SRV = TEXTOS["servidor"]
SOMENTE_LEITURA = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}


def _dependencias() -> dict:
    """Pacote e encoder da chamada: vazio em produção (a função carrega o pacote de
    `GALAXIES_ARTEFATOS` uma vez e o SigLIP uma vez); os testes injetam o pacote pequeno."""
    return {}


def traduzir(e: Exception) -> ToolError:
    """O erro de negócio ou de entrada vira a mensagem da config, que diz o que reenviar."""
    texto = str(e)
    if "sem fuso horário" in texto:
        return ToolError(_ERROS["data_sem_fuso"])
    if "canal fora da base" in texto:
        return ToolError(_ERROS["canal_desconhecido"])
    if "o vídeo é horizontal" in texto or "o limite é" in texto:
        return ToolError(_ERROS["video_invalido"])
    if isinstance(e, ToolError):
        return e
    log.warning("erro na tool: %s: %s", type(e).__name__, sem_chave(texto))
    return ToolError(_ERROS["generico"].format(detalhe=sem_chave(texto) or type(e).__name__))


class TraduzErros(Middleware):
    """A validação do Pydantic acontece antes da função da tool: sem isto, o erro de data sem
    fuso ou de vídeo horizontal chegaria ao host no formato cru do Pydantic."""

    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except ErroDeEntrada as e:
            raise traduzir(e) from e


mcp = FastMCP("previa", instructions=_SRV["instrucoes"])
mcp.add_middleware(TraduzErros())


@mcp.tool(annotations=SOMENTE_LEITURA, timeout=_SRV["timeout_prever_s"],
          description=TEXTOS["tools"]["prever_criativo"])  # fmt: skip
def prever_criativo(pedido: PredictionRequest) -> PredictionResponse:
    try:
        return previsao.prever_criativo(pedido, **_dependencias())
    except Exception as e:
        raise traduzir(e) from e


@mcp.tool(annotations=SOMENTE_LEITURA, timeout=_SRV["timeout_comparar_s"],
          description=TEXTOS["tools"]["comparar_criativos"])  # fmt: skip
def comparar_criativos(pedidos: list[PredictionRequest]) -> dict:
    try:
        return previsao.comparar_criativos(pedidos, **_dependencias())
    except Exception as e:
        raise traduzir(e) from e


@mcp.resource("model://card", mime_type="text/markdown")
def model_card() -> str:
    """O model card: o que o modelo mede, como foi medido e onde não funciona."""
    return (RAIZ / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")


@mcp.resource("schema://prediction-request", mime_type="application/json")
def esquema_entrada() -> dict:
    return PredictionRequest.model_json_schema()


@mcp.resource("schema://prediction-response", mime_type="application/json")
def esquema_saida() -> dict:
    return PredictionResponse.model_json_schema()


@mcp.prompt
def decidir_entre_criativos(marca: str) -> str:
    """Roteiro para o LLM do host comparar criativos de uma marca."""
    return TEXTOS["prompts"]["decidir_entre_criativos"].format(marca=marca)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # stdout é o canal do protocolo
    previsao.aquecer()  # pacote e encoder na memória antes da 1ª chamada (6.1)
    if "--http" in argv:
        mcp.run(transport="http", host=_SRV["host_http"], port=_SRV["porta_http"])  # em /mcp
    else:
        mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
