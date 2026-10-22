"""O grafo fixo da previsão e o agente das perguntas abertas (aula 6.2; A6; #15, #17).

Duas estruturas, cada uma no seu lugar (6.2):

- **O grafo fixo** (`criar_grafo`): `extrair` → `contexto` → `prever` → `explicar` → `responder`,
  sempre nessa ordem (6.2, EC1). Cada nó chama um passo de `previsao` (`passo_extrair`,
  `passo_contexto`, `passo_prever`, `passo_explicar`), os mesmos que `prever_criativo` chama em
  sequência: o número do grafo é o da função por construção (princípio 1). O LLM só pode entrar
  no `explicar`, e só no texto: a A4 (`previa.explicador.explicar`) devolve o template sem LLM ou
  quando o criativo não foi publicado (#15). O estado guarda JSON e é validado na leitura (EC3);
  o `InMemorySaver` exige um `thread_id` por criativo (EC4), e `stream_mode="updates"` mostra o
  que cada nó fez. Os objetos dos passos (DataFrames, vetores da capa) ficam numa memória da
  execução, fora do estado, que guarda só o resumo em JSON de cada passo.
- **O agente** (`criar_agente`): `create_agent` (EC5) com duas tools no nível do pedido (EC2):
  `prever_criativo`, que roda o grafo, e `comparar_criativos`, a função pura. O LLM decide quais
  criativos prever, nunca como o número é calculado. As descrições, as instruções e as mensagens
  de erro são as do servidor MCP (`configs/mcp.yaml`): as duas portas falam a mesma língua.

Por que não há aresta condicional, como na 6.2: o modelo de uso não lê o vídeo nem embute a
legenda (a legenda não entra em nenhuma feature); a única leitura é a da capa, no nó `extrair`
(`passo_extrair`). Ele registra na `trilha` o que foi lido, ou que a leitura foi pulada (criativo
só texto, sem capa), e o `prever` marca `extracao_falhou` (6.2, "No projeto Galaxies").

CLI da demo: `uv run python -m previa.agente PEDIDO.json [--publicado]`. Quem roda junta torch
(SigLIP) e LightGBM, então `um_runtime()` vem antes dos imports pesados (CLAUDE.md, OpenMP).
"""

import argparse
import json
import operator
import sys
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from previa.openmp import um_runtime

if __name__ == "__main__":
    um_runtime()

import yaml  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ToolCallLimitMiddleware  # noqa: E402
from langchain_core.tools import StructuredTool, ToolException  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.config import get_stream_writer  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from previa import previsao  # noqa: E402
from previa.caminhos import CONFIGS  # noqa: E402
from previa.coletor import sem_chave  # noqa: E402
from previa.schemas import PredictionRequest, PredictionResponse  # noqa: E402
from previa.servidor_mcp import TEXTOS, traduzir  # noqa: E402

CFG = yaml.safe_load((CONFIGS / "agente.yaml").read_text(encoding="utf-8"))
NOS = ("extrair", "contexto", "prever", "explicar", "responder")  # a ordem fixa (6.2, EC1)


def _limpo(texto: str) -> str:
    return " ".join(texto.split())


SISTEMA = _limpo(TEXTOS["servidor"]["instrucoes"])  # o mesmo texto que o MCP anuncia


# ---------------------------------------------------------------- o grafo fixo


class Estado(TypedDict, total=False):
    pedido: dict  # tudo em JSON: o checkpointer serializa sem surpresas (6.2, EC3)
    resposta: dict  # a PredictionResponse da função pura, em JSON
    narrativa: dict  # {"texto", "fonte", "problemas"} da A4
    saida: dict  # o que o grafo entrega: a resposta da tool, a narrativa e a trilha
    contexto: dict  # o resumo do passo de contexto (degrau da âncora, posts maduros)
    previsao: dict  # o resumo do passo de previsão (o número e o intervalo)
    trilha: Annotated[list[str], operator.add]  # reducer: cada nó ACRESCENTA o seu passo


def _o_que_ler(req: PredictionRequest) -> str:
    """O que a função pura vai ler no passo 4 (a mesma regra dela: a capa da marca, senão a imagem
    do criativo). Só registra: quem lê é a função, e a falta vira `extracao_falhou` lá."""
    if req.cover:
        return "capa da marca"
    if req.creative.tipo == "imagem":
        return "imagem do criativo (vira a capa)"
    return (
        f"pulado (criativo {req.creative.tipo} sem capa: nada para ler; "
        "a resposta sai com extracao_falhou)"
    )


def criar_grafo(
    *,
    bundle=None,
    encoder=None,
    explicar=None,
    llm=None,
    publicado: bool = False,
    checkpointer=None,
):
    """O fluxo da previsão como `StateGraph` fixo (6.2).

    `bundle` e `encoder` vão direto para `prever_criativo` (None: o pacote de `GALAXIES_ARTEFATOS`
    e o SigLIP real, carregados uma vez). `explicar` é a função da A4; o padrão é importado aqui
    dentro (import tardio), e os testes injetam uma falsa. `llm` e `publicado` só chegam ao
    `explicar`: sem `publicado`, a explicação é o template (#15). Sem `checkpointer`, um
    `InMemorySaver`: toda execução precisa de `thread_id` (EC4), um por criativo, porque a `trilha`
    de um mesmo thread acumula.
    """
    if explicar is None:
        from previa.explicador import explicar  # import tardio: a A4 só entra quando é usada

    memoria: dict[str, dict] = {}  # os objetos de cada execução, por thread_id; o estado é JSON

    def pedido(s) -> PredictionRequest:
        return PredictionRequest.model_validate(s["pedido"])  # valida na fronteira (EC3)

    def da_execucao(config) -> dict:
        return memoria.setdefault(
            config["configurable"]["thread_id"], {"bundle": bundle or previsao.carregar()}
        )

    def extrair(s, config):
        m, req = da_execucao(config), pedido(s)
        m["ext"] = previsao.passo_extrair(req, m["bundle"], encoder)
        passo = f"extrair: {_o_que_ler(req)}"
        get_stream_writer()({"aviso": passo})  # stream_mode="custom", para a interface
        return {"trilha": [passo]}

    def contexto(s, config):
        m = da_execucao(config)
        ctx = previsao.passo_contexto(pedido(s), m["bundle"])
        m["ctx"] = ctx
        resumo = {"degrau": str(ctx["degrau"]), "posts_maduros": int(ctx["n_maduros"])}
        passo = (
            f"contexto: âncora pelo degrau {resumo['degrau']}, "
            f"{resumo['posts_maduros']} posts maduros"
        )
        return {"contexto": resumo, "trilha": [passo]}

    def prever(s, config):
        m = da_execucao(config)
        prev = previsao.passo_prever(pedido(s), m["bundle"], m["ctx"], m["ext"])  # o ÚNICO número
        m["prev"] = prev
        resumo = {"predicted_metric": prev["pred"], "interval_low": prev["lo_c"],
                  "interval_high": prev["hi_c"]}  # fmt: skip
        return {
            "previsao": resumo,
            "trilha": [f"prever: modelo {previsao.model_version(m['bundle'])['rotulo']}"],
        }

    def explicar_no(s, config):  # o único nó em que um LLM pode entrar, e só no texto
        m = memoria.pop(config["configurable"]["thread_id"])  # a execução acaba aqui
        r = previsao.passo_explicar(pedido(s), m["bundle"], m["ctx"], m["ext"], m["prev"])
        # um objeto novo, validado do JSON: o explicador não tem como mexer na resposta
        n = explicar(PredictionResponse.model_validate(r.model_dump(mode="json")), llm=llm,
                     publicado=publicado)  # fmt: skip
        narrativa = {
            "texto": str(n["texto"]),
            "fonte": str(n["fonte"]),
            "problemas": [str(p) for p in n.get("problemas", [])],
        }
        passo = f"explicar: fonte {narrativa['fonte']}"
        if narrativa["problemas"]:
            passo += f", {len(narrativa['problemas'])} problema(s) na validação"
        return {"resposta": r.model_dump(mode="json"), "narrativa": narrativa, "trilha": [passo]}

    def responder(s):
        passo = "responder: resposta da tool, narrativa e trilha"
        saida = {
            "resposta": s["resposta"],
            "narrativa": s["narrativa"],
            "trilha": [*s["trilha"], passo],
        }
        return {"saida": saida, "trilha": [passo]}

    g = StateGraph(Estado)
    for nome, no in zip(NOS, (extrair, contexto, prever, explicar_no, responder), strict=True):
        g.add_node(nome, no)
    g.add_edge(START, NOS[0])
    for a, b in zip(NOS, NOS[1:], strict=False):  # as arestas fixas, uma atrás da outra
        g.add_edge(a, b)
    g.add_edge(NOS[-1], END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(), name=CFG["grafo"]["nome"])


def rodar_grafo(grafo, pedido, *, thread_id: str) -> dict:
    """Roda o grafo num pedido (dict ou `PredictionRequest`) e devolve a `saida` (JSON)."""
    req = (
        pedido
        if isinstance(pedido, PredictionRequest)
        else PredictionRequest.model_validate(pedido)
    )
    config = {"configurable": {"thread_id": thread_id}}
    return grafo.invoke({"pedido": req.model_dump(mode="json")}, config)["saida"]


# ---------------------------------------------------------------- o agente


class PedidosParaComparar(BaseModel):
    """O contrato de `comparar_criativos`: a lista de pedidos, cada um no formato do contrato."""

    pedidos: list[PredictionRequest] = Field(
        description="um pedido completo por criativo, todos do mesmo canal"
    )


def _erro_legivel(e: Exception) -> str:
    """O erro vira a mensagem do `configs/mcp.yaml`, que diz o que reenviar (a mesma do MCP)."""
    return sem_chave(str(traduzir(e)))


def criar_tools(*, bundle=None, encoder=None, explicar=None, llm=None, publicado=False) -> list:
    """As duas tools do agente, no nível do pedido (6.2, EC2): o LLM não tem como passar feature."""
    grafo = criar_grafo(
        bundle=bundle, encoder=encoder, explicar=explicar, llm=llm, publicado=publicado
    )

    def prever_criativo(**pedido) -> dict:
        try:  # um thread por criativo (EC4)
            return rodar_grafo(grafo, pedido, thread_id=f"prever-{uuid.uuid4().hex}")
        except Exception as e:
            raise ToolException(_erro_legivel(e)) from e

    def comparar_criativos(pedidos: list) -> dict:
        try:
            reqs = [PredictionRequest.model_validate(p) for p in pedidos]
            r = previsao.comparar_criativos(reqs, bundle=bundle, encoder=encoder)
        except Exception as e:
            raise ToolException(_erro_legivel(e)) from e
        return {**r, "previsoes": [p.model_dump(mode="json") for p in r["previsoes"]]}

    comum = {"handle_tool_error": True, "handle_validation_error": _erro_legivel}
    return [
        StructuredTool.from_function(
            prever_criativo,
            name="prever_criativo",
            description=_limpo(TEXTOS["tools"]["prever_criativo"]),
            args_schema=PredictionRequest,
            **comum,
        ),
        StructuredTool.from_function(
            comparar_criativos,
            name="comparar_criativos",
            description=_limpo(TEXTOS["tools"]["comparar_criativos"]),
            args_schema=PedidosParaComparar,
            **comum,
        ),
    ]


def criar_agente(
    llm,
    *,
    bundle=None,
    encoder=None,
    explicar=None,
    llm_explicacao=None,
    publicado: bool = False,
    checkpointer=None,
):
    """O agente ReAct das perguntas abertas (6.2): `create_agent` com as duas tools.

    `llm`: um `BaseChatModel` ou "provedor:modelo". `llm_explicacao` e `publicado` vão ao nó
    `explicar` do grafo (sem `publicado`, template, #15). Com o checkpointer, um `thread_id` por
    conversa. O `ToolCallLimitMiddleware` corta um LLM em laço (`configs/agente.yaml`).
    """
    tools = criar_tools(
        bundle=bundle, encoder=encoder, explicar=explicar, llm=llm_explicacao, publicado=publicado
    )
    limite = ToolCallLimitMiddleware(run_limit=CFG["agente"]["limite_de_chamadas"])
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SISTEMA,
        middleware=[limite],
        checkpointer=checkpointer or InMemorySaver(),
        name=CFG["agente"]["nome"],
    )


# ---------------------------------------------------------------- CLI da demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Roda o grafo fixo num pedido e imprime a saída (JSON)."
    )
    p.add_argument("pedido", type=Path, help="arquivo JSON no formato do PredictionRequest")
    p.add_argument(
        "--publicado",
        action="store_true",
        help="criativo já publicado: a explicação pode passar pelo LLM (sem isto, template, #15)",
    )
    args = p.parse_args(argv)
    try:
        pedido = PredictionRequest.model_validate_json(args.pedido.read_text(encoding="utf-8"))
        llm = None
        if args.publicado:
            from previa.explicador import llm_padrao

            llm = llm_padrao()
        previsao.aquecer()  # pacote e encoder na memória antes do 1º nó (6.1)
        grafo = criar_grafo(llm=llm, publicado=args.publicado)
        config = {"configurable": {"thread_id": args.pedido.stem}}
        entrada = {"pedido": pedido.model_dump(mode="json")}
        for modo, dado in grafo.stream(entrada, config, stream_mode=["updates", "custom"]):
            if modo == "custom":
                print(f"aviso: {dado['aviso']}", file=sys.stderr)
            else:  # a inspeção nó a nó vai para o stderr; o stdout fica só com o JSON
                for no, delta in dado.items():
                    print(f"{no:<10} -> {[k for k in delta if k != 'trilha']}", file=sys.stderr)
        saida = grafo.get_state(config).values["saida"]
    except Exception as e:
        print(f"erro: {_erro_legivel(e)}", file=sys.stderr)
        raise SystemExit(1) from e
    print(json.dumps(saida, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
