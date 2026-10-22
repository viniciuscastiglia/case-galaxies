"""O grafo fixo e o agente (aula 6.2; A6; A7 pela invariante, aula 6.6; #15, #17).

Pacote pequeno da base sintética, encoder falso, explicador falso e LLM falso (6.2, EC6): nunca a
base real, nunca o SigLIP real, nunca o Gemini. O explicador da A4 é injetado por parâmetro, então
estes testes não dependem dele.
"""

import inspect
import json
import re
import sys
import types

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from test_capa import EncoderFalso
from test_modelo import _partes, base, cfg_av  # noqa: F401
from test_servidor_mcp import INJECAO, _pedido, capa, pacote  # noqa: F401

import previa.agente as agente
import previa.previsao as previsao
from previa.agente import NOS, criar_agente, criar_grafo, rodar_grafo
from previa.schemas import PredictionRequest


class FakeComTools(GenericFakeChatModel):
    """O LLM falso da 6.2: devolve as mensagens do roteiro, na ordem. O `create_agent` chama
    `bind_tools()`, que o `GenericFakeChatModel` não implementa; aqui ele devolve o modelo."""

    def bind_tools(self, tools, **kwargs):
        return self


class ExplicarFalso:
    """No lugar da A4: guarda os argumentos e devolve um texto fixo (a interface combinada)."""

    def __init__(self):
        self.chamadas = []

    def __call__(self, resposta, llm=None, publicado=False):
        self.chamadas.append({"resposta": resposta, "llm": llm, "publicado": publicado})
        return {"texto": resposta.explanation.resumo, "fonte": "template", "problemas": []}


def _grafo(pacote, **kw):  # noqa: F811
    kw.setdefault("explicar", ExplicarFalso())
    return criar_grafo(bundle=pacote, encoder=EncoderFalso(), **kw)


def _config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def _pura(pacote, pedido):  # noqa: F811
    req = PredictionRequest.model_validate(pedido)
    return previsao.prever_criativo(req, bundle=pacote, encoder=EncoderFalso())


def numeros(texto: str) -> list[float]:
    """Os números de um texto em formato brasileiro (6.6): "4.237" -> 4237.0; "0,884" -> 0.884."""
    achados = re.findall(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?", texto)
    return [float(n.replace(".", "").replace(",", ".")) for n in achados]


def valores(obj) -> list[float]:
    """Todos os números de uma saída de tool (6.6)."""
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in valores(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in valores(x)]
    if isinstance(obj, bool):
        return []
    return [float(obj)] if isinstance(obj, (int, float)) else []


def _br(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


# ---------------------------------------------------------------- o grafo fixo


def test_invariante_grafo_igual_funcao_igual_modelo(pacote, capa, monkeypatch):  # noqa: F811
    """A7 (princípio 1): o número do grafo é o da função pura e o `expm1` do q do modelo."""
    pedido = _pedido(capa)
    capt = {}
    original = previsao.prever_intervalo

    def espiao(*a, **k):
        capt["r"] = original(*a, **k)
        return capt["r"]

    monkeypatch.setattr(previsao, "prever_intervalo", espiao)
    saida = rodar_grafo(_grafo(pacote), pedido, thread_id="invariante")
    q_do_grafo = capt["r"][0]
    pura = _pura(pacote, pedido)
    assert saida["resposta"] == pura.model_dump(mode="json")  # duas portas, uma resposta
    assert saida["resposta"]["predicted_metric"] == pura.predicted_metric  # exatamente
    assert saida["resposta"]["predicted_metric"] == pytest.approx(np.expm1(q_do_grafo), rel=1e-12)
    assert capt["r"][0] == q_do_grafo  # a função pura, de novo, chegou ao mesmo q


def test_stream_updates_emite_os_nos_na_ordem_fixa(pacote, capa):  # noqa: F811
    grafo = _grafo(pacote)
    entrada = {"pedido": PredictionRequest.model_validate(_pedido(capa)).model_dump(mode="json")}
    nos = [no for upd in grafo.stream(entrada, _config(), stream_mode="updates") for no in upd]
    assert nos == ["extrair", "prever", "explicar", "responder"] == list(NOS)
    estado = grafo.get_state(_config()).values
    assert [t.split(":")[0] for t in estado["trilha"]] == nos
    assert estado["trilha"][0] == "extrair: capa da marca"
    # 6.2, EC3: o estado é JSON puro (o checkpointer serializa sem objetos Pydantic)
    assert json.loads(json.dumps(estado)) == estado
    assert len(list(grafo.get_state_history(_config()))) == len(nos) + 2  # antes, entrada e 4 nós


def test_criativo_so_texto_pula_a_extracao_e_sai_com_extracao_falhou(pacote):  # noqa: F811
    pedido = _pedido(None, creative={"tipo": "texto"})
    saida = rodar_grafo(_grafo(pacote), pedido, thread_id="texto")
    assert saida["trilha"][0].startswith("extrair: pulado")
    assert [t.split(":")[0] for t in saida["trilha"]] == list(NOS)
    flag = saida["resposta"]["confidence_flag"]
    assert flag["baixa_confianca"]
    assert "extracao_falhou" in {m["codigo"] for m in flag["motivos"]}


def test_imagem_do_criativo_e_registrada_como_a_capa(pacote, capa):  # noqa: F811
    criativo = {"tipo": "imagem", "arquivo": capa, "duration_s": 30}
    saida = rodar_grafo(_grafo(pacote), _pedido(None, creative=criativo), thread_id="img")
    assert saida["trilha"][0] == "extrair: imagem do criativo (vira a capa)"


def test_sem_thread_id_o_grafo_recusa(pacote, capa):  # noqa: F811
    entrada = {"pedido": PredictionRequest.model_validate(_pedido(capa)).model_dump(mode="json")}
    with pytest.raises(ValueError, match="thread_id"):
        _grafo(pacote).invoke(entrada)


def test_explicar_recebe_a_resposta_o_llm_e_o_publicado(pacote, capa):  # noqa: F811
    falso, llm = ExplicarFalso(), object()
    pedido = _pedido(capa)
    saida = rodar_grafo(
        _grafo(pacote, explicar=falso, llm=llm, publicado=True), pedido, thread_id="exp"
    )
    (chamada,) = falso.chamadas
    assert chamada["llm"] is llm and chamada["publicado"] is True
    assert chamada["resposta"].model_dump(mode="json") == saida["resposta"]
    assert saida["narrativa"] == {
        "texto": saida["resposta"]["explanation"]["resumo"],
        "fonte": "template",
        "problemas": [],
    }
    assert saida["trilha"][2] == "explicar: fonte template"


def test_explicacao_que_muda_o_numero_nao_chega_a_resposta(pacote, capa):  # noqa: F811
    """Um explicador que mexe no objeto não altera o número: a resposta sai do estado (JSON)."""

    def explicar_mexedor(resposta, llm=None, publicado=False):
        resposta.predicted_metric = 1_000_000.0  # um explicador defeituoso
        return {"texto": "1.000.000", "fonte": "llm", "problemas": []}

    pedido = _pedido(capa)
    saida = rodar_grafo(_grafo(pacote, explicar=explicar_mexedor), pedido, thread_id="mexe")
    assert saida["resposta"]["predicted_metric"] == _pura(pacote, pedido).predicted_metric


def test_sem_explicar_injetado_o_grafo_importa_o_da_a4_na_hora(pacote, capa, monkeypatch):  # noqa: F811
    """O padrão é `previa.explicador.explicar`, importado dentro da função (import tardio)."""
    falso = ExplicarFalso()
    monkeypatch.setitem(sys.modules, "previa.explicador", types.SimpleNamespace(explicar=falso))
    grafo = criar_grafo(bundle=pacote, encoder=EncoderFalso())
    rodar_grafo(grafo, _pedido(capa), thread_id="padrao")
    assert len(falso.chamadas) == 1 and falso.chamadas[0]["publicado"] is False


def test_injecao_na_legenda_nao_muda_o_numero_nem_aparece_na_saida(pacote, capa):  # noqa: F811
    """A7 (princípio 8): a legenda é dado; o número é o da legenda neutra."""
    grafo = _grafo(pacote)
    neutra = rodar_grafo(grafo, _pedido(capa), thread_id="neutra")
    suja = rodar_grafo(grafo, _pedido(capa, legenda=INJECAO), thread_id="suja")
    assert suja["resposta"]["predicted_metric"] == neutra["resposta"]["predicted_metric"]
    assert suja["resposta"] == neutra["resposta"]
    tudo = json.dumps(suja, ensure_ascii=False).lower()
    assert "ignore" not in tudo and "1.000.000" not in tudo and "1000000" not in tudo


# ---------------------------------------------------------------- o agente (create_agent)


def _agente(pacote, roteiro, **kw):  # noqa: F811
    kw.setdefault("explicar", ExplicarFalso())
    return criar_agente(
        FakeComTools(messages=iter(roteiro)), bundle=pacote, encoder=EncoderFalso(), **kw
    )


def _chamada(nome, args, id_="c1"):
    return AIMessage("", tool_calls=[{"name": nome, "args": args, "id": id_}])


def _perguntar(ag, texto="Quantas curtidas?"):
    return ag.invoke({"messages": [{"role": "user", "content": texto}]}, _config("conversa"))[
        "messages"
    ]


def test_agente_so_cita_o_numero_da_tool_que_e_o_da_funcao_pura(pacote, capa):  # noqa: F811
    """A7 (6.6, camada 3): o número na ToolMessage é o da função pura; o texto final só cita
    números que vieram da tool."""
    pedido = _pedido(capa)
    pura = _pura(pacote, pedido)
    final = (
        f"Cerca de {_br(pura.predicted_metric)} curtidas, entre {_br(pura.interval_low)} e "
        f"{_br(pura.interval_high)} (intervalo de 80%)."
    )
    msgs = _perguntar(_agente(pacote, [_chamada("prever_criativo", pedido), AIMessage(final)]))
    (tool_msg,) = [m for m in msgs if isinstance(m, ToolMessage)]
    assert tool_msg.status != "error", tool_msg.content
    saida = json.loads(tool_msg.content)
    assert saida["resposta"]["predicted_metric"] == pura.predicted_metric  # exatamente
    assert saida["resposta"] == pura.model_dump(mode="json")
    permitidos = set(valores(saida)) | {80.0}  # "80%" é o nível do intervalo
    citados = numeros(msgs[-1].content)
    assert citados and all(any(abs(c - p) < 0.5 for p in permitidos) for c in citados)


def test_agente_comparar_ranqueia_pelo_numero_da_funcao(pacote, capa):  # noqa: F811
    a = _pedido(capa)
    b = _pedido(capa, legenda="Outra legenda", planned_datetime="2026-02-11T07:00:00-03:00")
    roteiro = [_chamada("comparar_criativos", {"pedidos": [a, b]}), AIMessage("Pronto.")]
    msgs = _perguntar(_agente(pacote, roteiro), "A ou B?")
    (tool_msg,) = [m for m in msgs if isinstance(m, ToolMessage)]
    r = json.loads(tool_msg.content)
    por_nome = {x["criativo"]: x["predicted_metric"] for x in r["ranking"]}
    assert por_nome == {
        "A": _pura(pacote, a).predicted_metric,
        "B": _pura(pacote, b).predicted_metric,
    }


def test_erro_de_contrato_vira_toolmessage_com_o_texto_do_mcp(pacote, capa):  # noqa: F811
    """6.2: o erro não derruba o agente; o LLM lê o que reenviar (o mesmo texto do MCP)."""
    sem_fuso = _pedido(capa, planned_datetime="2026-02-10T19:00:00")
    roteiro = [_chamada("prever_criativo", sem_fuso), AIMessage("Qual é o fuso?")]
    msgs = _perguntar(_agente(pacote, roteiro))
    (tool_msg,) = [m for m in msgs if isinstance(m, ToolMessage)]
    assert " ".join(agente.TEXTOS["erros"]["data_sem_fuso"].split()) in tool_msg.content
    assert msgs[-1].content == "Qual é o fuso?"


def test_erro_de_negocio_vira_toolmessage_legivel(pacote, capa):  # noqa: F811
    roteiro = [_chamada("prever_criativo", _pedido(capa, canal="UCnaoexiste")), AIMessage("ok")]
    msgs = _perguntar(_agente(pacote, roteiro))
    (tool_msg,) = [m for m in msgs if isinstance(m, ToolMessage)]
    assert " ".join(agente.TEXTOS["erros"]["canal_desconhecido"].split()) in tool_msg.content


def test_injecao_na_legenda_pelo_agente_fica_fora_da_saida_da_tool(pacote, capa):  # noqa: F811
    neutra = _pura(pacote, _pedido(capa)).predicted_metric
    roteiro = [_chamada("prever_criativo", _pedido(capa, legenda=INJECAO)), AIMessage("ok")]
    msgs = _perguntar(_agente(pacote, roteiro))
    (tool_msg,) = [m for m in msgs if isinstance(m, ToolMessage)]
    assert json.loads(tool_msg.content)["resposta"]["predicted_metric"] == neutra
    assert "ignore" not in tool_msg.content.lower() and "1.000.000" not in tool_msg.content


def test_as_tools_sao_do_nivel_do_pedido_com_os_textos_do_mcp(pacote):  # noqa: F811
    """6.2, EC2 e EC5: duas tools no nível do pedido, descrições do `configs/mcp.yaml`."""
    criadas = agente.criar_tools(bundle=pacote, encoder=EncoderFalso(), explicar=ExplicarFalso())
    tools = {t.name: t for t in criadas}
    assert set(tools) == {"prever_criativo", "comparar_criativos"}
    assert tools["prever_criativo"].args_schema is PredictionRequest
    for nome, t in tools.items():
        assert t.description == " ".join(agente.TEXTOS["tools"][nome].split())
        assert t.handle_tool_error is True
    assert agente.SISTEMA == " ".join(agente.TEXTOS["servidor"]["instrucoes"].split())
    assert "create_react_agent" not in inspect.getsource(agente)  # descontinuado (6.2, EC5)


# ---------------------------------------------------------------- CLI da demo


def test_cli_imprime_a_saida_do_grafo_em_json(pacote, tmp_path, monkeypatch, capsys):  # noqa: F811
    """`python -m previa.agente PEDIDO.json`: o stdout é só o JSON; sem `--publicado`, template."""
    falso = ExplicarFalso()
    monkeypatch.setitem(sys.modules, "previa.explicador", types.SimpleNamespace(explicar=falso))
    monkeypatch.setattr(previsao, "aquecer", lambda: None)
    monkeypatch.setattr(previsao, "carregar", lambda: pacote)
    pedido = _pedido(None, creative={"tipo": "texto"})
    arquivo = tmp_path / "lumina_texto.json"
    arquivo.write_text(json.dumps(pedido), encoding="utf-8")
    agente.main([str(arquivo)])
    out, err = capsys.readouterr()
    saida = json.loads(out)
    assert saida["resposta"] == _pura(pacote, pedido).model_dump(mode="json")
    assert [t.split(":")[0] for t in saida["trilha"]] == list(NOS)
    assert "extrair" in err and "aviso: extrair: pulado" in err
    assert falso.chamadas[0]["publicado"] is False and falso.chamadas[0]["llm"] is None


def test_cli_com_pedido_invalido_sai_com_erro_legivel(tmp_path, capsys):
    arquivo = tmp_path / "ruim.json"
    arquivo.write_text(json.dumps(_pedido(None, planned_datetime="2026-02-10T19:00:00")))
    with pytest.raises(SystemExit) as e:
        agente.main([str(arquivo)])
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert " ".join(agente.TEXTOS["erros"]["data_sem_fuso"].split()) in err
