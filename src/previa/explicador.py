"""O explicador ancorado: um LLM que só conta o que o modelo disse (A4; aula 5.7; EDD 3).

Três peças, como na 5.7:

- `payload(resposta)`: o JSON fechado que o LLM recebe (#23). O número, o intervalo de 80%, o
  multiplicador sobre o canal, os motivos do flag, os `top_drivers` (nome, direção e
  multiplicador), os comparáveis (`post_id`, data e curtidas), a proveniência e o rótulo da
  versão. Sem legenda, sem transcrição, sem JSON do VLM, sem o SHAP inteiro (5.7, erro comum 1):
  é uma lista fechada de campos, e nenhum texto livre da resposta entra (princípio 8).
- `validar(texto, resposta)`: o validador determinístico de fidelidade (5.7; EDD 3, asserções de
  negócio e testes negativos). Lista vazia = aprovado. Confere números, fatores, intervalo, flag,
  termos proibidos e persona. Ele é literal de propósito: se um fato verdadeiro importa, o lugar
  dele é o payload, não o validador mais frouxo (5.7, erro comum 4).
- `explicar(resposta, llm, publicado)`: sem LLM, ou com criativo não publicado, devolve o
  template que a tool já calcula (#15, #17; princípio 5: criativo não publicado nunca vai para o
  free tier do Gemini). Com LLM e Short publicado: um pedido, o validador, UMA nova tentativa com
  os problemas e, se falhar de novo, o template com os problemas na saída (5.7: "falhe fechado").
  Nunca altera um número da resposta (princípio 1): o número vem do modelo, o LLM só narra.

O validador não enxerga afirmações sem número nem nome de fator ("a apresentadora é carismática"):
é necessário, não suficiente; o juiz offline da 5.7 é o que mede isso (fora desta peça).
"""

import json
import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from previa.caminhos import CONFIGS
from previa.coletor import sem_chave
from previa.drivers import carregar_cfg as carregar_cfg_explicacao
from previa.previsao import _nome_do_fator  # o mesmo nome de fator do template (paridade)
from previa.schemas import PredictionResponse

FUSO = ZoneInfo("America/Sao_Paulo")  # a data do comparável como a marca a conhece

# Números no formato brasileiro (5.7): "4.237" é quatro mil duzentos e trinta e sete, "0,884" é
# zero vírgula oitocentos e oitenta e quatro.
NUM = re.compile(r"\d+(?:[.,]\d+)*")
COLUNA = re.compile(r"\b[a-zà-ú]+(?:_[a-zà-ú0-9]+)+\b", re.IGNORECASE)  # jargão: hist_mediana
PALAVRA = re.compile(r"[\wà-úÀ-Ú]+")
# A proveniência do template (F9; `previsao._proveniencia`): só entra no payload se bater com este
# formato inteiro, para nenhum texto livre passar junto com ela.
PROVENIENCIA = re.compile(
    r"Modelo treinado (?:em [\d.]+ Shorts de \d+ canais, )?até \d{2}/\d{2}/\d{4}; "
    r"curtidas aos 30 dias(?: \(a contagem de Shorts e canais está indisponível: "
    r"a base do pacote não foi achada\))?\."
)
BAIXA = "Baixa confiança:"


def carregar_cfg(caminho=None) -> dict:
    """`configs/explicador.yaml`: o LLM, o prompt, a retentativa e os termos proibidos."""
    caminho = caminho or CONFIGS / "explicador.yaml"
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- o payload fechado (#23)


def _nomes_dos_fatores(resposta: PredictionResponse) -> list[str]:
    return [_nome_do_fator(d.feature) for d in resposta.top_drivers]


def _proveniencia(resposta: PredictionResponse) -> str | None:
    for r in resposta.explanation.ressalvas:
        if PROVENIENCIA.fullmatch(r):
            return r
    return None


def payload(resposta: PredictionResponse) -> dict:
    """Tudo o que o explicador pode citar, e nada além disso (5.7; 6.3, Tool 4; #23).

    As contas e os arredondamentos são do código (5.7, erro comum 2): curtidas inteiras,
    multiplicadores com duas casas e a data do comparável já em dd/mm/aaaa.
    """
    flag = resposta.confidence_flag
    return {
        "predicted_metric": round(resposta.predicted_metric),
        "interval_low": round(resposta.interval_low),
        "interval_high": round(resposta.interval_high),
        "nivel_intervalo": 0.8,
        "relative_to_channel": round(resposta.relative_to_channel, 2),
        "confidence_flag": {
            "baixa_confianca": flag.baixa_confianca,
            "motivos": [m.texto for m in flag.motivos],  # o texto que o código escreveu
        },
        "top_drivers": [
            {"fator": nome, "direcao": d.direcao, "multiplicador": round(d.multiplicador, 2)}
            for nome, d in zip(_nomes_dos_fatores(resposta), resposta.top_drivers, strict=True)
        ],
        "comparaveis": [
            {
                "post_id": c.post_id,
                "data": pd.Timestamp(c.published_at).tz_convert(FUSO).strftime("%d/%m/%Y"),
                "curtidas": c.likes,
            }
            for c in resposta.comparables
        ],
        "proveniencia": _proveniencia(resposta),
        "versao_do_modelo": resposta.model_version.rotulo,
    }


# ---------------------------------------------------------------- o texto do template


def texto_do_template(resposta: PredictionResponse) -> str:
    """A explicação que a tool já devolve (#17), no mesmo formato de três seções do LLM."""
    e = resposta.explanation
    partes = [e.resumo]
    if e.fatores:
        partes.append("Fatores:\n" + "\n".join(f"- {f}" for f in e.fatores))
    if e.ressalvas:
        partes.append("Ressalvas:\n" + "\n".join(f"- {r}" for r in e.ressalvas))
    return "\n\n".join(partes)


# ---------------------------------------------------------------- o validador (5.7; EDD 3)


def _para_float(txt: str) -> float:
    """'4.237' -> 4237; '0,884' -> 0.884 (convenção pt-BR, 5.7)."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", txt):
        return float(txt.replace(".", ""))
    return float(txt.replace(".", "").replace(",", "."))


def _bate_curtidas(c: float, v: float) -> bool:
    """O número citado `c` é uma forma aceita das curtidas `v`?

    A tolerância de arredondamento: o inteiro mais próximo (|c − v| ≤ 0,5) ou o arredondado do
    template (`previsao._br_int`), dois algarismos significativos, com até meia unidade do segundo
    algarismo e só múltiplo dela. Para 102,6 a unidade é 10: passam 103 e 100, não 105 nem 110.
    No empate (1.250), os dois lados passam. Abaixo de 100, só o inteiro.
    """
    if abs(c - v) <= 0.5:
        return True
    if v < 100:
        return False
    unidade = 10 ** (len(str(int(v))) - 2)
    return c % unidade == 0 and abs(c - v) <= unidade / 2


def _numeros_de(texto: str) -> list[float]:
    return [_para_float(n) for n in NUM.findall(texto)]


def _permitidos(p: dict, resposta: PredictionResponse) -> tuple[list[float], set[float]]:
    """(curtidas, exatos): as curtidas aceitam a tolerância; o resto, só o valor exato.

    Exatos: os multiplicadores e o relativo ao canal, as duas conversões que o prompt autoriza
    (multiplicador em porcentagem; o nível 0,8 em 80% ou "8 de cada 10") e os números dos textos
    que o código escreveu (motivos do flag, proveniência, datas dos comparáveis).
    """
    curtidas = [resposta.predicted_metric, resposta.interval_low, resposta.interval_high]
    curtidas += [p["predicted_metric"], p["interval_low"], p["interval_high"]]
    curtidas += [c["curtidas"] for c in p["comparaveis"]]
    nivel = p["nivel_intervalo"]
    exatos = {nivel, round(nivel * 100), round(nivel * 10), 10}
    for m in [d["multiplicador"] for d in p["top_drivers"]] + [p["relative_to_channel"]]:
        exatos |= {m, round(abs(m - 1) * 100)}
    textos = p["confidence_flag"]["motivos"] + [c["data"] for c in p["comparaveis"]]
    textos.append(p["proveniencia"] or "")
    for t in textos:
        exatos |= set(_numeros_de(t))
    return curtidas, {round(float(x), 2) for x in exatos}


def _secoes(texto: str) -> dict[str, list[str]]:
    """Os itens ("- ...") das seções "Fatores:" e "Ressalvas:" do texto."""
    secoes: dict[str, list[str]] = {}
    atual = None
    for linha in texto.splitlines():
        s = linha.strip()
        if s in ("Fatores:", "Ressalvas:"):
            atual = s[:-1]
            secoes[atual] = []
        elif atual and s.startswith("- "):
            secoes[atual].append(s[2:].strip())
    return secoes


def _negado(texto: str, inicio: int, janela: int) -> bool:
    antes = PALAVRA.findall(texto[:inicio].lower())[-janela:]
    return "não" in antes or "nem" in antes


def _sem(texto: str, trechos) -> str:
    """O texto sem os trechos que já são do payload (nomes de fator, post_id, versão), para os
    dígitos e os sublinhados deles não contarem como número ou jargão."""
    for t in sorted({t for t in trechos if t}, key=len, reverse=True):
        texto = re.sub(re.escape(t), " ", texto, flags=re.IGNORECASE)
    return texto


def validar(texto: str, resposta: PredictionResponse) -> list[str]:
    """Checagem determinística de fidelidade (5.7). Lista vazia = aprovado."""
    cfg = carregar_cfg()
    p = payload(resposta)
    nomes = [d["fator"] for d in p["top_drivers"]]
    mv = resposta.model_version
    limpo = _sem(texto, nomes + [c["post_id"] for c in p["comparaveis"]] + [mv.rotulo, mv.run_id])
    problemas = []

    # 1. todo número citado está no payload (com a tolerância das curtidas)
    curtidas, exatos = _permitidos(p, resposta)
    citados = [(n, _para_float(n)) for n in NUM.findall(limpo)]
    for bruto, c in citados:
        if round(c, 2) not in exatos and not any(_bate_curtidas(c, v) for v in curtidas):
            problemas.append(f"número que não está no payload: {bruto}")

    # 2. o intervalo aparece sempre (as duas pontas)
    for campo, v in (
        ("interval_low", resposta.interval_low),
        ("interval_high", resposta.interval_high),
    ):
        if not any(_bate_curtidas(c, v) for _, c in citados):
            problemas.append(f"o intervalo não foi citado ({campo})")

    # 3. os fatores citados estão contidos nos top_drivers, na mesma ordem
    secoes = _secoes(texto)
    indices = []
    for item in secoes.get("Fatores", []):
        i = next((k for k, n in enumerate(nomes) if item.lower().startswith(n.lower())), None)
        if i is None:
            problemas.append(f"fator fora dos top_drivers: {item[:60]!r}")
        else:
            indices.append(i)
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        problemas.append("os fatores não estão na ordem dos top_drivers")
    vocabulario = {_nome_do_fator(f) for f in resposta.features_used}
    vocabulario |= {_nome_do_fator(b) for b in carregar_cfg_explicacao()["blocos"]}
    for nome in sorted(vocabulario - set(nomes)):
        if COLUNA.fullmatch(nome):  # nome de coluna: a checagem do jargão, logo abaixo, pega
            continue
        if re.search(rf"(?<![\wà-ú]){re.escape(nome)}(?![\wà-ú])", limpo, re.IGNORECASE):
            problemas.append(f"fator fora dos top_drivers: {nome!r}")
    for coluna in sorted(set(COLUNA.findall(limpo))):
        problemas.append(f"nome de coluna (jargão) fora dos top_drivers: {coluna!r}")

    # 4. o flag: com ele, as ressalvas abrem com "Baixa confiança:"; sem ele, nada de inventá-lo
    ressalvas = secoes.get("Ressalvas", [])
    if not ressalvas:
        problemas.append("sem a seção 'Ressalvas:': a incerteza precisa aparecer")
    if p["confidence_flag"]["baixa_confianca"]:
        if not ressalvas or not ressalvas[0].startswith(BAIXA):
            problemas.append(f"o flag está ligado, mas a 1ª ressalva não abre com '{BAIXA}'")
    elif "baixa confiança" in texto.lower():
        problemas.append("'Baixa confiança' no texto, mas o flag está desligado")

    # 5. termos proibidos e persona (princípio 9; 5.7, erro comum 6)
    for padrao in cfg["proibidos_afirmativos"]:
        for m in re.finditer(rf"\b(?:{padrao})", texto, re.IGNORECASE):
            if not _negado(texto, m.start(), cfg["janela_negacao"]):
                problemas.append(f"termo proibido: {m.group(0)!r}")
    for padrao in cfg["proibidos_sempre"]:
        for m in re.finditer(padrao, texto, re.IGNORECASE):
            problemas.append(f"termo proibido (persona ou promessa): {m.group(0)!r}")
    return problemas


# ---------------------------------------------------------------- o LLM e o laço de correção


def llm_padrao():
    """O Gemini do explicador (#15): `gemini-3.5-flash-lite`, temperatura 0, timeout de 300 s."""
    chave = os.environ.get("GOOGLE_API_KEY")
    if not chave:
        raise RuntimeError(
            "GOOGLE_API_KEY não definida: carregue o .env (set -a; source .env; set +a) para "
            "usar o explicador LLM, ou chame explicar() sem llm para o template"
        )
    from langchain_google_genai import ChatGoogleGenerativeAI  # import tardio: só quem usa

    cfg = carregar_cfg()["llm"]
    return ChatGoogleGenerativeAI(
        model=cfg["modelo"],
        temperature=cfg["temperatura"],
        timeout=cfg["timeout_s"],
        google_api_key=chave,
    )


def _texto_da_mensagem(saida) -> str:
    """O texto do `AIMessage` (o Gemini pode devolver o conteúdo em blocos)."""
    if isinstance(saida, AIMessage):
        return saida.text
    return str(getattr(saida, "content", saida))


def explicar(resposta: PredictionResponse, llm=None, publicado: bool = False) -> dict:
    """{"texto": str, "fonte": "llm" | "template", "problemas": list[str]}.

    Sem `llm`, ou com `publicado=False`, devolve o template sem chamar o LLM (#15). Com os dois:
    um pedido com o payload, o validador, uma nova tentativa com os problemas e, se falhar de
    novo, o template com os problemas (5.7). Uma falha do LLM (rede, cota) também fecha com o
    template, com o erro em `problemas` (princípio 7), e o número fica (6.3, Tool 4).
    """
    template = texto_do_template(resposta)
    if llm is None or not publicado:
        return {"texto": template, "fonte": "template", "problemas": []}
    cfg = carregar_cfg()
    p = payload(resposta)
    mensagens = [SystemMessage(cfg["sistema"]), HumanMessage(json.dumps(p, ensure_ascii=False))]
    problemas: list[str] = []
    for _ in range(2):  # uma tentativa e uma correção, não cinco (5.7)
        try:
            texto = _texto_da_mensagem(llm.invoke(mensagens)).strip()
        except Exception as e:  # qualquer falha do LLM (rede, cota) fecha com o template
            problemas = [f"o LLM falhou: {type(e).__name__}: {sem_chave(str(e))[:300]}"]
            break
        problemas = validar(texto, resposta)
        if not problemas:
            return {"texto": texto, "fonte": "llm", "problemas": []}
        pedido = cfg["retentativa"].format(problemas="; ".join(problemas))
        mensagens = [*mensagens, AIMessage(texto), HumanMessage(pedido)]
    return {"texto": template, "fonte": "template", "problemas": problemas}
