"""A demonstração (TAPI §10; aulas 6.6 e 6.7; decisões #4 e #18): o "de-para" no golden set.

Regra da #18, escrita antes de rodar: de cada canal reservado fora do treino (o golden, #4), o Short
maduro mais recente; o par A/B sai do mesmo canal (os dois mais recentes). Cada Short entra como
cliente novo: `channel_stats` só com os posts anteriores (no par, os anteriores ao mais antigo dos
dois, porque a marca compara criativos com o mesmo passado), a capa do i.ytimg.com e a data real.
O número é o do modelo de uso; a explicação é o template (só Shorts publicados, mas sem LLM).

As respostas ficam em `DADOS/demo/` (fora do git), o plano B se a chamada ao vivo falhar (hub,
"Roteiro de demonstração"). Com `--mcp`, o par A/B do 1º canal passa pelo servidor MCP real, por
stdio, e a saída mostra que o número pelo MCP é o da função.

Com `--langchain`, o mesmo par passa pelo adaptador do LangGraph (`MultiServerMCPClient`, uma
sessão, `load_mcp_tools`; 6.5, erro comum 4), com o servidor lançado de OUTRO diretório e os
caminhos absolutos no ambiente, como um host faria (6.5, erro comum 2), e mede o tempo da chamada.

Com `--golden`, os 30 Shorts do golden, cada um como cliente novo, com a cobertura do intervalo e o
par-a-par dentro do canal (6.6, camada 4), em `DADOS/demo/golden.json`; sai com código 1 se algum
alarme de `configs/demo.yaml` disparar.

    uv run python -m previa.demo [--mcp] [--langchain]
    uv run python -m previa.demo --golden
    uv run python -m previa.demo --sanidade   (um canal fictício por tamanho, aula 1.4)
"""

from previa.openmp import um_runtime

if __name__ == "__main__":
    um_runtime()

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from previa.caminhos import ARTEFATOS, CONFIGS, DADOS, RAIZ  # noqa: E402

PASTA = DADOS / "demo"
GOLDEN = DADOS / "golden" / "golden_v1.parquet"
FUSO = "America/Sao_Paulo"


def selecionar(g: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[pd.Series, pd.Series]]]:
    """Regra da #18: o mais recente de cada canal e o par A/B (os 2 mais recentes do canal)."""
    g = g.sort_values(["handle", "published_at"])
    recentes = g.groupby("handle", sort=True).tail(1).reset_index(drop=True)
    pares = []
    for _, c in g.groupby("handle", sort=True):
        if len(c) >= 2:
            pares.append((c.iloc[-2], c.iloc[-1]))
    return recentes, pares


def _capa(video_id: str) -> str:
    destino = PASTA / "capas" / f"{video_id}.jpg"
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg", destino)
    return str(destino)


def pedido(s: pd.Series, hist: pd.DataFrame, corte=None) -> dict:
    """O Short do golden como pedido de um cliente novo: só os posts anteriores a `corte`."""
    quando = pd.Timestamp(s["published_at"])
    antes = hist[hist["published_at"] < pd.Timestamp(corte if corte is not None else quando)]
    posts = [
        {"published_at": p.published_at.isoformat(),
         "likes": None if np.isnan(p.y) else int(round(np.expm1(p.y)))}
        for p in antes.itertuples()
    ]  # fmt: skip
    return {
        "creative": {"tipo": "video", "arquivo": f"{s['video_id']}.mp4",
                     "duration_s": float(s["duration_s"]), "vertical": True},
        "caption": s["title"],
        "planned_datetime": quando.tz_convert(FUSO).isoformat(),
        "sector": s["setor"],
        "cover": _capa(s["video_id"]),
        "channel_stats": {"posts": posts},
    }  # fmt: skip


def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def ler_config_demo(caminho=CONFIGS / "demo.yaml") -> dict:
    """`configs/demo.yaml`: os limiares de alarme do golden."""
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def avaliar_golden(prev: pd.DataFrame, cfg: dict) -> dict:
    """O de-para do golden em números (6.6, camada 4): cobertura e par-a-par dentro do canal.

    `prev`: um Short por linha (`channel_id`, `real`, `predicted_metric`, `interval_low`,
    `interval_high`). Os pares são do mesmo canal; empate no real fica fora e empate na previsão
    vale 0,5 (`metricas.por_canal`, a regra da 2.4). Fora dos limiares de `configs/demo.yaml`, um
    alarme por regra: são folgados, para pegar um modelo quebrado, não para medir um bom.
    """
    from previa.metricas import por_canal

    real = prev["real"].to_numpy(float)
    dentro = int(((prev["interval_low"] <= real) & (real <= prev["interval_high"])).sum())
    t = por_canal(real, prev["predicted_metric"].to_numpy(float), prev["channel_id"].to_numpy())
    acertos, pares = float(t["acertos"].sum()), int(t["pares"].sum())
    n = len(prev)
    r = {
        "n": n,
        "dentro": dentro,
        "cobertura": dentro / n if n else np.nan,
        "acertos": acertos,
        "pares": pares,
        "par_a_par": acertos / pares if pares else np.nan,
    }
    lim = cfg["golden"]
    alarmes = []
    if not lim["cobertura_min"] <= r["cobertura"] <= lim["cobertura_max"]:
        alarmes.append(
            f"cobertura {r['cobertura']:.0%} fora de {lim['cobertura_min']:.0%} a "
            f"{lim['cobertura_max']:.0%}"
        )
    if not r["par_a_par"] >= lim["par_a_par_min"]:
        alarmes.append(f"par-a-par {r['par_a_par']:.0%} abaixo de {lim['par_a_par_min']:.0%}")
    return r | {"alarmes": alarmes}


async def _pelo_mcp(pedidos: list[dict]) -> dict:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    transporte = StdioTransport(
        command="uv", args=["run", "--directory", str(RAIZ), "python", "-m", "previa.servidor_mcp"]
    )
    async with Client(transporte, timeout=600) as c:
        print("tools do servidor:", [t.name for t in await c.list_tools()])
        r = await c.call_tool("comparar_criativos", {"pedidos": pedidos})
    return r.structured_content


def _golden(g: pd.DataFrame) -> int:
    """Os 30 Shorts do golden pelo mesmo caminho do de-para; imprime e grava a avaliação."""
    from previa.historico import historico_congelado
    from previa.previsao import prever_criativo
    from previa.schemas import PredictionRequest

    linhas = []
    for _, s in g.sort_values(["handle", "published_at"]).iterrows():
        r = prever_criativo(
            PredictionRequest.model_validate(pedido(s, historico_congelado(s["channel_id"])))
        )
        linhas.append({"handle": s["handle"], "video_id": s["video_id"],
                       "channel_id": s["channel_id"], "real": float(s["likes"]),
                       "predicted_metric": r.predicted_metric, "interval_low": r.interval_low,
                       "interval_high": r.interval_high,
                       "motivos": [m.codigo for m in r.confidence_flag.motivos]})  # fmt: skip
    prev = pd.DataFrame(linhas)
    a = avaliar_golden(prev, ler_config_demo())
    print(f"golden: {a['n']} Shorts de {prev['channel_id'].nunique()} canais fora da base")
    print(f"dentro do intervalo de 80%: {a['dentro']} de {a['n']} ({a['cobertura']:.1%})")
    print(f"par-a-par no canal: {a['acertos']:g} de {a['pares']} ({a['par_a_par']:.1%})")
    for motivo in sorted({m for ms in prev["motivos"] for m in ms}):
        print(f"flag {motivo}: {sum(motivo in ms for ms in prev['motivos'])} de {a['n']}")
    for alarme in a["alarmes"]:
        print(f"ALARME: {alarme}")
    (PASTA / "golden.json").write_text(
        json.dumps({"avaliacao": a, "previsoes": linhas}, ensure_ascii=False, indent=1)
    )
    print(f"\nsalvo em {PASTA / 'golden.json'}")
    print("comando: uv run python -m previa.demo --golden")
    return 1 if a["alarmes"] else 0


def sanidade_por_tamanho(
    cfg: dict, quando: str, setor: str, capa: str | None = None, bundle=None, encoder=None
) -> pd.DataFrame:
    """Um canal fictício por nível de `cfg["sanidade"]["curtidas"]` (aula 1.4), pela tool.

    Cada canal é `channel_stats` com `posts` Shorts semanais, todos maduros (o mais novo 31 dias
    antes de `quando`), com as mesmas curtidas. Uma linha por nível: o número, o intervalo, a
    razão (alto + 1) / (baixo + 1), que é a largura na escala log1p do modelo, e os códigos do flag.
    """
    from previa.previsao import prever_criativo
    from previa.schemas import PredictionRequest

    sc = cfg["sanidade"]
    t = pd.Timestamp(quando)
    datas = [t - pd.Timedelta(days=31 + 7 * k) for k in range(sc["posts"])]
    criativo = {"tipo": "imagem", "arquivo": capa} if capa else {"tipo": "texto"}
    linhas = []
    for nivel in sc["curtidas"]:
        posts = [{"published_at": d.isoformat(), "likes": int(nivel)} for d in datas]
        req = PredictionRequest.model_validate({
            "creative": criativo, "caption": "Canal fictício de sanidade",
            "planned_datetime": t.isoformat(), "sector": setor, "cover": capa,
            "channel_stats": {"posts": posts},
        })  # fmt: skip
        r = prever_criativo(req, bundle=bundle, encoder=encoder)
        linhas.append({
            "curtidas_canal": nivel, "predicted_metric": r.predicted_metric,
            "interval_low": r.interval_low, "interval_high": r.interval_high,
            # na escala do modelo (log1p): o alto/baixo cru exagera nos canais pequenos, pelo +1
            "razao_intervalo": (r.interval_high + 1) / (r.interval_low + 1),
            "motivos": [m.codigo for m in r.confidence_flag.motivos],
        })  # fmt: skip
    return pd.DataFrame(linhas)


def config_do_servidor() -> dict:
    """O servidor MCP como um host o lança: de outro diretório, com os caminhos absolutos no
    ambiente (`GALAXIES_ARTEFATOS`, `PREVIA_DADOS`; 6.5, erro comum 2)."""
    env = {**os.environ, "GALAXIES_ARTEFATOS": str(ARTEFATOS), "PREVIA_DADOS": str(DADOS)}
    return {
        "previa": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "--directory", str(RAIZ), "python", "-m", "previa.servidor_mcp"],
            "cwd": "/",
            "env": env,
        }
    }


def _json_da_tool(saida) -> dict:
    """A saída de uma tool do adaptador (texto ou blocos de conteúdo) como o dict da tool."""
    if isinstance(saida, tuple):  # (conteúdo, artefato)
        saida = saida[0]
    if isinstance(saida, list):
        saida = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in saida)
    return json.loads(saida)


async def _pelo_langchain(pedidos: list[dict]) -> tuple[dict, dict, list[str]]:
    """(resultado de `comparar_criativos`, segundos por chamada, nomes das tools) pelo adaptador.

    Uma sessão para todas as chamadas (6.5, erro comum 4). O servidor já sobe aquecido (`aquecer`
    no `main` dele): a 1ª chamada mede capas novas (o SigLIP roda), a 2ª as mesmas capas, que vêm do
    cache pelo conteúdo (`previa.capa`).
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    cliente = MultiServerMCPClient(config_do_servidor())
    tempos = {}
    async with cliente.session("previa") as sessao:
        tools = {t.name: t for t in await load_mcp_tools(sessao)}
        comparar = tools["comparar_criativos"]
        for rotulo in ("capas_novas", "capas_do_cache"):
            t0 = time.perf_counter()
            r = await comparar.ainvoke({"pedidos": pedidos})
            tempos[rotulo] = time.perf_counter() - t0
    return _json_da_tool(r), tempos, sorted(tools)


def main(argv=None) -> int:
    from previa.historico import historico_congelado
    from previa.previsao import comparar_criativos, prever_criativo
    from previa.schemas import PredictionRequest

    argv = list(sys.argv[1:] if argv is None else argv)
    PASTA.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(GOLDEN)
    if "--golden" in argv:
        return _golden(g)
    if "--sanidade" in argv:
        capas = sorted((PASTA / "capas").glob("*.jpg"))
        quando = pd.Timestamp.now(tz=FUSO).normalize() + pd.Timedelta(hours=19)
        t = sanidade_por_tamanho(
            ler_config_demo(), quando.isoformat(), "beleza", str(capas[0]) if capas else None
        )
        nome = capas[0].name if capas else "-"
        print(f"sanidade por tamanho (beleza, {quando:%d/%m/%Y %H:%M}, capa {nome}):")
        print(t.round(2).to_string(index=False))
        print("comando: uv run python -m previa.demo --sanidade")
        return 0
    recentes, pares = selecionar(g)
    salvo = {"de_para": [], "pares": []}

    print("== De-para: o Short maduro mais recente de cada canal fora da base (#18)\n")
    print(
        f"{'canal':<26}{'setor':<12}{'previsto':>10}{'intervalo de 80%':>20}{'real':>8}"
        "  dentro  flag"
    )
    for _, s in recentes.iterrows():
        r = prever_criativo(
            PredictionRequest.model_validate(pedido(s, historico_congelado(s["channel_id"])))
        )
        dentro = r.interval_low <= s["likes"] <= r.interval_high
        motivos = ",".join(m.codigo for m in r.confidence_flag.motivos) or "-"
        faixa = f"{_n(r.interval_low)} a {_n(r.interval_high)}"
        print(f"{s['handle']:<26}{s['setor']:<12}{_n(r.predicted_metric):>10}{faixa:>20}"
              f"{_n(s['likes']):>8}  {'sim' if dentro else 'não':<6}  {motivos}")  # fmt: skip
        salvo["de_para"].append({"handle": s["handle"], "video_id": s["video_id"],
                                 "real": float(s["likes"]),
                                 **r.model_dump(mode="json")})  # fmt: skip

    print("\n== A ou B? Os dois Shorts mais recentes de cada canal, com o mesmo passado\n")
    for a, b in pares:
        hist = historico_congelado(a["channel_id"])
        reqs = [
            PredictionRequest.model_validate(pedido(x, hist, corte=a["published_at"]))
            for x in (a, b)
        ]
        comp = comparar_criativos(reqs)
        p = comp["pares"][0]
        venceu_prev = "B" if p["p_b_maior_que_a"] >= 0.5 else "A"
        venceu_real = (
            "B" if b["likes"] > a["likes"] else ("A" if a["likes"] > b["likes"] else "empate")
        )
        prev = {x["criativo"]: x["predicted_metric"] for x in comp["ranking"]}
        print(f"{a['handle']:<26} previsto A {_n(prev['A'])} × B {_n(prev['B'])}"
              f" · P(B > A) {p['p_b_maior_que_a']:.0%}"
              f" · real A {_n(a['likes'])} × B {_n(b['likes'])}"
              f" · aposta {venceu_prev}, venceu {venceu_real}")  # fmt: skip
        salvo["pares"].append({"handle": a["handle"], "real_a": float(a["likes"]),
                               "real_b": float(b["likes"]), "ranking": comp["ranking"],
                               "pares": comp["pares"]})  # fmt: skip

    primeiro = salvo["de_para"][0]
    print(f"\n== A explicação de um criativo ({primeiro['handle']})\n")
    print(primeiro["explanation"]["resumo"] if isinstance(primeiro["explanation"], dict)
          else primeiro["explanation"])  # fmt: skip
    for d in primeiro["top_drivers"][:3]:
        print(" -", {k: d[k] for k in d if k in ("feature", "nome", "direcao", "multiplicador")})
    print(" - versão:", primeiro["model_version"]["rotulo"])

    (PASTA / "respostas.json").write_text(
        json.dumps(salvo, ensure_ascii=False, indent=1, default=str)
    )
    print(f"\nrespostas salvas em {PASTA / 'respostas.json'} (plano B)")

    if "--mcp" in argv:
        a, b = pares[0]
        hist = historico_congelado(a["channel_id"])
        pedidos = [pedido(x, hist, corte=a["published_at"]) for x in (a, b)]
        print(f"\n== O mesmo A ou B pelo servidor MCP ({a['handle']})\n")
        pelo_mcp = _pelo_mcp(pedidos)
        r = asyncio.run(pelo_mcp)
        print(
            "ranking pelo MCP:",
            [(x["criativo"], round(x["predicted_metric"], 1)) for x in r["ranking"]],
        )
        print("igual ao da função:", r["ranking"] == salvo["pares"][0]["ranking"])
    if "--langchain" in argv:
        a, b = pares[0]
        hist = historico_congelado(a["channel_id"])
        pedidos = [pedido(x, hist, corte=a["published_at"]) for x in (a, b)]
        print(
            f"\n== O mesmo A ou B pelo adaptador do LangGraph, de outro diretório ({a['handle']})\n"
        )
        r, tempos, nomes = asyncio.run(_pelo_langchain(pedidos))
        print("tools carregadas:", nomes)
        print("ranking pelo adaptador:",
              [(x["criativo"], round(x["predicted_metric"], 1)) for x in r["ranking"]])  # fmt: skip
        print("igual ao da função:", r["ranking"] == salvo["pares"][0]["ranking"])
        n = len(pedidos)
        print(f"tempo de uma comparação de {n} criativos com o servidor aquecido: "
              f"{tempos['capas_novas']:.2f} s com capas novas (~{tempos['capas_novas'] / n:.2f} s "
              f"por criativo), {tempos['capas_do_cache']:.2f} s com as mesmas capas")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
