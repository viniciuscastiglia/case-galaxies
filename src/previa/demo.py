"""A demonstração (TAPI §10; aulas 6.6 e 6.7; decisões #4 e #18): o "de-para" no golden set.

Regra da #18, escrita antes de rodar: de cada canal reservado fora do treino (o golden, #4), o Short
maduro mais recente; o par A/B sai do mesmo canal (os dois mais recentes). Cada Short entra como
cliente novo: `channel_stats` só com os posts anteriores (no par, os anteriores ao mais antigo dos
dois, porque a marca compara criativos com o mesmo passado), a capa do i.ytimg.com e a data real.
O número é o do modelo de uso; a explicação é o template (só Shorts publicados, mas sem LLM).

As respostas ficam em `DADOS/demo/` (fora do git), o plano B se a chamada ao vivo falhar (hub,
"Roteiro de demonstração"). Com `--mcp`, o par A/B do 1º canal passa pelo servidor MCP real, por
stdio, e a saída mostra que o número pelo MCP é o da função.

    uv run python -m previa.demo [--mcp]
"""

from previa.openmp import um_runtime

if __name__ == "__main__":
    um_runtime()

import asyncio  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from previa.caminhos import DADOS, RAIZ  # noqa: E402

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


def main(argv=None) -> int:
    from previa.historico import historico_congelado
    from previa.previsao import comparar_criativos, prever_criativo
    from previa.schemas import PredictionRequest

    argv = list(sys.argv[1:] if argv is None else argv)
    PASTA.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(GOLDEN)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
