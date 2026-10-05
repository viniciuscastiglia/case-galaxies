"""Curadoria de canais (aula 1.4 da trilha da liga).

Funil: candidatos (data/candidatos.csv) -> resolver handles (channels.list forHandle, 1 unidade
cada) -> medir critérios escritos ANTES de olhar resultados (docs/PLANO_DA_BASE.md) -> planilha
data/canais.csv com incluídos E excluídos, cada um com o motivo. A decisão final é humana:
a coluna `incluido` só vira 'sim' de vez depois da revisão do Vinicius.
"""

import argparse
import os
import re
import sys
import traceback
from datetime import datetime, timedelta, timezone

import httplib2
import numpy as np
import pandas as pd
from googleapiclient.errors import HttpError

from previa.coletor import (
    DADOS,
    PARTES_CANAL,
    Cliente,
    CotaEsgotada,
    abrir_base,
    buscar_canais,
    buscar_videos,
    listar_ids,
    parse_video,
    playlist_uploads,
    sem_chave,
)

MIN_SHORTS = 30  # histórico: pelo menos 30 Shorts
ATIVIDADE_DIAS = 90  # atividade: pelo menos 1 Short nos últimos 90 dias
MIN_CURTIDAS_VISIVEIS = 0.5  # likeCount presente na maioria dos Shorts
MAX_PAGINAS_AVALIACAO = 10  # lê até os 500 uploads mais recentes de cada candidato

# handle: 3 a 30 caracteres; letras, números, "_", "-" e "."; não pode ser e-mail
RE_HANDLE = re.compile(r"(?<![\w.@])@([\w][\w.-]{1,28}[\w])")
FAIXAS = [0, 10_000, 100_000, 1_000_000, np.inf]
ROTULOS = ["até 10 mil", "10–100 mil", "100 mil–1 mi", "mais de 1 mi"]
COLUNAS = [
    "channel_id",
    "handle",
    "nome",
    "titulo",
    "setor",
    "tipo",
    "faixa_inscritos",
    "inscritos_verificacao",
    "pais",
    "n_shorts_estimado",
    "ultimo_short",
    "frac_curtidas_visiveis",
    "incluido",
    "motivo",
    "fonte",
    "observacao",
    "verificado_em",
]


def _vazio(valor) -> bool:
    if isinstance(valor, str):
        return valor.strip() == ""
    return valor is None or bool(pd.isna(valor))


def normalizar_handle(texto: str) -> str:
    """'youtube.com/@Marca/shorts', 'Marca' ou ' @marca ' -> '@marca'."""
    h = texto.strip().split("youtube.com/")[-1].split("/")[0].split("?")[0]
    return "@" + h.lstrip("@").lower()


def achatar_canal(c: dict, handle: str | None = None) -> dict:
    sn, st = c["snippet"], c.get("statistics", {})
    return {
        "channel_id": c["id"],
        "handle": handle or sn.get("customUrl"),
        "titulo": sn["title"],
        "pais": sn.get("country"),  # opcional: muitos canais não preenchem
        "criado_em": sn["publishedAt"],
        "inscritos": None if st.get("hiddenSubscriberCount") else int(st["subscriberCount"]),
        "n_videos": int(st.get("videoCount", 0)),
        "uploads": c["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def resolver_handles(cli, handles: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """forHandle aceita UM handle por chamada: custa 1 unidade por handle."""
    linhas, nao_achados = [], []
    for h in handles:
        resp = cli.chamar(cli.yt.channels().list(part=PARTES_CANAL, forHandle=h))
        itens = resp.get("items", [])  # handle inexistente: sem "items"
        if itens:
            linhas.append(achatar_canal(itens[0], handle=h))
        else:
            nao_achados.append(h)
    return pd.DataFrame(linhas), nao_achados


def atualizar_em_lote(cli, channel_ids: list[str]) -> pd.DataFrame:
    """Com os IDs em mãos, 1 unidade a cada 50 canais."""
    return pd.DataFrame([achatar_canal(c) for c in buscar_canais(cli, channel_ids)])


def handles_em_textos(textos: list[str]) -> pd.Series:
    """Conta menções a @handles em descrições e títulos (colabs, parcerias)."""
    achados = [f"@{h.lower()}" for t in textos for h in RE_HANDLE.findall(t or "")]
    return pd.Series(achados, dtype="str").value_counts()


def canais_em_destaque(cli, channel_id: str) -> list[str]:
    """Seções 'multipleChannels' da página do canal (1 unidade por canal)."""
    req = cli.yt.channelSections().list(part="snippet,contentDetails", channelId=channel_id)
    ids = []
    for s in cli.chamar(req).get("items", []):
        if s["snippet"]["type"] == "multipleChannels":
            ids += s.get("contentDetails", {}).get("channels", [])
    return ids


def buscar_por_nome(cli_busca, nome: str) -> str | None:
    """Reserva para handle não encontrado (aula 1.4): search.list com type=channel, no balde
    próprio de 100 chamadas por dia. Devolve o channel_id do primeiro resultado."""
    req = cli_busca.yt.search().list(
        part="snippet",
        q=nome,
        type="channel",
        regionCode="BR",
        relevanceLanguage="pt",
        maxResults=1,
    )
    itens = cli_busca.chamar(req).get("items", [])
    return itens[0]["id"]["channelId"] if itens else None


def faixa(inscritos) -> str | None:
    if _vazio(inscritos):
        return None
    return str(pd.cut([float(inscritos)], FAIXAS, labels=ROTULOS, right=False)[0])


def estratificar(canais: pd.DataFrame) -> pd.DataFrame:
    """Tabela cruzada setor × tipo × faixa de inscritos: mostra onde a seleção está rala."""
    faixas = pd.cut(canais["inscritos"].astype(float), FAIXAS, labels=ROTULOS, right=False)
    # object em vez de categoria: as margens ('total') não cabem numa coluna categórica
    tabela = pd.crosstab(
        [canais["setor"], canais["tipo"]],
        faixas.astype(object).rename("faixa"),
        margins=True,
        margins_name="total",
    )
    return tabela[[r for r in ROTULOS if r in tabela.columns] + ["total"]]


def avaliar_canal(
    cli, canal: dict, hoje: datetime, max_paginas: int = MAX_PAGINAS_AVALIACAO
) -> dict:
    """Mede os critérios lendo os uploads mais recentes. Só elegibilidade (quantos Shorts,
    atividade, curtidas visíveis): quantas curtidas os vídeos têm não entra na decisão."""
    try:
        ids = listar_ids(cli, playlist_uploads(canal["channel_id"]), max_paginas=max_paginas)
    except HttpError as e:
        if e.status_code != 404:
            raise
        ids = []
    videos = [parse_video(it, hoje) for it in buscar_videos(cli, ids)]
    shorts = [v for v in videos if v["is_short"]]
    ultimo = max((v["published_at"] for v in shorts), default=None)
    visiveis = sum(v["likes"] is not None for v in shorts)
    return {
        "n_shorts_estimado": len(shorts),
        "ultimo_short": ultimo.date().isoformat() if ultimo else None,
        "frac_curtidas_visiveis": visiveis / len(shorts) if shorts else None,
        "leu_tudo": len(ids) < max_paginas * 50,
    }


def aplicar_criterios(linha: dict, hoje: datetime) -> tuple[str, str]:
    """Critérios da aula 1.4, escritos antes de olhar resultados. Devolve (incluido, motivo)
    com incluido em {'sim', 'não', 'revisar'}; o olho humano confirma país, idioma e setor."""
    pais = linha.get("pais")
    if not _vazio(pais) and pais != "BR":
        return "não", f"país {pais}"
    if _vazio(linha.get("inscritos_verificacao")):
        return "não", "inscritos ocultos"
    n = int(linha.get("n_shorts_estimado") or 0)
    if n < MIN_SHORTS:
        if linha.get("leu_tudo", True):
            return "não", f"menos de {MIN_SHORTS} Shorts ({n})"
        return "revisar", f"{n} Shorts entre os {MAX_PAGINAS_AVALIACAO * 50} uploads mais recentes"
    ultimo = linha.get("ultimo_short")
    corte = hoje - timedelta(days=ATIVIDADE_DIAS)
    if _vazio(ultimo) or datetime.fromisoformat(ultimo).replace(tzinfo=timezone.utc) < corte:
        return "não", f"sem Short há mais de {ATIVIDADE_DIAS} dias"
    frac = linha.get("frac_curtidas_visiveis")
    if not _vazio(frac) and float(frac) <= MIN_CURTIDAS_VISIVEIS:
        return "não", "curtidas ocultas"
    if _vazio(pais):
        return "revisar", "país não informado: conferir idioma e país"
    return "sim", "passa nos critérios automáticos"


def _linha_da_planilha(cli, base: dict, r: dict | None, por_busca: bool, hoje: datetime) -> dict:
    """Uma linha da planilha para um candidato: não encontrado, ou medido e julgado."""
    if r is None:
        return {**base, "incluido": "não", "motivo": "handle não encontrado"}
    linha = {
        **base,
        "channel_id": r["channel_id"],
        "titulo": r["titulo"],
        "pais": r["pais"],
        "inscritos_verificacao": r["inscritos"],
        "faixa_inscritos": faixa(r["inscritos"]),
        **avaliar_canal(cli, r, hoje),
    }
    linha["incluido"], linha["motivo"] = aplicar_criterios(linha, hoje)
    if linha["incluido"] == "sim" and (_vazio(base["setor"]) or _vazio(base["tipo"])):
        linha["incluido"], linha["motivo"] = "revisar", "definir setor e tipo na revisão"
    if por_busca:
        linha["fonte"] = f"{base['fonte']}+busca" if base["fonte"] else "busca"
        achado = f"handle não encontrado; a busca achou {r['handle']} ({r['titulo']})"
        linha["observacao"] = "; ".join(x for x in (base["observacao"], achado) if x)
        if linha["incluido"] == "sim":
            linha["incluido"], linha["motivo"] = "revisar", "achado por busca: confirmar o canal"
    return linha


def avaliar_candidatos(
    cli, candidatos: pd.DataFrame, hoje: datetime, cli_busca=None
) -> pd.DataFrame:
    """candidatos: colunas handle, nome, setor, tipo, fonte, observacao. Devolve as linhas da
    planilha de curadoria (COLUNAS), incluindo os excluídos e o motivo de cada um.

    Um erro de API num candidato vira 'revisar' sem derrubar os outros. Se a cota geral acabar,
    devolve o que já avaliou: o resto entra na próxima rodada (novos_candidatos). Se o balde de
    busca acabar, a avaliação segue sem a busca de reserva."""
    cand = candidatos.fillna("").assign(handle=lambda d: d["handle"].map(normalizar_handle))
    linhas, busca = [], cli_busca
    for c in cand.drop_duplicates("handle").to_dict("records"):
        base = {k: c.get(k, "") for k in ("handle", "nome", "setor", "tipo", "fonte", "observacao")}
        base["verificado_em"] = hoje.date().isoformat()
        try:
            resolvidos, _ = resolver_handles(cli, [c["handle"]])
            r = resolvidos.iloc[0].to_dict() if len(resolvidos) else None
            por_busca = False
            if r is None and busca is not None and c.get("nome"):
                try:
                    cid = buscar_por_nome(busca, c["nome"])
                except CotaEsgotada:  # acabou o balde de busca: segue sem ela
                    cid, busca = None, None
                achados = atualizar_em_lote(cli, [cid]) if cid else pd.DataFrame()
                if len(achados):
                    r, por_busca = achados.iloc[0].to_dict(), True
            linha = _linha_da_planilha(cli, base, r, por_busca, hoje)
        except (CotaEsgotada, OSError, httplib2.HttpLib2Error):
            break  # cota ou rede: o que já foi avaliado fica; o resto, na próxima rodada
        except HttpError as e:  # o texto do erro traz a chave na URL: só o status entra
            motivo = f"erro da API (HTTP {e.status_code}): avaliar de novo"
            linha = {**base, "incluido": "revisar", "motivo": motivo}
        linhas.append(linha)
    planilha = pd.DataFrame(linhas).reindex(columns=COLUNAS)
    duplicado = planilha["channel_id"].notna() & planilha.duplicated("channel_id")
    if duplicado.any():
        planilha.loc[duplicado, ["incluido", "motivo"]] = ["não", "canal repetido na lista"]
    return planilha


def novos_candidatos(candidatos: pd.DataFrame, planilha: pd.DataFrame) -> pd.DataFrame:
    """Só os candidatos que ainda não estão na planilha: a revisão humana nunca é sobrescrita."""
    ja = set(planilha["handle"].dropna().map(normalizar_handle)) if len(planilha) else set()
    return candidatos[~candidatos["handle"].map(normalizar_handle).isin(ja)]


def expandir(
    cli, con, planilha: pd.DataFrame, candidatos: pd.DataFrame, top: int = 30
) -> pd.DataFrame:
    """Candidatos novos contra o viés de sobrevivência (aula 1.4): handles citados nas
    descrições dos canais incluídos (fonte 'colab') e canais em destaque na página deles
    (fonte 'destaque'). Setor e tipo ficam para a revisão humana."""
    incluidos = planilha[planilha["incluido"].fillna("").str.strip().str.lower() == "sim"]
    ids = incluidos["channel_id"].dropna().tolist()
    ja = set(candidatos["handle"].map(normalizar_handle)) | set(planilha["handle"].dropna())
    textos = [
        r[0]
        for r in con.execute(
            "SELECT description FROM videos WHERE list_contains(?, channel_id)", [ids]
        ).fetchall()
    ]
    novos = [
        {
            "handle": h,
            "nome": "",
            "setor": "",
            "tipo": "",
            "fonte": "colab",
            "observacao": f"citado {n}x nas descrições da base",
        }
        for h, n in handles_em_textos(textos).items()
        if h not in ja
    ][:top]
    conhecidos = set(planilha["channel_id"].dropna())
    em_destaque = [c for cid in ids for c in canais_em_destaque(cli, cid) if c not in conhecidos]
    for r in atualizar_em_lote(cli, list(dict.fromkeys(em_destaque))).to_dict("records"):
        h = None if _vazio(r.get("handle")) else normalizar_handle(r["handle"])
        if h and h not in ja:
            novos.append(
                {
                    "handle": h,
                    "nome": r["titulo"],
                    "setor": "",
                    "tipo": "",
                    "fonte": "destaque",
                    "observacao": "em destaque na página de um incluído",
                }
            )
    colunas = ["handle", "nome", "setor", "tipo", "fonte", "observacao"]
    return pd.DataFrame(novos, columns=colunas).drop_duplicates("handle")


def _executar(args, yt, cli, hoje: datetime) -> None:
    caminho_cand, caminho_plan = DADOS / "candidatos.csv", DADOS / "canais.csv"
    candidatos = pd.read_csv(caminho_cand, dtype=str).fillna("")
    planilha = (
        pd.read_csv(caminho_plan, dtype=str)
        if caminho_plan.exists()
        else pd.DataFrame(columns=COLUNAS)
    )
    if args.acao == "expandir":
        novos = expandir(cli, abrir_base(), planilha, candidatos)
        pd.concat([candidatos, novos], ignore_index=True).to_csv(caminho_cand, index=False)
        print(f"{len(novos)} candidatos novos anexados a {caminho_cand}")
        return
    cli_busca = None
    if args.buscar_faltantes:
        cli_busca = Cliente(yt, limite=90, arquivo=DADOS / "cota_busca.json")
    pendentes = novos_candidatos(candidatos, planilha)
    avaliados = avaliar_candidatos(cli, pendentes, hoje, cli_busca)
    planilha = pd.concat([planilha, avaliados], ignore_index=True).reindex(columns=COLUNAS)
    planilha.to_csv(caminho_plan, index=False)
    print(planilha["incluido"].value_counts().to_string())
    faltam = pendentes["handle"].map(normalizar_handle).nunique() - len(avaliados)
    if faltam:
        print(f"{faltam} candidatos ficaram para a próxima rodada (a cota acabou ou a rede caiu)")
    incluidos = planilha[planilha["incluido"] == "sim"]
    if len(incluidos):
        tabela = incluidos.assign(inscritos=incluidos["inscritos_verificacao"])
        print(estratificar(tabela).to_string())


def main(argv=None) -> int:
    from googleapiclient.discovery import build

    parser = argparse.ArgumentParser(prog="python -m previa.curadoria")
    parser.add_argument("acao", choices=["avaliar", "expandir"])
    parser.add_argument(
        "--buscar-faltantes",
        action="store_true",
        help="usa search.list (balde de 100/dia) para handles não encontrados",
    )
    args = parser.parse_args(argv)
    chave = os.environ.get("YOUTUBE_API_KEY")
    if not chave:
        print(
            "YOUTUBE_API_KEY não definida: carregue o .env (set -a; source .env; set +a).",
            file=sys.stderr,
        )
        return 2
    yt = build("youtube", "v3", developerKey=chave, cache_discovery=False)
    cli, hoje = Cliente(yt), datetime.now(timezone.utc)
    try:
        _executar(args, yt, cli, hoje)
    except Exception:  # erro inesperado: mostra sem a chave e sai com falha
        print(sem_chave(traceback.format_exc()), file=sys.stderr)
        return 1
    print("unidades gastas hoje (PT):", cli.gasto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
