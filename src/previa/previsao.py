"""`prever_criativo` e `comparar_criativos`: a API principal do Prévia (aulas 6.1 e 6.3).

Uma função pura (6.1, erro comum 3): o servidor MCP e o grafo só convertem formatos e erros e
nunca calculam uma feature. O número vem sempre do modelo (princípio 1): a resposta é
`expm1(prever_intervalo(...)[0])`, o modelo pontual que o teste final mede (#40). O LLM não entra;
a explicação é um template (#17) e só usa números calculados. A legenda e o texto do criativo são
dado, nunca instrução (princípio 8): a legenda só chega às features que o modelo calcula e nunca é
copiada para o texto da resposta.

O pacote carrega uma vez por processo (`modelo.carregar`, cacheado), nunca na importação, e
`aquecer()` o carrega junto com o encoder antes da primeira chamada (6.1). Este módulo não importa
torch: o encoder do SigLIP entra por argumento ou por `_encoder()`, e quem serve a tool chama
`previa.openmp.um_runtime()` antes dos imports (CLAUDE.md, armadilha do OpenMP).
"""

from functools import lru_cache
from math import erf, expm1, log1p, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from previa.amostra import ler_config as ler_config_midia
from previa.ancora import ancora
from previa.caminhos import ARTEFATOS, DADOS
from previa.capa import BLOCOS_DA_CAPA, extras_da_capa
from previa.drivers import carregar_cfg as carregar_cfg_explicacao
from previa.drivers import top_drivers
from previa.flag import confidence_flag
from previa.intervalo import em_curtidas
from previa.modelo import Bundle, carregar, features_de_um_short, model_version, prever_intervalo
from previa.schemas import PredictionRequest, PredictionResponse

Z80 = (
    1.2816  # o quantil 90% da normal: a largura do intervalo de 80% é 2 · 1,2816 · σ (6.3, Tool 5)
)
N_COMPARAVEIS = 3
SEM_CANAL = (
    "canal fora da base: reenvie com `channel_stats` (inscritos e posts com data); "
    "o canal {canal!r} não está na base do modelo"
)


# ---------------------------------------------------------------- formatos brasileiros


def _br(x: float, casas: int = 2) -> str:
    """Decimal com vírgula (`1,12`)."""
    return f"{x:.{casas}f}".replace(".", ",")


def _br_int(v: float) -> str:
    """Curtidas ditas com dois algarismos significativos e milhar com ponto: 12345 -> '12.000'.

    Um número com precisão que o modelo não tem seria mentira; abaixo de 100 vai inteiro.
    """
    v = max(float(v), 0.0)
    if v >= 100:
        casa = len(str(int(v))) - 2
        v = round(v, -casa)
    return f"{int(round(v)):,}".replace(",", ".")


def _data_br(texto: str) -> str:
    return pd.Timestamp(texto).strftime("%d/%m/%Y")


# ---------------------------------------------------------------- recursos do processo


@lru_cache(maxsize=1)
def _encoder():
    """O SigLIP real, carregado uma vez (import tardio: só o processo que serve a tool)."""
    from previa.visual import Siglip

    return Siglip(ler_config_midia()["siglip"]["ckpt"])


def aquecer(encoder=None) -> None:
    """Carrega o pacote e o encoder agora, para a primeira chamada não pagar a espera (6.1)."""
    carregar()
    if encoder is None:
        _encoder()


@lru_cache(maxsize=2)
def _ler_base(caminho: str, mtime_ns: int) -> pd.DataFrame:
    colunas = ["post_id", "channel_id", "published_at", "metric_target", "sector"]
    colunas += ["subscribers_at_collection", "duplicate_group"]
    return pd.read_parquet(caminho, columns=colunas)


def _base_do_pacote(bundle: Bundle) -> pd.DataFrame | None:
    """A base congelada do pacote (`DADOS / bundle.base`), ou None se o arquivo não existe."""
    caminho = DADOS / bundle.base
    if not bundle.base or not caminho.is_file():
        return None
    return _ler_base(str(caminho), caminho.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _ler_indice(caminho: str, mtime_ns: int):
    return joblib.load(caminho)


def _indice_de_comparaveis():
    caminho = Path(ARTEFATOS) / "comparaveis.joblib"
    if not caminho.is_file():
        return None
    return _ler_indice(str(caminho), caminho.stat().st_mtime_ns)


# ---------------------------------------------------------------- passos 2 e 3: posts e âncora

_COLUNAS_POSTS = ["channel_id", "published_at", "y"]


def _y(curtidas) -> pd.Series:
    """log1p das curtidas; ocultas (None/NaN) ficam NaN e não contam como maduras (F10)."""
    return np.log1p(pd.to_numeric(pd.Series(curtidas, dtype="object"), errors="coerce"))


def _inscritos(req: PredictionRequest, bundle: Bundle):
    """Inscritos do `channel_stats`, ou da base se o canal existe nela; None se nenhum dos dois."""
    if req.channel_stats is not None:
        return req.channel_stats.subscribers
    base = _base_do_pacote(bundle)
    if base is None or req.channel_id is None:
        return None
    subs = base.loc[base["channel_id"] == req.channel_id, "subscribers_at_collection"].dropna()
    return int(subs.iloc[-1]) if len(subs) else None


def _posts_do_canal(req: PredictionRequest, bundle: Bundle, quando: pd.Timestamp):
    """(posts, inscritos) do canal: da base congelada do pacote ou do `channel_stats`.

    Só posts anteriores à data planejada (o histórico vira feature só com o que existe antes).
    """
    if req.channel_stats is not None:
        s = req.channel_stats
        posts = pd.DataFrame(
            {
                "channel_id": "canal_do_cliente",
                "published_at": pd.to_datetime([p.published_at for p in s.posts], utc=True),
                "y": _y([p.likes for p in s.posts]).to_numpy(dtype=float),
            },
            columns=_COLUNAS_POSTS,
        )
        inscritos = s.subscribers
    else:
        base = _base_do_pacote(bundle)
        do_canal = base[base["channel_id"] == req.channel_id] if base is not None else None
        if do_canal is None or do_canal.empty:
            raise ValueError(SEM_CANAL.format(canal=req.channel_id))
        posts = pd.DataFrame(
            {
                "channel_id": do_canal["channel_id"].to_numpy(),
                "published_at": do_canal["published_at"].to_numpy(),
                "y": _y(do_canal["metric_target"].to_numpy()).to_numpy(dtype=float),
            }
        )
        subs = do_canal["subscribers_at_collection"].dropna()
        inscritos = int(subs.iloc[-1]) if len(subs) else None
    posts["published_at"] = pd.to_datetime(posts["published_at"], utc=True)
    return posts[posts["published_at"] < quando].reset_index(drop=True), inscritos


# ---------------------------------------------------------------- passo 8: fatores aprovados


def _aprovados(cfg: dict, bundle: Bundle, x: pd.DataFrame) -> set[str]:
    """Os nomes de driver dos blocos que a #36 aprovou como argumento (`blocos_argumento`).

    "capa" cobre o bloco legível "capa (SigLIP)" (os PCs) e as colunas dos blocos extras da capa
    (as sondas e a aparência), que aparecem com o próprio nome.
    """
    aprovados: set[str] = set()
    for bloco in cfg.get("blocos_argumento", []):
        aprovados |= {n for n in cfg["blocos"] if n.lower().startswith(bloco.lower())}
        if bloco == "capa":
            for b in BLOCOS_DA_CAPA:
                aprovados |= set(bundle.colunas_extras.get(b, [])) & set(x.columns)
    return aprovados


def _nome_do_fator(feature: str) -> str:
    """Como a frase chama o fator: a capa com o que a sonda ou a medida mede."""
    if feature.startswith("sonda_") and feature.endswith("_thumb"):
        return f"a capa ({feature[len('sonda_') : -len('_thumb')].replace('_', ' ')})"
    if feature.startswith("thumb_"):
        return f"a capa ({feature[len('thumb_') :].replace('_', ' ')})"
    if feature.startswith("capa"):
        return "a capa (aparência geral)"
    return feature


# ---------------------------------------------------------------- passo 9: comparáveis


def _comparaveis(emb, setor: str, quando: pd.Timestamp) -> tuple[list[dict], str | None]:
    """(até 3 comparáveis, ressalva). Sem índice ou sem capa: lista vazia e o motivo, nunca mudo.

    O índice (`ARTEFATOS/comparaveis.joblib`, `IndiceComparaveis`) é por setor e só devolve o
    passado maduro (30+ dias antes de `quando`), filtrado antes da busca (6.4).
    """
    indice = _indice_de_comparaveis()
    if indice is None:
        return [], "Comparáveis indisponíveis: o índice de Shorts parecidos não está no pacote."
    if emb is None:
        return [], "Comparáveis indisponíveis: sem a capa não há como buscar Shorts parecidos."
    v = indice.buscar(emb, setor, quando, k=N_COMPARAVEIS)
    curtidas = "likes" if "likes" in v.columns else "metric_target"
    return [
        {
            "post_id": str(r["post_id"]),
            "published_at": pd.Timestamp(r["published_at"]).tz_convert("UTC"),
            "likes": int(r[curtidas]),
            "similaridade": float(r["similarity"]),
        }
        for _, r in v.iterrows()
    ], None


# ---------------------------------------------------------------- passo 10: a explicação


def _proveniencia(bundle: Bundle) -> str:
    """F9: de onde o modelo veio, em quantos Shorts e canais, e que as curtidas são aos 30 dias."""
    ate = _data_br(bundle.ultimo_treino)
    base = _base_do_pacote(bundle)
    if base is None:
        return (
            f"Modelo treinado até {ate}; curtidas aos 30 dias "
            "(a contagem de Shorts e canais está indisponível: a base do pacote não foi achada)."
        )
    corte = pd.Timestamp(bundle.ultimo_treino)
    if corte.tzinfo is None:
        corte = corte.tz_localize("UTC")
    t = base[
        (base["published_at"] <= corte)
        & base["metric_target"].notna()
        & (base["duplicate_group"] == base["post_id"])
    ]
    return (
        f"Modelo treinado em {_br_int(len(t))} Shorts de {t['channel_id'].nunique()} canais, "
        f"até {ate}; curtidas aos 30 dias."
    )


def _explicacao(pred, lo, hi, drivers, flag, comparaveis_ressalva, bundle) -> dict:
    resumo = (
        f"Cerca de {_br_int(pred)} curtidas, entre {_br_int(lo)} e {_br_int(hi)} "
        "em 8 de cada 10 criativos como este."
    )
    fatores = [
        f"{_nome_do_fator(d['feature'])} {d['direcao']} a previsão, "
        f"×{_br(d['multiplicador'])} em relação ao Short típico do canal"
        for d in drivers
    ]
    ressalvas = []
    if flag.baixa_confianca:
        ressalvas.append("Baixa confiança: " + "; ".join(m.texto for m in flag.motivos) + ".")
    ressalvas.append("Isto é uma estimativa do modelo, não métrica do YouTube.")
    if drivers:
        ressalvas.append(
            "Os fatores são associações que o modelo aprendeu na base, não causas: "
            "trocar a capa não garante o mesmo efeito."
        )
    if comparaveis_ressalva:
        ressalvas.append(comparaveis_ressalva)
    ressalvas.append(_proveniencia(bundle))
    return {"resumo": resumo, "fatores": fatores, "ressalvas": ressalvas}


# ---------------------------------------------------------------- a API


def prever_criativo(
    req: PredictionRequest,
    *,
    bundle: Bundle | None = None,
    encoder=None,
    posts_do_canal: pd.DataFrame | None = None,
) -> PredictionResponse:
    """Prevê as curtidas aos 30 dias de um Short ainda não publicado (6.1, 6.3).

    `posts_do_canal` (opcional): `channel_id`, `published_at` (UTC) e `y` (log1p das curtidas)
    no lugar da base do pacote, para o golden set e para quem já tem o histórico em memória.
    Erros são explícitos: canal fora da base sem `channel_stats`, capa que não existe e âncora
    sem saída levantam (princípio 7); só o que a tool sabe degradar (capa ausente) vira flag.
    """
    bundle = bundle or carregar()
    # os passos em ordem fixa; o grafo da A6 chama as mesmas funções, um nó por passo (6.2, EC1)
    ext = passo_extrair(req, bundle, encoder)
    ctx = passo_contexto(req, bundle, posts_do_canal)
    prev = passo_prever(req, bundle, ctx, ext)
    return passo_explicar(req, bundle, ctx, ext, prev)


# ---------------------------------------------------------------- os passos (Tools 1 a 4 da 6.3)


def passo_extrair(req: PredictionRequest, bundle: Bundle, encoder=None) -> dict:
    """Tool 1, a capa (F11): a da marca ou a imagem do criativo; sem ela, blocos ausentes."""
    c = req.creative
    capa = req.cover or (c.arquivo if c.tipo == "imagem" else None)
    if capa is not None and encoder is None:
        encoder = _encoder()
    valores, embeddings, capa_ausente = extras_da_capa(capa, bundle, encoder)
    return {
        "capa": capa,
        "valores": valores,
        "embeddings": embeddings,
        "capa_ausente": capa_ausente,
    }


def passo_contexto(req: PredictionRequest, bundle: Bundle, posts_do_canal=None) -> dict:
    """Tool 2, o contexto: os posts anteriores do canal, os inscritos e a âncora (#34)."""
    quando = pd.Timestamp(req.planned_datetime).tz_convert("UTC")
    if posts_do_canal is not None:  # injetado (golden set, cliente externo): não toca a base
        posts = posts_do_canal[_COLUNAS_POSTS].copy()
        posts["published_at"] = pd.to_datetime(posts["published_at"], utc=True)
        posts = posts[posts["published_at"] < quando].reset_index(drop=True)
        inscritos = _inscritos(req, bundle)
    else:
        posts, inscritos = _posts_do_canal(req, bundle, quando)
    ref, degrau, n_maduros = ancora(posts, inscritos, req.sector, quando, bundle.ancora)
    return {"quando": quando, "posts": posts, "inscritos": inscritos, "ref": ref,
            "degrau": degrau, "n_maduros": n_maduros}  # fmt: skip


def passo_prever(req: PredictionRequest, bundle: Bundle, ctx: dict, ext: dict) -> dict:
    """Tool 3, a única fonte do número: features (a função do treino), número, intervalo e flag."""
    c = req.creative
    ausentes = {b: "sem capa" for b in ext["capa_ausente"] if b in bundle.extras}
    ausentes |= {  # um bloco do pacote que esta tool ainda não extrai: ausente, e o flag diz
        b: "bloco ainda não extraído pela tool" for b in bundle.extras if b not in BLOCOS_DA_CAPA
    }
    novo = {
        "published_at": ctx["quando"],
        "caption": req.caption,  # dado, nunca instrução: só vira feature calculada
        "sector": req.sector,
        "duration_s": c.duration_s,
        "extras": ext["valores"],
        "extras_ausentes": ausentes,
        "embeddings": ext["embeddings"],
    }
    x = features_de_um_short(bundle, ctx["posts"], novo)
    if c.tipo == "video" and c.duration_s is None:
        nao_extraido_duracao = ["a duração (duração não informada)"]
    else:
        nao_extraido_duracao = []
    nao_extraido = [f"o bloco {b}" for b in x.attrs["extracao_falhou"] if b not in BLOCOS_DA_CAPA]
    extracao_falhou = (["a capa"] if ext["capa_ausente"] else []) + nao_extraido_duracao
    extracao_falhou += nao_extraido

    # o número (modelo pontual, #40) e o intervalo (CQR)
    numero, lo_log, hi_log = prever_intervalo(bundle, x, ctx["ref"], ctx["degrau"])
    lo_c, hi_c = em_curtidas(lo_log, hi_log)

    # o flag; fora do regime avaliado (#34) é sempre canal_novo
    n_flag = ctx["n_maduros"]
    if ctx["degrau"] != "mediana_do_canal":
        n_flag = min(ctx["n_maduros"], bundle.limiares_flag["min_posts"] - 1)
    flag = confidence_flag(
        hi_log - lo_log, n_flag, extracao_falhou, None, req.planned_datetime, bundle.limiares_flag
    )
    return {"x": x, "numero": numero, "lo_c": float(lo_c), "hi_c": float(hi_c),
            "pred": max(expm1(numero), 0.0), "flag": flag}  # fmt: skip


def passo_explicar(
    req: PredictionRequest, bundle: Bundle, ctx: dict, ext: dict, prev: dict
) -> PredictionResponse:
    """Tool 4, a explicação: os fatores aprovados (#36, F7), os comparáveis e o template (#17)."""
    cfg = carregar_cfg_explicacao()
    x = prev["x"]
    drivers = top_drivers(bundle.modelo, x, None, cfg, aprovados=_aprovados(cfg, bundle, x))
    comparaveis, ressalva = _comparaveis(ext["embeddings"].get("thumb"), req.sector, ctx["quando"])
    pred, lo_c, hi_c, flag = prev["pred"], prev["lo_c"], prev["hi_c"], prev["flag"]
    return PredictionResponse(
        predicted_metric=pred,
        interval_low=lo_c,
        interval_high=hi_c,
        relative_to_channel=float(np.exp(prev["numero"] - ctx["ref"])),
        confidence_flag=flag,
        top_drivers=drivers,
        comparables=comparaveis,
        explanation=_explicacao(pred, lo_c, hi_c, drivers, flag, ressalva, bundle),
        features_used=list(bundle.features),
        model_version={k: v for k, v in model_version(bundle).items() if k != "calibracao"},
    )


# ---------------------------------------------------------------- comparar


def _chave_do_canal(req: PredictionRequest) -> str:
    if req.channel_id is not None:
        return f"id:{req.channel_id}"
    return "stats:" + req.channel_stats.model_dump_json()


def _p_b_maior(ra: PredictionResponse, rb: PredictionResponse) -> float:
    """P(B > A) = Φ((qB − qA) / √(σA² + σB²)), com q o número em log e σ = largura / (2 · 1,2816).

    Trata os dois como independentes; como os dois compartilham canal e modelo, é uma aproximação
    (conservadora: a covariância positiva diminuiria a variância da diferença, 6.3, Tool 5).
    """

    def q_sigma(r):
        lo, hi = log1p(r.interval_low), log1p(r.interval_high)
        return log1p(r.predicted_metric), (hi - lo) / (2 * Z80)

    qa, sa = q_sigma(ra)
    qb, sb = q_sigma(rb)
    s = sqrt(sa**2 + sb**2)
    if s == 0:
        return 0.5 if qa == qb else float(qb > qa)
    return 0.5 * (1 + erf(((qb - qa) / s) / sqrt(2)))


def _frase_do_par(a: str, b: str, p_b: float) -> str:
    melhor, outro, p_melhor = (b, a, p_b) if p_b >= 0.5 else (a, b, 1 - p_b)
    return (
        f"{melhor} tem a previsão mais alta, mas em ~{round(100 * (1 - p_melhor))}% dos cenários, "
        f"{outro} rende mais. A conta trata os dois como independentes, uma aproximação "
        "conservadora."
    )


def comparar_criativos(
    reqs: list[PredictionRequest], *, bundle: Bundle | None = None, encoder=None
) -> dict:
    """Compara criativos do mesmo canal: o ranking pelo número e P(B > A) de cada par (6.3).

    Cada criativo passa pelo `prever_criativo` (o número é o do modelo, sempre). Os criativos são
    nomeados A, B, C... na ordem recebida.
    """
    if len(reqs) < 2:
        raise ValueError("comparar_criativos precisa de pelo menos 2 criativos")
    if len({_chave_do_canal(r) for r in reqs}) != 1:
        raise ValueError(
            "todos os criativos devem ser do mesmo canal (mesmo 'channel_id' ou os mesmos "
            "'channel_stats'): a comparação cancela o que é do canal"
        )
    bundle = bundle or carregar()
    respostas = [prever_criativo(r, bundle=bundle, encoder=encoder) for r in reqs]
    nomes = [chr(ord("A") + i) for i in range(len(reqs))]
    ordem = sorted(range(len(reqs)), key=lambda i: respostas[i].predicted_metric, reverse=True)
    ranking = [
        {
            "posicao": k + 1,
            "criativo": nomes[i],
            "predicted_metric": respostas[i].predicted_metric,
            "interval_low": respostas[i].interval_low,
            "interval_high": respostas[i].interval_high,
        }
        for k, i in enumerate(ordem)
    ]
    pares = []
    for i in range(len(reqs)):
        for j in range(i + 1, len(reqs)):
            p_b = _p_b_maior(respostas[i], respostas[j])
            pares.append(
                {
                    "a": nomes[i],
                    "b": nomes[j],
                    "p_b_maior_que_a": p_b,
                    "p_a_maior_que_b": 1 - p_b,
                    "frase": _frase_do_par(nomes[i], nomes[j], p_b),
                }
            )
    return {"ranking": ranking, "pares": pares, "previsoes": respostas}
