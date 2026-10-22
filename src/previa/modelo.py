"""O modelo de uso (aulas 3.4, 6.1 e 6.3; princípio 4 do CLAUDE.md).

Treinado uma vez no bloco de treino (até 30 dias antes da calibração), salvo com uma versão,
carregado uma vez por processo (`lru_cache`, 6.1), e as features de um Short novo montadas com as
MESMAS funções do treino: `montar_features` sobre os posts do canal mais a linha nova (sem y),
`manuais` sobre a legenda e o PCA de cada bloco de embedding ajustado no treino. O teste de
paridade (tests/test_modelo.py) prova que a previsão vê o que o treino viu.

O treino segue o runner (`previa.experimento`): alvo relativo ao canal (05/10), early stopping nos
últimos `val_dias` do treino e refit com o nº de árvores escolhido. A previsão é a saída do modelo
mais a mediana madura do canal (`referencia`), em log1p(curtidas). O intervalo (quantis e CQR)
chega com a M6; até lá, o bundle tem o modelo pontual.
"""

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import joblib
import lightgbm as lgb  # antes de qualquer torch no processo (hub, troubleshooting)
import mlflow
import numpy as np
import pandas as pd

from previa.caminhos import ARTEFATOS, CONFIGS
from previa.esquema import validar_tabela_de_treino
from previa.experimento import (
    EXPERIMENTO,
    Config,
    aplicar_pca,
    codigo_setor,
    coluna_alvo,
    commit_atual,
    expandir_features,
    hash_base,
    ler_config,
    ler_duracoes,
    ler_embeddings_do_bloco,
    ler_extra,
    preparar,
    prever_absoluto,
    treinar_gbm,
)
from previa.features import historico_do_canal, montar_features
from previa.flag import carregar_cfg as carregar_cfg_flag
from previa.flag import limiares_da_calibracao
from previa.intervalo import (
    conter_o_numero,
    escores_cqr,
    intervalo,
    prever_quantis,
    treinar_quantis,
)
from previa.intervalo import ler_config as ler_config_intervalo
from previa.intervalo import q_chapeu as calcular_q_chapeu
from previa.texto import manuais

NOVO = "__novo__"


@dataclass
class Bundle:
    modelo: lgb.Booster
    features: list[str]
    pcas: dict
    versao: str
    lacuna_dias: int
    ultimo_treino: str
    arvores: int
    config: dict = field(default_factory=dict)
    run_id: str | None = None  # o run do MLflow que registrou este pacote (#37, F23)
    base: str = ""  # o nome do arquivo da base congelada
    hash_base: str = ""  # a impressão digital da base de treino
    extras: list[str] = field(default_factory=list)  # os blocos extras (F17): `ler_extra(nome)`
    colunas_extras: dict = field(default_factory=dict)  # bloco extra -> suas colunas em `features`
    # M6 (campos novos com default: um pacote antigo carrega sem eles)
    quantis: dict = field(default_factory=dict)  # alpha -> Booster, nas mesmas linhas e features
    q_chapeu: float | None = None  # o q̂ do CQR, só da calibração (5.3)
    calibracao: dict = field(default_factory=dict)  # {"inicio", "fim"} do bloco de calibração
    limiares_flag: dict = field(default_factory=dict)  # `flag.limiares_da_calibracao` (5.4)
    ancora: dict = field(default_factory=dict)  # `ancora.ajustar_ancora` (#34)


def fim_do_treino(cfg_av: dict) -> pd.Timestamp:
    """Onde o treino do modelo de uso termina: `cal_inicio` - `gap_dias`."""
    return cfg_av["cal_inicio"] - pd.Timedelta(days=cfg_av["gap_dias"])


def treinar_bundle(
    cfg: Config, base, duracoes, cfg_av, embeddings=None, extras=None, cfg_q=None, cfg_flag=None
) -> Bundle:
    """O modelo final da config: todas as linhas do bloco de treino, sem a calibração e o teste.

    `extras`: nome do bloco -> tabela por `post_id` (`ler_extra`), como no runner. Os curingas da
    config (`vlm_*`, `sonda_*`, `det_*`) viram as colunas do bloco, e o pacote guarda a lista
    expandida: a previsão monta a mesma ordem.

    M6: além do pontual, treina os 3 quantis nas mesmas linhas e features, calcula o q̂ e os
    limiares do flag no bloco de calibração do `cfg_av` (só linhas `linha`, nunca o treino) e
    ajusta a âncora do cold start (#34) com os posts do treino.
    """
    from previa.ancora import ajustar_ancora  # aqui: previa.ancora importa este módulo

    cfg_q = cfg_q or ler_config_intervalo(CONFIGS / "intervalo.yaml")
    cfg_flag = cfg_flag or carregar_cfg_flag()
    d = preparar(
        base, duracoes, cfg_av, cfg.lacuna_dias, min_posts_alvo=cfg.min_posts_alvo, extras=extras
    )
    base_feats = expandir_features(cfg.features, d.columns)  # curinga sem coluna levanta
    validar_tabela_de_treino(d, base_feats, cfg.lacuna_dias)  # o contrato do runner (F15; 1.8, 3.5)
    tr = d.index[d["linha"] & (d["published_at"] < fim_do_treino(cfg_av))]
    dobra, feats, pcas = d.loc[tr].copy(), list(base_feats), {}
    for bloco, n_comp in cfg.pca.items():
        cols, pca = aplicar_pca(
            embeddings[bloco], d.loc[tr, "post_id"], dobra["post_id"], n_comp, bloco, cfg.seed
        )
        dobra[list(cols.columns)] = cols.to_numpy()
        feats += list(cols.columns)
        pcas[bloco] = pca
    modelo, n = treinar_gbm(dobra, tr, cfg, feats)
    quantis = treinar_quantis(dobra, tr, cfg, feats, cfg_q)
    # a calibração: depois do gap, antes do teste; linhas rotuladas que o modelo não viu
    quando = d["published_at"]
    cal_idx = d.index[
        d["linha"] & (quando >= cfg_av["cal_inicio"]) & (quando < cfg_av["teste_inicio"])
    ]
    # sem `linhas_de_treino`: o teto por canal e a idade do rótulo são do treino e encolheriam
    # a calibração em silêncio; aqui só alvo definido e âncora (sem ela não há escore)
    cal_idx = cal_idx[(d.loc[cal_idx, [coluna_alvo(cfg), "ref"]].notna().all(axis=1)).to_numpy()]
    if len(cal_idx) == 0:
        raise ValueError(
            f"a calibração ({cfg_av['cal_inicio']} a {cfg_av['teste_inicio']}) não tem nenhum "
            "post rotulado: sem ela não há q̂ nem limiares do flag"
        )
    assert d.loc[cal_idx, "linha"].all(), "a calibração só usa linhas marcadas (5.3, erro 3)"
    assert not cal_idx.intersection(tr).size, "a calibração não pode tocar o treino (5.3, erro 2)"
    cal = d.loc[cal_idx].copy()
    for bloco, pca in pcas.items():  # o PCA do treino, aplicado à calibração (nada reajustado)
        ids = embeddings[bloco].index.intersection(pd.Index(cal["post_id"]))
        cols = [f"{bloco}_pc{k:02d}" for k in range(pca.n_components_)]
        cal[cols] = np.nan
        sel = cal["post_id"].isin(ids)
        cal.loc[sel, cols] = pca.transform(
            embeddings[bloco].loc[cal.loc[sel, "post_id"]].to_numpy()
        )
    q, _ = prever_quantis(quantis, cal[feats], cal["ref"], cfg)
    qhat = calcular_q_chapeu(escores_cqr(q[:, 0], q[:, 2], cal["y"]), cfg_q.alpha_cqr)
    if not np.isfinite(qhat):
        raise ValueError(
            f"calibração pequena demais ({len(cal)} posts) para o nível {1 - cfg_q.alpha_cqr:.0%}"
        )
    numero = prever_absoluto(modelo, cal[feats], cal["ref"], cfg)
    lo, hi = conter_o_numero(*intervalo(q[:, 0], q[:, 2], qhat), numero)
    limiares = limiares_da_calibracao(pd.Series(hi - lo), None, fim_do_treino(cfg_av), cfg_flag)
    inscritos = base.groupby("channel_id")["subscribers_at_collection"].last()
    ancora = ajustar_ancora(d.loc[tr], inscritos, fim_do_treino(cfg_av), cfg.lacuna_dias)
    assinatura = f"{feats}|{hash_base(base)}|{n}".encode()
    versao = f"gbm-{cfg.nome}-{date.today():%Y%m%d}-{hashlib.sha1(assinatura).hexdigest()[:7]}"
    ultimo = str(d.loc[tr, "published_at"].max())
    return Bundle(
        modelo, feats, pcas, versao, cfg.lacuna_dias, ultimo, n, asdict(cfg),
        base=str(cfg_av.get("base", "")), hash_base=hash_base(base),
        extras=list(cfg.extras),
        colunas_extras={n: [c for c in feats if c in b.columns] for n, b in (extras or {}).items()},
        quantis=quantis, q_chapeu=float(qhat),
        calibracao={
            "inicio": str(pd.Timestamp(cfg_av["cal_inicio"]).date()),
            "fim": str(pd.Timestamp(cfg_av["teste_inicio"]).date()),
            "n": int(len(cal_idx)),
        },
        limiares_flag=limiares, ancora=ancora,
    )  # fmt: skip


def model_version(bundle: Bundle) -> dict:
    """De qual modelo veio o número: a tool devolve isto em toda resposta (#37, F23; aula 6.5)."""
    if not bundle.run_id:
        raise ValueError("o pacote não tem run id: registre-o com registrar_bundle antes de servir")
    rotulo = f"{bundle.versao} · run {bundle.run_id[:8]} · treino até {bundle.ultimo_treino[:10]}"
    return {
        "versao": bundle.versao,
        "run_id": bundle.run_id,
        "ultimo_treino": bundle.ultimo_treino,
        "base": bundle.base,
        "hash_base": bundle.hash_base,
        "rotulo": rotulo,
        "calibracao": bundle.calibracao,  # fora do contrato `ModelVersion` (schemas.py)
    }


def salvar(bundle: Bundle, destino=ARTEFATOS) -> Path:
    """Grava `modelo.joblib` e, ao lado, `model_version.json` (só com run id; #37)."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destino / "modelo.joblib")
    if bundle.run_id:
        (destino / "model_version.json").write_text(
            json.dumps(model_version(bundle), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return destino / "modelo.joblib"


def registrar_bundle(bundle: Bundle, cfg_av: dict, caminho_config, destino=ARTEFATOS) -> str:
    """Abre o run do modelo de uso no MLflow, salva o pacote e registra os dois arquivos nele.

    O arquivo que a tool lê da pasta é o mesmo que fica como artefato do run (#37, opção A).
    """
    mlflow.set_experiment(EXPERIMENTO)
    destino = Path(destino)
    with mlflow.start_run(run_name=f"modelo_de_uso-{bundle.versao}") as run:
        mlflow.set_tags(
            {
                "papel": "modelo_de_uso",
                "git_commit": commit_atual(),
                "hash_base": bundle.hash_base,
                "data_corte": str(
                    fim_do_treino(cfg_av).date()
                ),  # o fim do treino, não o do walk-forward
                "config": str(caminho_config),
            }
        )
        bundle.run_id = run.info.run_id
        try:
            # grava numa pasta temporária, registra no run e só então troca os arquivos da pasta
            # de uso: se o MLflow falhar, o pacote bom anterior continua lá (falha não silenciosa)
            destino.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destino) as tmp:
                pacote = salvar(bundle, tmp)
                mlflow.log_artifact(str(pacote))
                mlflow.log_artifact(str(pacote.parent / "model_version.json"))
                for nome in ("modelo.joblib", "model_version.json"):
                    os.replace(Path(tmp) / nome, destino / nome)
        except Exception:
            bundle.run_id = None
            raise
        carregar.cache_clear()  # um `carregar` anterior no processo guardaria o pacote antigo
        return run.info.run_id


def main(argv=None) -> int:
    """`treinar CONFIG [AVALIACAO]`: treina o modelo final, registra o run e imprime o rótulo."""
    from previa.dados import carregar_base
    from previa.splits import ler_avaliacao

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) not in (2, 3) or argv[0] != "treinar":
        print("uso: python -m previa.modelo treinar CONFIG [AVALIACAO]", file=sys.stderr)
        return 2
    caminho = Path(argv[1])
    cfg = ler_config(caminho)
    caminho_av = argv[2] if len(argv) == 3 else CONFIGS / "avaliacao.yaml"
    cfg_av = ler_avaliacao(caminho_av)
    base = carregar_base(caminho_config=caminho_av)  # base, datas, nome e hash da mesma config
    embeddings = {b: ler_embeddings_do_bloco(b) for b in cfg.pca}
    extras = {e: ler_extra(e) for e in cfg.extras}
    bundle = treinar_bundle(cfg, base, ler_duracoes(), cfg_av, embeddings, extras)
    registrar_bundle(bundle, cfg_av, caminho, destino=ARTEFATOS)
    print(model_version(bundle)["rotulo"])
    return 0


@lru_cache(maxsize=1)
def carregar(origem=ARTEFATOS) -> Bundle:
    """Lê o modelo do disco UMA vez por processo; as chamadas seguintes vêm do cache (6.1)."""
    return joblib.load(Path(origem) / "modelo.joblib")


def features_de_um_short(bundle: Bundle, posts_canal: pd.DataFrame, novo: dict) -> pd.DataFrame:
    """As features de um Short ainda não publicado, numa linha, na ordem de `bundle.features`.

    `posts_canal`: os posts do canal (channel_id, published_at UTC, y = log1p das curtidas, NaN se
    ocultas), de qualquer idade: a lacuna de maturação é aplicada aqui, como no treino.
    `novo`: published_at (com fuso), caption, sector, duration_s, `embeddings` (bloco -> vetor) e
    `extras` (coluna -> valor, dos blocos extras). Coluna de extra ausente levanta, a menos que
    `novo["extras_ausentes"]` (bloco -> motivo) declare o bloco que faltou: aí vira NaN e o motivo
    sai em `x.attrs["extracao_falhou"]`, para o flag de confiança (princípio 7).
    """
    canal = posts_canal["channel_id"].iloc[0] if len(posts_canal) else "canal_novo"
    quando = pd.Timestamp(novo["published_at"]).tz_convert("UTC")
    linha = pd.DataFrame({"channel_id": [canal], "published_at": [quando], "y": [np.nan]})
    todas = pd.concat([posts_canal[["channel_id", "published_at", "y"]], linha.set_axis([NOVO])])
    f = montar_features(todas, bundle.lacuna_dias).loc[[NOVO]]
    duracao = novo.get("duration_s")
    f["duracao_s"] = np.nan if duracao is None else float(duracao)
    f["setor"] = codigo_setor(pd.Series([novo["sector"]], index=f.index))
    f = f.join(manuais(pd.Series([novo.get("caption", "")], index=f.index)))
    for bloco, pca in bundle.pcas.items():
        vetor = (novo.get("embeddings") or {}).get(bloco)
        comps = (
            pca.transform(np.asarray(vetor, float)[None, :])[0]
            if vetor is not None
            else np.full(pca.n_components_, np.nan)  # sem embedding: o GBM trata como ausente
        )
        for k, v in enumerate(comps):
            f[f"{bloco}_pc{k:02d}"] = v
    declarados = novo.get("extras_ausentes") or {}
    valores = novo.get("extras") or {}
    for bloco in bundle.extras:
        for col in bundle.colunas_extras.get(bloco, []):
            if col in valores:
                f[col] = float(valores[col])
            elif bloco in declarados:
                f[col] = np.nan  # o bloco falhou e o chamador disse; o GBM trata como ausente
            else:
                raise ValueError(
                    f"a coluna {col} do bloco extra {bloco} não veio em novo['extras']; se o bloco "
                    "falhou, declare-o em novo['extras_ausentes'] (bloco -> motivo)"
                )
    for k, v in bundle.config.get("fixar", {}).items():  # #21: controles fixados na previsão
        f[k] = v
    x = f[bundle.features]
    x.attrs["extracao_falhou"] = {b: str(m) for b, m in declarados.items() if b in bundle.extras}
    return x


def referencia(posts_canal: pd.DataFrame, quando) -> tuple[float, int]:
    """A mediana madura do canal (log1p, 30 dias) em `quando` e quantos posts ela usa.

    É a âncora do alvo relativo (05/10; 1.6): a previsão é essa mediana mais a saída do modelo.
    """
    linha = pd.DataFrame(
        {"channel_id": ["canal"], "published_at": [pd.Timestamp(quando).tz_convert("UTC")]},
    ).assign(y=np.nan)
    posts = posts_canal[["published_at", "y"]].assign(channel_id="canal")
    h = historico_do_canal(pd.concat([posts, linha.set_axis([NOVO])]), 30).loc[NOVO]
    return float(h["hist_mediana"]), int(h["hist_n"])


MIN_POSTS_ANCORA = (
    3  # #20 (2.1): com menos, o baseline nem existe; entra a cadeia de #34 (ancora.py)
)
DEGRAUS = ("mediana_do_canal", "inscritos", "mediana_do_setor")


def prever_log(bundle: Bundle, x: pd.DataFrame, ref=None, degrau="mediana_do_canal") -> float:
    """log1p das curtidas previstas para a linha `x` (o único lugar de onde sai o número).

    No alvo relativo, `ref` é a âncora de `previa.ancora.ancora` e `degrau` o degrau da cadeia de
    cold start (#34; 2.1): "mediana_do_canal" (regime avaliado), "inscritos" ou "mediana_do_setor".
    Os dois últimos dão número, mas o chamador deve sinalizar `canal_novo` (5.4).
    """
    r = float(bundle.modelo.predict(x)[0])
    if bundle.config.get("alvo", "relativo") != "relativo":
        return r
    if degrau not in DEGRAUS:
        raise ValueError(f"degrau da âncora desconhecido: {degrau!r}; use um de {DEGRAUS}")
    if ref is None or np.isnan(ref):
        raise ValueError(f"alvo relativo: sem âncora para a previsão (degrau {degrau})")
    return r + float(ref)


def prever_intervalo(
    bundle: Bundle, x: pd.DataFrame, ref=None, degrau="mediana_do_canal"
) -> tuple[float, float, float]:
    """(número, baixo, alto) em log1p: o número é o do modelo pontual (#40), o intervalo é CQR.

    Baixo e alto: os quantis ordenados (5.2), alargados pelo q̂ da calibração (5.3) e, se
    preciso, até conter o número (só aumenta a cobertura).
    """
    if not bundle.quantis or bundle.q_chapeu is None:
        raise ValueError("o pacote não tem quantis nem q̂: treine-o de novo com a M6")
    numero = prever_log(bundle, x, ref, degrau)
    alvo = bundle.config.get("alvo", "relativo")
    ref_x = pd.Series(0.0 if ref is None else float(ref), index=x.index)
    q, _ = prever_quantis(bundle.quantis, x, ref_x, SimpleNamespace(alvo=alvo))
    lo, hi = conter_o_numero(*intervalo(q[:, 0], q[:, 2], bundle.q_chapeu), numero)
    return numero, float(lo[0]), float(hi[0])


if __name__ == "__main__":
    # Rodado como `python -m previa.modelo`, este arquivo é o `__main__`, e o pickle do pacote
    # gravaria `__main__.Bundle`, que nenhum outro processo (o MCP, o índice) consegue carregar.
    # Chamar o `main` do módulo importado faz o pacote referenciar `previa.modelo.Bundle`.
    from previa import modelo as _modelo

    raise SystemExit(_modelo.main())
