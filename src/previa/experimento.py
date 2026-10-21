"""Experimentos rastreáveis (aula 2.5) com o GBM da Opção A (M4; aula 3.4).

Uma config YAML por experimento (`configs/experimentos/`), lida numa dataclass: uma chave com
erro de digitação levanta TypeError. Cada rodada vai para o MLflow (experimento
`previa-curtidas`, em `mlflow.db`) com a config, o commit, o hash da base e as métricas da
manchete (#5 e #6) em duas avaliações de desenvolvimento, sem abrir o teste:

- walk-forward: os trimestres de 2025, cada um previsto por um modelo treinado até 30 dias antes;
- canal novo no futuro: os canais em grupos; cada grupo é previsto em 2025 por um modelo treinado
  só com os outros canais, até 30 dias antes de 2025 (aula 2.3).

O GBM segue a 3.4: early stopping nos últimos `val_dias` do treino, refit no treino inteiro com o
nº de árvores escolhido (erro comum 1); setor como código 0/1, sem `channel_id` (erro comum 4) e
sem inscritos (princípio 2). As linhas de treino e de avaliação são as primeiras cópias com
curtidas desde `treino_inicio`; o histórico usa todos os posts. O baseline é a mediana madura
das mesmas features (`hist_mediana` com `hist_n` >= 3), como na leitura da M2.

Cada run guarda a data de corte da avaliação (tag `data_corte`, F22) e o `comparar` recusa runs com
cortes diferentes (2.5, erro comum 3). Cada avaliação traz também a importância por ganho das
features (F35): uma feature com mais da metade do ganho é alarme de vazamento (2.2).

Uso: uv run python -m previa.experimento configs/experimentos/opcao_a.yaml
     uv run python -m previa.experimento comparar RUN_A RUN_B   (ganho de B sobre A, pareado)
     uv run python -m previa.experimento painel RUN_ID   (o painel de métricas de um run salvo)
"""

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb  # antes do mlflow (2.5, erro comum 7)
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from previa.baselines import alvo
from previa.baselines import ler_config as ler_config_baselines
from previa.caminhos import CONFIGS
from previa.coletor import DADOS
from previa.dados import carregar_base
from previa.esquema import validar_tabela_de_treino
from previa.features import montar_features, quantis_do_canal
from previa.metricas import (
    f1_faixas_canal,
    ic_dif_mae_pareado,
    ic_ganho_pareado,
    ic_por_canal,
    mae_log,
    mape,
    pairwise_janela,
    par_a_par_peso_posts,
    rmse_log,
    spearman_global,
    spearman_intra_canal,
)
from previa.splits import _utc, canal_novo_no_futuro, folds_de_desenvolvimento, ler_avaliacao
from previa.texto import manuais

EXPERIMENTO = "previa-curtidas"
SETORES = ["automotivo", "beleza"]
FIXOS = {"verbosity": -1, "deterministic": True, "force_row_wise": True}  # 2.5: determinismo
NAO_METRICA = ("arvores", "previsoes", "dobras", "importancia")  # o que não é float (registrar)
LIMITE_IMPORTANCIA = 0.5  # 2.2: uma feature com mais da metade do ganho é "importância dominante"


@dataclass
class Config:
    nome: str
    hipotese: str
    descricao: str
    features: list[str]
    lacuna_dias: int = 30
    val_dias: int = 180
    seed: int = 0
    max_arvores: int = 5000
    paciencia: int = 100
    lgbm: dict = field(default_factory=dict)
    pca: dict = field(default_factory=dict)  # bloco de embedding -> nº de componentes (3.6, 4.4)
    alvo: str = "relativo"  # 05/10 (README, 1.6): log da razão com a mediana madura do canal
    min_posts_alvo: int = 10  # #20: posts maduros para o alvo relativo existir (1.6)
    teto_por_canal: int | None = None  # #8 (H6): no máximo N linhas de treino por canal
    idade_rotulo_max: int | None = None  # #21 (H5 b): só rótulos com até N dias de idade
    fixar: dict = field(default_factory=dict)  # #21 (H5 a): feature fixada na previsão
    # blocos por join (4.8): sondas_thumb, aparencia_thumb, vlm...
    extras: list[str] = field(default_factory=list)
    so_amostra: bool = False  # avaliar só nos Shorts da amostra multimodal (H8, 4.8)
    ruido: int = 0  # controle negativo (2.6): `ruido_1`..`ruido_N`, sem relação com o alvo


def ler_config(caminho) -> Config:
    return Config(**yaml.safe_load(Path(caminho).read_text(encoding="utf-8")))


def commit_atual() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        sujo = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "sem-git"
    return commit + ("+sujo" if sujo else "")  # commitar antes de rodar o que vai para o README


def hash_base(df: pd.DataFrame) -> str:
    """Impressão digital do conteúdo da base: muda se qualquer valor mudar (2.5)."""
    conteudo = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    return hashlib.sha256(conteudo).hexdigest()[:12]


def ler_duracoes() -> pd.Series:
    import duckdb

    with duckdb.connect(str(DADOS / "shorts.duckdb"), read_only=True) as con:
        v = con.execute("SELECT video_id, duration_s FROM videos").df()
    return v.set_index("video_id")["duration_s"]


def codigo_setor(setor: pd.Series) -> pd.Series:
    """0 = automotivo, 1 = beleza. Numérico, e não categórica: com dois setores os splits são os
    mesmos, e o SHAP intervencional (o fundo da #16, aula 5.6) não aceita splits categóricos."""
    codigo = setor.map({s: float(i) for i, s in enumerate(SETORES)})
    if codigo.isna().any():  # falha nunca silenciosa: "Beleza" ou "beauty" não viram ausente
        raise ValueError(f"setor fora de {SETORES}: {sorted(set(setor[codigo.isna()]))}")
    return codigo


def ler_extra(nome: str) -> pd.DataFrame:
    """Um bloco de features por post_id, só com colunas numéricas (bool vira 0/1; 4.8, erro 4)."""
    import json

    if nome == "sondas_thumb":
        t = pd.read_parquet(DADOS / "sondas_thumb_v1.parquet").set_index("video_id")
    elif nome == "aparencia_thumb":  # cor e brilho da capa (4.3), a parte OpenCV do bloco (#12)
        t = pd.read_parquet(DADOS / "aparencia_thumb_v1.parquet").set_index("video_id")
    elif nome == "vlm":  # a rubrica (#13); só os campos com kappa > 0,6 entram na config
        linhas = [json.loads(x) for x in (DADOS / "vlm" / "rubrica_v1.jsonl").open()]
        t = pd.DataFrame(linhas).drop_duplicates("post_id", keep="last").set_index("post_id")
        t = t[[c for c in t.columns if not c.startswith("_")]].add_prefix("vlm_")
    elif nome in ("sinais", "audio", "metadados"):
        pasta = {
            "sinais": "sinais_baixo_nivel_v1",
            "audio": "audio_v1",
            "metadados": "metadados_v1",
        }
        t = pd.read_parquet(DADOS / "cache" / f"{pasta[nome]}.parquet").set_index("video_id")
        if nome == "sinais":
            t = t.add_prefix("sinal_")
        elif nome == "metadados":
            t = t.add_prefix("meta_")
    else:
        raise ValueError(f"bloco extra desconhecido: {nome}")
    t = t.select_dtypes(include=["number", "bool"]).astype(float)
    t.index.name = "post_id"
    return t


def ler_embeddings_do_bloco(bloco: str) -> pd.DataFrame:
    """Os embeddings salvos pelos processos com torch (previa.texto e previa.visual)."""
    if bloco == "titulo":
        from previa.texto import ler_embeddings

        return ler_embeddings(DADOS / "emb_titulo_v1.parquet")
    if bloco == "thumb":
        from previa.visual import carregar_embeddings

        return carregar_embeddings(DADOS / "emb_thumb_v1")
    raise ValueError(f"bloco de embedding desconhecido: {bloco}")


def preparar(base, duracoes, cfg_av, lacuna_dias=30, min_posts=3, min_posts_alvo=10, extras=None):
    """A base com y, o alvo relativo, as features da M3, duração, setor, o baseline e `linha`.

    `ref` é a mediana madura do canal (30 dias, em log1p) e `n_ref` quantos posts ela usa; o
    baseline é `ref` com 3+ posts (#20, 2.1) e o alvo relativo `y_rel = y - ref` existe com 10+
    (#20, 1.6). A previsão de um modelo relativo é `r + ref`. `faixa_p25/p75/p95` são os quantis
    de y do mesmo histórico (30 dias, 10+ posts, #7, 1.6): a régua das faixas, nunca feature.
    """
    d = base.copy()
    d["y"] = alvo(d)
    d = d.join(montar_features(d, lacuna_dias))
    d["duracao_s"] = d["post_id"].map(duracoes).astype(float)
    d["setor"] = codigo_setor(d["sector"])
    if "caption" in d:  # as manuais do título (#9; 3.6)
        d = d.join(manuais(d["caption"]))
    for nome, bloco in (extras or {}).items():  # early fusion por post_id, NaN sem mídia (4.8)
        assert bloco.index.is_unique, f"{nome}: post_id duplicado"
        d = d.join(bloco.astype(float), on="post_id")
    d["linha"] = (
        (d["duplicate_group"] == d["post_id"])
        & d["y"].notna()
        & (d["published_at"] >= cfg_av["treino_inicio"])
    )
    maduro = (
        montar_features(d, 30) if lacuna_dias != 30 else d
    )  # a referência é sempre a de 30 dias
    d["ref"], d["n_ref"] = maduro["hist_mediana"], maduro["hist_n"]  # mediana madura do canal (log)
    d["p_base"] = d["ref"].where(d["n_ref"] >= min_posts)
    d["y_rel"] = (d["y"] - d["ref"]).where(d["n_ref"] >= min_posts_alvo)
    return d.join(quantis_do_canal(d, 30, min_posts_alvo))  # a régua das faixas (#7, 1.6)


def folds(d, cfg_av) -> dict:
    """Os pares (treino, teste) das duas avaliações, só com linhas marcadas."""
    linhas = d.index[d["linha"]]
    wf = [
        (tr.intersection(linhas), te.intersection(linhas))
        for tr, te in folds_de_desenvolvimento(d, cfg_av)
    ]
    inicio = _utc(cfg_av["walk_forward"]["inicio"])
    datas = d.loc[linhas, "published_at"]
    treino = linhas[datas < inicio - pd.Timedelta(days=cfg_av["gap_dias"])]
    teste = wf[0][1]
    for _, te in wf[1:]:
        teste = teste.union(te)
    cn = cfg_av["canal_novo"]
    novo = canal_novo_no_futuro(d, treino, teste, cn["n_grupos"], cn["seed"])
    return {"walk_forward": wf, "canal_novo": novo}


def aplicar_pca(emb: pd.DataFrame, ids_treino, ids_todos, n, prefixo, seed=0):
    """PCA ajustado só nos embeddings do treino e aplicado a todos (3.6 e 4.4, erro comum de ambas).

    Devolve as colunas `<prefixo>_pcNN` no índice `ids_todos` (NaN para quem não tem embedding) e o
    PCA ajustado, que a previsão reaplica nos Shorts novos.
    """
    no_treino = emb.index.intersection(pd.Index(ids_treino))
    pca = PCA(n_components=n, random_state=seed).fit(emb.loc[no_treino].to_numpy())
    com_emb = pd.Index(ids_todos).intersection(emb.index)
    cols = [f"{prefixo}_pc{k:02d}" for k in range(n)]
    saida = pd.DataFrame(np.nan, index=pd.Index(ids_todos), columns=cols)
    saida.loc[com_emb] = pca.transform(emb.loc[com_emb].to_numpy())
    return saida, pca


def expandir_features(features: list[str], colunas) -> list[str]:
    """`vlm_*` vira as colunas com esse prefixo, na ordem da tabela (4.8: um prefixo por bloco)."""
    saida = []
    for f in features:
        if f.endswith("*"):
            achadas = [c for c in colunas if c.startswith(f[:-1])]
            if not achadas:  # senão o tratamento vira o controle sem ninguém perceber
                raise ValueError(f"{f}: nenhuma coluna com esse prefixo (faltou o bloco extra?)")
            saida += achadas
        else:
            saida.append(f)
    return saida


def coluna_alvo(cfg: Config) -> str:
    return "y_rel" if cfg.alvo == "relativo" else "y"


def linhas_de_treino(d, idx, cfg: Config) -> pd.Index:
    """As linhas de `idx` com o alvo definido (no relativo, 10+ posts maduros no canal).

    Com `idade_rotulo_max`, só rótulos até essa idade (#21); com `teto_por_canal`, as N linhas
    mais recentes de cada canal (#8). O histórico das features continua com todos os posts.
    """
    sel = d.loc[idx]
    sel = sel[sel[coluna_alvo(cfg)].notna()]
    if cfg.idade_rotulo_max is not None:
        sel = sel[sel["target_age_days"] <= cfg.idade_rotulo_max]
    if cfg.teto_por_canal is not None:
        sel = (
            sel.sort_values("published_at", kind="stable")
            .groupby("channel_id")
            .tail(cfg.teto_por_canal)
        )
    return idx[idx.isin(sel.index)]


def fixar_na_previsao(x: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fixa features de controle na previsão (#21: a idade do rótulo vale 30 para o Short novo)."""
    return x.assign(**cfg.fixar) if cfg.fixar else x


def prever_absoluto(modelo, x: pd.DataFrame, ref: pd.Series, cfg: Config) -> np.ndarray:
    """log1p das curtidas: a saída do modelo, mais a mediana do canal se o alvo for relativo."""
    p = modelo.predict(x)
    return p + ref.to_numpy() if cfg.alvo == "relativo" else p


def treinar_gbm(d, idx, cfg: Config, features=None):
    """Early stopping nos últimos `val_dias` de `idx`; refit em `idx` inteiro (aula 3.4)."""
    features, y = features or cfg.features, coluna_alvo(cfg)
    tr = d.loc[linhas_de_treino(d, idx, cfg)].sort_values("published_at", kind="stable")
    corte = tr["published_at"].max() - pd.Timedelta(days=cfg.val_dias)
    ajuste, val = tr[tr["published_at"] <= corte], tr[tr["published_at"] > corte]
    params = {**cfg.lgbm, **FIXOS, "seed": cfg.seed}
    d_ajuste = lgb.Dataset(ajuste[features], ajuste[y])
    busca = lgb.train(
        params,
        d_ajuste,
        num_boost_round=cfg.max_arvores,
        valid_sets=[lgb.Dataset(val[features], val[y], reference=d_ajuste)],
        callbacks=[lgb.early_stopping(cfg.paciencia, verbose=False), lgb.log_evaluation(0)],
    )
    n = max(busca.best_iteration, 1)  # escolhido na validação temporal
    modelo = lgb.train(params, lgb.Dataset(tr[features], tr[y]), num_boost_round=n)
    return modelo, n


def _tabela_por_dobra(a: pd.DataFrame, col: str, dobras: pd.Series | None) -> pd.DataFrame:
    """Acertos e pares por canal, só com pares do mesmo fold (o mesmo modelo, como A e B da
    Lumina); as tabelas por fold somam, porque acertos e pares são aditivos."""
    if dobras is None:
        return pairwise_janela(a, col)
    partes = [pairwise_janela(g, col) for _, g in a.groupby(dobras.reindex(a.index).to_numpy())]
    return pd.concat(partes).groupby(level=0).sum()


def ganho_entre(d, te, p_a, p_b, n_boot=2000, seed=0, dobras=None) -> dict:
    """Ganho de B sobre A nas duas réguas (#27), pareado por canal, nas linhas com baseline (#5).

    A manchete da #6 (`ganho` > 0: B ordena melhor) e o erro em curtidas (`dif_mae_log` < 0: B erra
    menos o número), cada um com o IC 95% do bootstrap de canais.
    """
    ok = te[d.loc[te, "p_base"].notna().to_numpy()]
    a = d.loc[ok, ["channel_id", "published_at", "y"]].assign(
        pa=p_a.reindex(ok), pb=p_b.reindex(ok)
    )
    tab_a, tab_b = _tabela_por_dobra(a, "pa", dobras), _tabela_por_dobra(a, "pb", dobras)
    n = a.groupby("channel_id").size()
    ma, mb = par_a_par_peso_posts(tab_a, n), par_a_par_peso_posts(tab_b, n)
    lo, hi = ic_ganho_pareado(tab_b, tab_a, n, n_boot, seed)
    mae_a, mae_b = mae_log(a["y"], a["pa"]), mae_log(a["y"], a["pb"])
    mlo, mhi = ic_dif_mae_pareado(a["y"], a["pa"], a["pb"], a["channel_id"], n_boot, seed)
    return {
        "manchete_a": ma,
        "manchete_b": mb,
        "ganho": mb - ma,
        "ic_ganho_lo": float(lo),
        "ic_ganho_hi": float(hi),
        "mae_log_a": mae_a,
        "mae_log_b": mae_b,
        "dif_mae_log": mae_b - mae_a,
        "ic_dif_mae_lo": float(mlo),
        "ic_dif_mae_hi": float(mhi),
        "linhas": len(a),
    }


def _skill_mae(t) -> float:
    """1 - MAE do modelo / MAE do baseline (2.1, regra 3); NaN se o baseline não erra nada."""
    mae_base = mae_log(t["y"], t["p_base"])
    return 1 - mae_log(t["y"], t["p"]) / mae_base if mae_base > 0 else np.nan


def painel(a: pd.DataFrame, n_boot=2000, seed=0) -> dict:
    """O painel de métricas do TAPI 6.1 (aula 2.4), cada uma com o IC 95% de canais.

    `a` tem `channel_id, y, p, p_base, faixa_p25, faixa_p75, faixa_p95, dobra`, uma linha por Short
    avaliado (y, p e p_base em log1p, p absoluto). Devolve `mae_log`, `e_mae` (e^MAE, o fator
    típico de erro, com o IC de e^IC do MAE), `rmse_log`, `mape` (sobre o log, como a 2.4: em
    curtidas cruas ele não diz nada), `spearman_global`, `f1_faixas` (#7, só nas linhas com
    histórico) e `skill_mae` (1 - MAE do modelo / MAE do baseline, 2.1 regra 3), cada uma com
    `_lo` e `_hi`; e, por dobra k, `dobra{k}_mae_log` e `dobra{k}_spearman_intra` (só o valor: o
    IC de uma dobra com poucos canais não diria nada). A cobertura espera o intervalo (M6). NaN em
    y, p ou p_base levanta ValueError (princípio 7); o F1 sem nenhuma linha com histórico é NaN,
    com aviso.
    """
    for col in ("y", "p", "p_base", "dobra"):
        if a[col].isna().any():  # falha nunca silenciosa: NaN viraria erro zero ou métrica NaN
            raise ValueError(f"painel: {col} com NaN (filtre as linhas antes)")
    a = a[["channel_id", "y", "p", "p_base", "faixa_p25", "faixa_p75", "faixa_p95", "dobra"]]
    metricas = {
        "mae_log": lambda t: mae_log(t["y"], t["p"]),
        "rmse_log": lambda t: rmse_log(t["y"], t["p"]),
        "mape": lambda t: mape(t["y"], t["p"]),
        "spearman_global": lambda t: spearman_global(t["y"], t["p"]),
        "f1_faixas": lambda t: f1_faixas_canal(
            t["y"], t["p"], t["faixa_p25"], t["faixa_p75"], t["faixa_p95"]
        ),
        "skill_mae": _skill_mae,
    }
    saida = {}
    for nome, fn in metricas.items():
        lo, hi = ic_por_canal(fn, a, n_boot, seed)  # a mesma seed: os mesmos canais em todas
        saida |= {nome: fn(a), f"{nome}_lo": float(lo), f"{nome}_hi": float(hi)}
    if np.isnan(saida["f1_faixas"]):
        warnings.warn(
            "painel: nenhuma linha com histórico (10+ posts maduros): F1 das faixas indefinido",
            stacklevel=2,
        )
    for sufixo in ("", "_lo", "_hi"):
        saida[f"e_mae{sufixo}"] = float(np.exp(saida[f"mae_log{sufixo}"]))
    for k, g in a.groupby("dobra", sort=True):
        saida[f"dobra{int(k)}_mae_log"] = mae_log(g["y"], g["p"])
        saida[f"dobra{int(k)}_spearman_intra"] = spearman_intra_canal(
            g["y"], g["p"], g["channel_id"]
        )
    return saida


def _linhas_avaliadas(d, te, p_modelo, dobras=None) -> pd.DataFrame:
    """As linhas de `te` com baseline (#5), com a previsão, a dobra e a régua das faixas (#7)."""
    ok = te[d.loc[te, "p_base"].notna().to_numpy()]
    colunas = ["channel_id", "published_at", "y", "p_base", "faixa_p25", "faixa_p75", "faixa_p95"]
    dobra = 0 if dobras is None else dobras.reindex(ok)
    return d.loc[ok, colunas].assign(p=p_modelo.reindex(ok), dobra=dobra)


def avaliar(d, te, p_modelo, n_boot=2000, seed=0, dobras=None) -> dict:
    """A manchete da #6 para o modelo e o baseline, nas linhas em que o baseline existe.

    As chaves `painel_*` acrescentam o painel de métricas (TAPI 6.1; `painel`) nas mesmas linhas.
    """
    a = _linhas_avaliadas(d, te, p_modelo, dobras)
    tab_m, tab_b = _tabela_por_dobra(a, "p", dobras), _tabela_por_dobra(a, "p_base", dobras)
    n = a.groupby("channel_id").size()
    mm, mb = par_a_par_peso_posts(tab_m, n), par_a_par_peso_posts(tab_b, n)
    lo, hi = ic_ganho_pareado(tab_m, tab_b, n, n_boot, seed)
    return {
        "linhas": len(a),
        "canais": a["channel_id"].nunique(),
        "pares": int(tab_m["pares"].sum()),
        "manchete_modelo": mm,
        "manchete_base": mb,
        "ganho": mm - mb,
        "ic_ganho_lo": float(lo),
        "ic_ganho_hi": float(hi),
        "par_a_par_por_par_modelo": tab_m["acertos"].sum() / tab_m["pares"].sum(),
        "mae_log_modelo": mae_log(a["y"], a["p"]),
        "mae_log_base": mae_log(a["y"], a["p_base"]),
        "spearman_intra_modelo": spearman_intra_canal(a["y"], a["p"], a["channel_id"]),
        "spearman_intra_base": spearman_intra_canal(a["y"], a["p_base"], a["channel_id"]),
        **{f"painel_{k}": v for k, v in painel(a, n_boot, seed).items()},
    }


def importancia_por_ganho(modelos) -> pd.Series:
    """O ganho (`gain`) de cada feature, somado nos modelos das dobras e normalizado para somar 1.

    Em ordem decrescente, indexado pelo nome que o modelo guardou (aula 2.2: uma feature que
    concentra o ganho é a primeira suspeita de vazamento). Sem nenhum split no ganho total, a
    fatia não existe: ValueError (princípio 7), e não uma série de NaN.
    """
    ganho = pd.Series(dtype=float)
    for m in modelos:
        por_modelo = pd.Series(m.feature_importance("gain"), index=m.feature_name())
        ganho = ganho.add(por_modelo, fill_value=0.0)
    if not ganho.sum() > 0:
        raise ValueError("importância por ganho: nenhum split nos modelos (ganho total zero)")
    return (ganho / ganho.sum()).sort_values(ascending=False, kind="stable")


def avisar_importancia_dominante(resultado: dict, limite=LIMITE_IMPORTANCIA) -> None:
    """Avisa quando uma feature concentra mais de `limite` do ganho em alguma avaliação (2.2)."""
    for avaliacao, r in resultado.items():
        if r["importancia_maior_fatia"] > limite:
            warnings.warn(
                f"importância dominante em {avaliacao}: {r['importancia'].index[0]} tem "
                f"{r['importancia_maior_fatia']:.2f} do ganho (limite {limite}); "
                "confira vazamento (aula 2.2)",
                stacklevel=2,
            )


def rodar(
    cfg: Config,
    base,
    duracoes,
    cfg_av,
    n_boot=2000,
    seed=0,
    embeddings=None,
    extras=None,
    so_avaliar=None,
    min_posts=3,
) -> dict:
    """Treina e avalia nas duas avaliações; devolve as métricas e as árvores de cada modelo.

    `embeddings` (bloco -> DataFrame com índice post_id) alimenta os blocos de `cfg.pca`, com o
    PCA ajustado de novo no treino de cada fold. `extras` (bloco -> DataFrame com índice post_id)
    entra por join; `so_avaliar` (post_ids) restringe as linhas avaliadas, como na amostra (H8).
    Cada avaliação traz `importancia` (Series, ganho normalizado das features) e
    `importancia_maior_fatia` (float, a maior fatia; o alarme é `avisar_importancia_dominante`).
    """
    d = preparar(
        base,
        duracoes,
        cfg_av,
        cfg.lacuna_dias,
        min_posts=min_posts,
        min_posts_alvo=cfg.min_posts_alvo,
        extras=extras,
    )
    if cfg.ruido:  # o tamanho do acaso nesta avaliação (2.6, erro comum 2)
        ruido = np.random.default_rng(cfg.seed).normal(size=(len(d), cfg.ruido))
        for k in range(cfg.ruido):
            d[f"ruido_{k + 1}"] = ruido[:, k]
    base_feats = expandir_features(cfg.features, d.columns)
    validar_tabela_de_treino(d, base_feats)  # o contrato, antes de qualquer fit (F15; 1.8, 3.5)
    for nome, bloco in (extras or {}).items():  # senão o tratamento vira o controle (4.8, erro 4)
        if not set(bloco.columns) & set(base_feats):
            raise ValueError(f"bloco extra {nome}: nenhuma coluna dele está nas features da config")
    embeddings = embeddings or {}
    saida = {}
    for nome, pares in folds(d, cfg_av).items():
        p, arvores, todos = pd.Series(np.nan, index=d.index), [], d.index[:0]
        modelos = []
        id_dobra = pd.Series(-1, index=d.index)
        for k, (tr, te) in enumerate(pares):
            if so_avaliar is not None:
                te = te[d.loc[te, "post_id"].isin(so_avaliar).to_numpy()]
            dobra, feats = d.loc[tr.union(te)].copy(), list(base_feats)
            for bloco, n_comp in cfg.pca.items():
                cols, _ = aplicar_pca(
                    embeddings[bloco], d.loc[tr, "post_id"], dobra["post_id"], n_comp, bloco, seed
                )
                dobra[list(cols.columns)] = cols.to_numpy()
                feats += list(cols.columns)
            modelo, n = treinar_gbm(dobra, tr, cfg, feats)
            arvores.append(n)
            modelos.append(modelo)
            x_te = fixar_na_previsao(dobra.loc[te, feats], cfg)
            p.loc[te] = prever_absoluto(modelo, x_te, dobra.loc[te, "ref"], cfg)
            id_dobra.loc[te] = k
            todos = todos.union(te)
        importancia = importancia_por_ganho(modelos)  # F35, 2.2
        saida[nome] = {
            **avaliar(d, todos, p, n_boot, seed, dobras=id_dobra),
            "n_features": len(feats),
            "importancia": importancia,
            "importancia_maior_fatia": float(importancia.iloc[0]),
            "arvores": arvores,
            "previsoes": p.loc[todos].set_axis(d.loc[todos, "post_id"]),
            "dobras": id_dobra.loc[todos].set_axis(d.loc[todos, "post_id"]),
        }
    return saida


def tabela_de_previsoes(resultado) -> pd.DataFrame:
    """As previsões fora da amostra de um run (`post_id`, `p`, `dobra`, `avaliacao`)."""
    return pd.concat(
        [
            pd.DataFrame({"p": r["previsoes"], "dobra": r["dobras"]})
            .reset_index()
            .assign(avaliacao=a)
            for a, r in resultado.items()
        ]
    )


def ganhos(d, prev_a, prev_b, n_boot=2000, seed=0) -> pd.DataFrame:
    """Ganho de B sobre A em cada avaliação dos dois, pareado por canal, nas tabelas de previsões.

    Os pares da manchete ficam dentro do mesmo fold de A (os dois runs usam os mesmos folds).
    """
    linha_de = pd.Series(d.index, index=d["post_id"])
    saida = {}
    for avaliacao in sorted(set(prev_a["avaliacao"]) & set(prev_b["avaliacao"])):
        p = {}
        for lado, prev in (("a", prev_a), ("b", prev_b)):
            t = prev[prev["avaliacao"] == avaliacao]
            idx = linha_de.loc[t["post_id"]].to_numpy()
            p[lado] = pd.Series(t["p"].to_numpy(), index=idx)
            if lado == "a":
                dobras = pd.Series(t["dobra"].to_numpy(), index=idx)
        te = p["a"].index.intersection(p["b"].index)
        saida[avaliacao] = ganho_entre(d, te, p["a"], p["b"], n_boot, seed, dobras)
    return pd.DataFrame(saida)


def painel_das_previsoes(d, prev, n_boot=2000, seed=0) -> pd.DataFrame:
    """O painel de cada avaliação de um run, refeito das previsões salvas (sem treinar).

    `prev` é a `tabela_de_previsoes` do run (`post_id, p, dobra, avaliacao`) e `d` a base
    preparada (`preparar`). Uma coluna por avaliação, uma linha por chave do `painel`, sem o
    prefixo `painel_`; com os mesmos `n_boot` e `seed`, bate com o que o `rodar` registrou.
    """
    linha_de = pd.Series(d.index, index=d["post_id"])
    saida = {}
    for avaliacao in sorted(set(prev["avaliacao"])):
        t = prev[prev["avaliacao"] == avaliacao]
        idx = linha_de.loc[t["post_id"]].to_numpy()
        p = pd.Series(t["p"].to_numpy(), index=idx)
        dobras = pd.Series(t["dobra"].to_numpy(), index=idx)
        a = _linhas_avaliadas(d, p.index.sort_values(), p, dobras)
        saida[avaliacao] = painel(a, n_boot, seed)
    return pd.DataFrame(saida)


def painel_do_run(run_id: str, n_boot=2000, seed=0, min_posts=3) -> pd.DataFrame:
    """O painel de um run salvo no MLflow, nas previsões dele sobre a base de desenvolvimento.

    Não treina e não abre o teste (`carregar_base()` sem `final`). Como o `comparar`, exige que o
    run tenha usado a base atual (o hash da base).
    """
    mlflow.set_experiment(EXPERIMENTO)  # o load_table procura no experimento ativo
    base = carregar_base()
    do_run = mlflow.get_run(run_id).data.tags.get("hash_base")
    if do_run != hash_base(base):
        raise ValueError(f"o run não usou a base atual ({hash_base(base)}): {do_run}; rode de novo")
    prev = mlflow.load_table("previsoes.json", run_ids=[run_id])
    d = preparar(base, ler_duracoes(), ler_avaliacao(), min_posts=min_posts)
    return painel_das_previsoes(d, prev, n_boot, seed)


def avaliacao_crua() -> dict:
    """O `configs/avaliacao.yaml` como está agora, em dict JSON (as datas ficam como texto)."""
    cru = yaml.safe_load((CONFIGS / "avaliacao.yaml").read_text(encoding="utf-8"))
    return json.loads(json.dumps(cru, default=str))


def data_de_corte(cfg_av: dict) -> str:
    """A data de corte da avaliação de desenvolvimento: o início do walk-forward (AAAA-MM-DD)."""
    return str(cfg_av["walk_forward"]["inicio"].date())


def conferir_cortes(tags: dict[str, dict], corte_atual: str) -> None:
    """Recusa comparar runs com datas de corte diferentes entre si ou da atual (2.5, erro 3; F22).

    `tags` é run_id -> tags do run. Run lacrado antes da F22 não tem `data_corte`: avisa e segue,
    conferido só pelo hash da base. Cortes diferentes levantam ValueError.
    """
    cortes = {}
    for run, t in tags.items():
        if "data_corte" in t:
            cortes[run] = t["data_corte"]
        else:
            warnings.warn(
                f"run {run} sem data_corte, anterior à F22: conferido só pelo hash da base",
                stacklevel=2,
            )
    if len(set(cortes.values()) | {corte_atual}) > 1:
        raise ValueError(
            f"os runs têm datas de corte diferentes (data_corte): {cortes}; atual {corte_atual}. "
            "Rode de novo com o mesmo corte (aula 2.5, erro comum 3)"
        )


def registrar(cfg: Config, caminho, resultado, base, tags=None, cfg_av=None) -> str:
    """Um run no MLflow: tags de rastreio, params, a config inteira e as métricas.

    Grava a data de corte (`data_corte`) e o `gap_dias` do `cfg_av` (sem ele, o do projeto), o
    `avaliacao.yaml` cru (`avaliacao.json`) e a importância por ganho (`importancia_gain.json`).
    """
    cfg_av = ler_avaliacao() if cfg_av is None else cfg_av
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=cfg.nome, description=cfg.descricao) as run:
        mlflow.set_tags(
            {
                "git_commit": commit_atual(),
                "hash_base": hash_base(base),
                "hipotese": cfg.hipotese,
                "config": str(caminho),
                "avaliacao": "desenvolvimento (2025)",
                "data_corte": data_de_corte(cfg_av),
                "gap_dias": cfg_av["gap_dias"],
                **(tags or {}),
            }
        )
        fora = ("lgbm", "features", "descricao")
        params = {k: v for k, v in asdict(cfg).items() if k not in fora}
        params |= {f"lgbm_{k}": v for k, v in cfg.lgbm.items()}
        params |= {"features": ",".join(cfg.features)}  # o nº expandido vai nas métricas
        params |= {f"{a}_arvores": ",".join(map(str, r["arvores"])) for a, r in resultado.items()}
        mlflow.log_params(params)
        mlflow.log_dict(asdict(cfg), "config.json")
        mlflow.log_dict(avaliacao_crua(), "avaliacao.json")
        mlflow.log_dict(
            {a: r["importancia"].to_dict() for a, r in resultado.items()}, "importancia_gain.json"
        )
        for a, r in resultado.items():
            mlflow.log_metrics({f"{a}_{k}": float(v) for k, v in r.items() if k not in NAO_METRICA})
        # para comparar modelos depois (#5, H2 a H8)
        mlflow.log_table(tabela_de_previsoes(resultado), "previsoes.json")
        mlflow.log_params({f"pca_{b}": n for b, n in cfg.pca.items()})
        return run.info.run_id


def comparar(run_a: str, run_b: str, n_boot=2000, seed=0, min_posts=3) -> pd.DataFrame:
    """Ganho do run B sobre o run A, pareado por canal, nas previsões salvas de cada um."""
    mlflow.set_experiment(EXPERIMENTO)  # o load_table procura no experimento ativo
    base = carregar_base()
    cfg_av = ler_avaliacao()
    tags = {r: dict(mlflow.get_run(r).data.tags) for r in (run_a, run_b)}
    conferir_cortes(tags, data_de_corte(cfg_av))  # antes do hash (F22; 2.5, erro comum 3)
    hashes = {r: t.get("hash_base") for r, t in tags.items()}
    if len(set(hashes.values()) | {hash_base(base)}) != 1:
        raise ValueError(
            f"os runs não usaram a base atual ({hash_base(base)}): {hashes}; rode os dois de novo"
        )
    prev = {r: mlflow.load_table("previsoes.json", run_ids=[r]) for r in (run_a, run_b)}
    d = preparar(base, ler_duracoes(), cfg_av, min_posts=min_posts)
    return ganhos(d, prev[run_a], prev[run_b], n_boot, seed)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["comparar"]:  # uv run python -m previa.experimento comparar RUN_A RUN_B
        cfg_b = ler_config_baselines()
        r = comparar(argv[1], argv[2], cfg_b["n_boot"], cfg_b["seed"], cfg_b["min_posts"])
        print(r.round(4).to_string())
        print(f"\ncomando: uv run python -m previa.experimento comparar {argv[1]} {argv[2]}")
        return 0
    if argv[:1] == ["painel"]:  # uv run python -m previa.experimento painel RUN_ID
        cfg_b = ler_config_baselines()
        r = painel_do_run(argv[1], cfg_b["n_boot"], cfg_b["seed"], cfg_b["min_posts"])
        print(r.round(4).to_string())
        print(f"\ncomando: uv run python -m previa.experimento painel {argv[1]}")
        return 0
    parser = argparse.ArgumentParser(prog="python -m previa.experimento")
    parser.add_argument("config")
    args = parser.parse_args(argv)
    cfg, cfg_b, cfg_av = ler_config(args.config), ler_config_baselines(), ler_avaliacao()
    base = carregar_base()  # sem o período de teste
    embeddings = {b: ler_embeddings_do_bloco(b) for b in cfg.pca}
    extras = {e: ler_extra(e) for e in cfg.extras}
    amostra = None
    if cfg.so_amostra:
        amostra = set(pd.read_csv(DADOS / "amostra_multimodal.csv")["post_id"])
    resultado = rodar(
        cfg,
        base,
        ler_duracoes(),
        cfg_av,
        cfg_b["n_boot"],
        cfg_b["seed"],
        embeddings,
        extras,
        amostra,
        cfg_b["min_posts"],
    )
    tabela = pd.DataFrame(
        {a: {k: v for k, v in r.items() if k not in NAO_METRICA} for a, r in resultado.items()}
    )
    print(tabela.round(4).to_string())
    avisar_importancia_dominante(resultado)  # F35: uma feature com mais da metade do ganho
    run_id = registrar(cfg, args.config, resultado, base, cfg_av=cfg_av)
    print(f"\nrun {run_id} · commit {commit_atual()} · base {hash_base(base)}")
    print(f"comando: uv run python -m previa.experimento {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
