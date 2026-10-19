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

Uso: uv run python -m previa.experimento configs/experimentos/opcao_a.yaml
     uv run python -m previa.experimento comparar RUN_A RUN_B   (ganho de B sobre A, pareado)
"""

import argparse
import hashlib
import subprocess
import sys
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
from previa.coletor import DADOS
from previa.dados import carregar_base
from previa.features import montar_features
from previa.metricas import (
    ic_ganho_pareado,
    mae_log,
    pairwise_janela,
    par_a_par_peso_posts,
    spearman_intra_canal,
)
from previa.splits import _utc, canal_novo_no_futuro, folds_de_desenvolvimento, ler_avaliacao
from previa.texto import manuais

EXPERIMENTO = "previa-curtidas"
SETORES = ["automotivo", "beleza"]
FIXOS = {"verbosity": -1, "deterministic": True, "force_row_wise": True}  # 2.5: determinismo
NAO_METRICA = ("arvores", "previsoes", "dobras")


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
    extras: list[str] = field(default_factory=list)  # blocos por join (4.8): sondas_thumb, vlm...
    so_amostra: bool = False  # avaliar só nos Shorts da amostra multimodal (H8, 4.8)


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
    (#20, 1.6). A previsão de um modelo relativo é `r + ref`.
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
    return d


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
    """Ganho de B sobre A na manchete da #6, pareado por canal, nas linhas com baseline (#5)."""
    ok = te[d.loc[te, "p_base"].notna().to_numpy()]
    a = d.loc[ok, ["channel_id", "published_at", "y"]].assign(
        pa=p_a.reindex(ok), pb=p_b.reindex(ok)
    )
    tab_a, tab_b = _tabela_por_dobra(a, "pa", dobras), _tabela_por_dobra(a, "pb", dobras)
    n = a.groupby("channel_id").size()
    ma, mb = par_a_par_peso_posts(tab_a, n), par_a_par_peso_posts(tab_b, n)
    lo, hi = ic_ganho_pareado(tab_b, tab_a, n, n_boot, seed)
    return {
        "manchete_a": ma,
        "manchete_b": mb,
        "ganho": mb - ma,
        "ic_ganho_lo": float(lo),
        "ic_ganho_hi": float(hi),
        "linhas": len(a),
    }


def avaliar(d, te, p_modelo, n_boot=2000, seed=0, dobras=None) -> dict:
    """A manchete da #6 para o modelo e o baseline, nas linhas em que o baseline existe."""
    ok = te[d.loc[te, "p_base"].notna().to_numpy()]
    a = d.loc[ok, ["channel_id", "published_at", "y", "p_base"]].assign(p=p_modelo.reindex(ok))
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
    }


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
    base_feats = expandir_features(cfg.features, d.columns)
    embeddings = embeddings or {}
    saida = {}
    for nome, pares in folds(d, cfg_av).items():
        p, arvores, todos = pd.Series(np.nan, index=d.index), [], d.index[:0]
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
            x_te = fixar_na_previsao(dobra.loc[te, feats], cfg)
            p.loc[te] = prever_absoluto(modelo, x_te, dobra.loc[te, "ref"], cfg)
            id_dobra.loc[te] = k
            todos = todos.union(te)
        saida[nome] = {
            **avaliar(d, todos, p, n_boot, seed, dobras=id_dobra),
            "n_features": len(feats),
            "arvores": arvores,
            "previsoes": p.loc[todos].set_axis(d.loc[todos, "post_id"]),
            "dobras": id_dobra.loc[todos].set_axis(d.loc[todos, "post_id"]),
        }
    return saida


def registrar(cfg: Config, caminho, resultado, base) -> str:
    """Um run no MLflow: tags de rastreio, params, a config inteira e as métricas."""
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=cfg.nome, description=cfg.descricao) as run:
        mlflow.set_tags(
            {
                "git_commit": commit_atual(),
                "hash_base": hash_base(base),
                "hipotese": cfg.hipotese,
                "config": str(caminho),
                "avaliacao": "desenvolvimento (2025)",
            }
        )
        fora = ("lgbm", "features", "descricao")
        params = {k: v for k, v in asdict(cfg).items() if k not in fora}
        params |= {f"lgbm_{k}": v for k, v in cfg.lgbm.items()}
        params |= {"features": ",".join(cfg.features)}  # o nº expandido vai nas métricas
        params |= {f"{a}_arvores": ",".join(map(str, r["arvores"])) for a, r in resultado.items()}
        mlflow.log_params(params)
        mlflow.log_dict(asdict(cfg), "config.json")
        for a, r in resultado.items():
            mlflow.log_metrics({f"{a}_{k}": float(v) for k, v in r.items() if k not in NAO_METRICA})
        previsoes = pd.concat(
            [
                pd.DataFrame({"p": r["previsoes"], "dobra": r["dobras"]})
                .reset_index()
                .assign(avaliacao=a)
                for a, r in resultado.items()
            ]
        )
        mlflow.log_table(previsoes, "previsoes.json")  # para comparar modelos depois (#5, H2 a H8)
        mlflow.log_params({f"pca_{b}": n for b, n in cfg.pca.items()})
        return run.info.run_id


def comparar(run_a: str, run_b: str, n_boot=2000, seed=0, min_posts=3) -> pd.DataFrame:
    """Ganho do run B sobre o run A, pareado por canal, nas previsões salvas de cada um."""
    mlflow.set_experiment(EXPERIMENTO)  # o load_table procura no experimento ativo
    base = carregar_base()
    hashes = {r: mlflow.get_run(r).data.tags.get("hash_base") for r in (run_a, run_b)}
    if len(set(hashes.values()) | {hash_base(base)}) != 1:
        raise ValueError(
            f"os runs não usaram a base atual ({hash_base(base)}): {hashes}; rode os dois de novo"
        )
    prev = {r: mlflow.load_table("previsoes.json", run_ids=[r]) for r in (run_a, run_b)}
    d = preparar(base, ler_duracoes(), ler_avaliacao(), min_posts=min_posts)
    linha_de = pd.Series(d.index, index=d["post_id"])
    saida = {}
    for avaliacao in sorted(set(prev[run_a]["avaliacao"]) & set(prev[run_b]["avaliacao"])):
        p = {}
        for r in (run_a, run_b):
            t = prev[r][prev[r]["avaliacao"] == avaliacao]
            idx = linha_de.loc[t["post_id"]].to_numpy()
            p[r] = pd.Series(t["p"].to_numpy(), index=idx)
            if r == run_a:
                dobras = pd.Series(t["dobra"].to_numpy(), index=idx)
        te = p[run_a].index.intersection(p[run_b].index)
        saida[avaliacao] = ganho_entre(d, te, p[run_a], p[run_b], n_boot, seed, dobras)
    return pd.DataFrame(saida)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["comparar"]:  # uv run python -m previa.experimento comparar RUN_A RUN_B
        cfg_b = ler_config_baselines()
        r = comparar(argv[1], argv[2], cfg_b["n_boot"], cfg_b["seed"], cfg_b["min_posts"])
        print(r.round(4).to_string())
        print(f"\ncomando: uv run python -m previa.experimento comparar {argv[1]} {argv[2]}")
        return 0
    parser = argparse.ArgumentParser(prog="python -m previa.experimento")
    parser.add_argument("config")
    args = parser.parse_args(argv)
    cfg, cfg_b = ler_config(args.config), ler_config_baselines()
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
        ler_avaliacao(),
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
    run_id = registrar(cfg, args.config, resultado, base)
    print(f"\nrun {run_id} · commit {commit_atual()} · base {hash_base(base)}")
    print(f"comando: uv run python -m previa.experimento {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
