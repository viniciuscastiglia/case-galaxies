"""O teste final (F16; aulas 2.3, erro comum 4, e 2.5, erro comum 4: o teste abre uma vez).

Treina no bloco de treino (até `cal_inicio` - gap) e prevê o teste (`teste_inicio` a `teste_fim`,
fim exclusivo), nos dois cenários da #5: o `temporal` (um modelo, o futuro de todos os canais) e o
`canal_novo` (os canais em grupos, cada um novo uma vez, no mesmo teste). A calibração fica de
fora do treino: é da M6. As métricas saem do runner (`rodar`, `avaliar`, `registrar`): nada é
recalculado aqui, para a régua ser a do desenvolvimento.

O teste só abre com `--abrir-o-teste`. O registro (`data/logs/teste_aberto.log`, escrito por
`carregar_base(final=True)`) prova quantas vezes foi olhado: com uma abertura anterior, o comando
recusa e mostra a primeira, a menos que se passe `--reabrir`. Manchete ou Spearman intra-canal
acima do limite de `configs/avaliacao.yaml` (`final.alarme_*`) é "bom demais": procure vazamento
(hub, troubleshooting).

Uso: uv run python -m previa.final configs/experimentos/CONFIG.yaml --abrir-o-teste [--reabrir]
"""

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from previa.baselines import ler_config as ler_config_baselines
from previa.caminhos import CONFIGS
from previa.coletor import DADOS
from previa.dados import carregar_base
from previa.experimento import (
    avisar_importancia_dominante,
    commit_atual,
    expandir_features,
    fixar_na_previsao,
    hash_base,
    ler_config,
    ler_duracoes,
    ler_embeddings_do_bloco,
    ler_extra,
    linhas_de_treino,
    matriz_da_dobra,
    registrar,
    rodar,
    tabela_da_config,
)
from previa.flag import avaliar_flag, confidence_flag, limiares_da_calibracao
from previa.flag import carregar_cfg as ler_cfg_flag
from previa.intervalo import (
    cobertura,
    conter_o_numero,
    escores_cqr,
    faixa_de_posts,
    intervalo,
    prever_quantis,
    q_chapeu,
    treinar_quantis,
)
from previa.intervalo import ler_config as ler_cfg_intervalo
from previa.splits import canal_novo_no_futuro, ler_avaliacao, split_treino_cal_teste

REGISTRO = DADOS / "logs" / "teste_aberto.log"
LOG_FINAL = DADOS / "logs" / "teste_final.log"
CHAVES_FINAL = ("ganho_minimo", "cobertura_min_extra", "alarme_manchete", "alarme_spearman")


def limites(cfg_av: dict) -> dict:
    """Os quatro limites de `final:` em configs/avaliacao.yaml; sem a chave, erro (sem padrão)."""
    f = cfg_av.get("final")
    faltam = [k for k in CHAVES_FINAL if not isinstance(f, dict) or k not in f]
    if faltam:
        raise ValueError(
            f"configs/avaliacao.yaml: falta final.{', final.'.join(faltam)} "
            "(ganho_minimo, cobertura_min_extra, alarme_manchete e alarme_spearman; F16)"
        )
    return {k: float(f[k]) for k in CHAVES_FINAL}


def caminho_da_base(cfg_av: dict) -> Path:
    """A base congelada que `carregar_base` abriria (a chave `base` do avaliacao.yaml)."""
    return DADOS / cfg_av["base"]


def conferir_cobertura(caminho_base, cfg_av: dict, obrigatorios: dict, extras: dict) -> None:
    """Antes de abrir: cobertura dos blocos nos posts canônicos do período de teste.

    Só os canônicos (`duplicate_group == post_id`) são avaliados. Lê `post_id`, `published_at` e
    `duplicate_group` do arquivo, sem curtidas. Duração e o embedding do título (calculados para
    todo post) exigem 100%; o embedding `thumb` e cada extra, `final.cobertura_min_extra` (NaN sem
    mídia é legítimo). A cobertura
    desses blocos é sempre impressa. Falhou: `ValueError`, e o teste não é aberto.
    """
    minimo = limites(cfg_av)["cobertura_min_extra"]
    ids = pd.read_parquet(caminho_base, columns=["post_id", "published_at", "duplicate_group"])
    datas = ids["published_at"]
    na_janela = (datas >= cfg_av["teste_inicio"]) & (datas < cfg_av["teste_fim"])
    canonicos = ids.loc[na_janela & (ids["duplicate_group"] == ids["post_id"]), "post_id"]
    n = len(canonicos)
    # o embedding da capa vale como um extra: as sondas vêm dele e a imagem pode faltar (99%)
    parciais = {k: obrigatorios[k] for k in obrigatorios if k == "thumb"}
    faltam = {
        k: int((~canonicos.isin(b.index)).sum())
        for k, b in obrigatorios.items()
        if k not in parciais
    }
    faltam = {k: v for k, v in faltam.items() if v}
    ruins = []
    for nome, bloco, rotulo in [
        *((k, b, "do embedding") for k, b in parciais.items()),
        *((k, b, "do extra") for k, b in extras.items()),
    ]:
        tem = int(canonicos.isin(bloco.dropna(how="all").index).sum())
        pct = tem / n if n else 0.0
        print(f"cobertura {rotulo} {nome}: {tem}/{n} ({pct:.1%}) nos posts do teste")
        if pct < minimo:
            ruins.append(f"{nome} {pct:.1%} < {minimo:.1%}")
    if faltam or ruins:
        raise ValueError(
            f"o teste NÃO foi aberto: de {n} posts canônicos do período de teste, faltam em "
            f"blocos obrigatórios {faltam}; extras abaixo do mínimo: {ruins}. "
            "Gere os blocos e rode de novo."
        )


def folds_finais(d, cfg_av) -> dict:
    """Os pares (treino, teste) do teste final, só com linhas marcadas (`d["linha"]`).

    O treino vai até `cal_inicio` - gap; o teste é o bloco de `split_treino_cal_teste`. A
    calibração (a M6) não entra no treino. No `canal_novo`, os canais do grupo saem do treino.
    """
    linhas = d.index[d["linha"]]
    treino, _, teste = split_treino_cal_teste(d, cfg_av)
    treino, teste = treino.intersection(linhas), teste.intersection(linhas)
    cn = cfg_av["canal_novo"]
    return {
        "temporal": [(treino, teste)],
        "canal_novo": canal_novo_no_futuro(d, treino, teste, cn["n_grupos"], cn["seed"]),
    }


def avaliar_final(
    cfg, base, duracoes, cfg_av, n_boot=2000, seed=0, embeddings=None, extras=None, min_posts=3
) -> dict:
    """`rodar` com os folds do teste final: o mesmo treino e a mesma métrica do desenvolvimento.

    A cobertura do intervalo no teste vem de `avaliar_intervalo`, na mesma abertura.
    """
    return rodar(
        cfg,
        base,
        duracoes,
        cfg_av,
        n_boot,
        seed,
        embeddings,
        extras,
        None,
        min_posts,
        folds_fn=folds_finais,
    )


def calibracao_valida(d, cal, treino, cfg_av) -> pd.Index:
    """As linhas de calibração: `linha`, em [cal_inicio, teste_inicio), com `p_base` (n_ref >= 3).

    São as mesmas linhas que `avaliar` corrige (5.3, erros 2 e 3). Nenhuma pode estar no treino:
    o q̂ vem de posts maduros que o modelo não viu. Falhou: `ValueError`.
    """
    datas = d.loc[cal, "published_at"]
    ok = (
        d.loc[cal, "linha"]
        & (datas >= cfg_av["cal_inicio"])
        & (datas < cfg_av["teste_inicio"])
        & d.loc[cal, "p_base"].notna()
    )
    cal = cal[ok.to_numpy()]
    vazou = cal.intersection(treino)
    if len(vazou):
        raise ValueError(
            f"{len(vazou)} linhas de calibração estão no treino (5.3, erro comum 3): "
            "o q̂ ficaria otimista"
        )
    return cal


def _pinball(y, q, alfa) -> float:
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(alfa * e, (alfa - 1) * e)))


def _tabela_cobertura(y, lo, hi, d_te, cfg_q) -> pd.DataFrame:
    """Total, por setor, por faixa de posts anteriores (n_ref) e por mês de publicação (F41)."""
    grupos = {
        "setor": d_te["sector"],
        "faixa_posts": faixa_de_posts(d_te["n_ref"], cfg_q),
        "mes": d_te["published_at"].dt.strftime("%Y-%m"),
    }
    partes = []
    total = cobertura(y, lo, hi, None).reset_index(names="grupo").assign(dimensao="total")
    partes.append(total)
    for dim, g in grupos.items():
        t = cobertura(y, lo, hi, g.reset_index(drop=True))
        t = t.drop(index="total").reset_index(names="grupo").assign(dimensao=dim)
        partes.append(t)
    saida = pd.concat(partes, ignore_index=True)
    saida["n"] = saida["n"].astype(int)
    return saida[["dimensao", "grupo", "n", "cobertura", "erro_padrao", "largura"]]


def avaliar_intervalo(cfg, base, duracoes, cfg_av, cfg_q, cfg_flag, previsoes, **kw) -> dict:
    """A M6 no teste: quantis, q̂ da calibração, cobertura do CQR e o flag (F16, F41; 5.2 a 5.4).

    Só o cenário temporal, com a MESMA matriz de features do modelo pontual. `previsoes` é a
    série do pontual (log1p, índice post_id), o número da tool (#40): o intervalo é alargado, se
    preciso, para contê-lo. O q̂ vem só da calibração; o teste nunca entra nele. `kw`: embeddings,
    extras, min_posts (como em `avaliar_final`).
    """
    embeddings, extras, min_posts = kw.get("embeddings"), kw.get("extras"), kw.get("min_posts", 3)
    seed = kw.get("seed", 0)
    d = tabela_da_config(cfg, base, duracoes, cfg_av, extras, min_posts)
    base_feats = expandir_features(cfg.features, d.columns)
    ((treino, teste),) = folds_finais(d, cfg_av)["temporal"]
    datas = d["published_at"]
    cal = d.index[
        (d["linha"] & (datas >= cfg_av["cal_inicio"]) & (datas < cfg_av["teste_inicio"])).to_numpy()
    ]
    cal = calibracao_valida(d, cal, treino, cfg_av)
    teste = teste[d.loc[teste, "p_base"].notna().to_numpy()]  # as linhas que `avaliar` corrige
    embeddings = embeddings or {}
    dobra_p, colunas_p = matriz_da_dobra(
        d, treino, treino.union(teste), cfg, base_feats, embeddings, seed
    )
    dobra, feats = matriz_da_dobra(
        d, treino, treino.union(cal).union(teste), cfg, base_feats, embeddings, seed
    )
    if feats != colunas_p:
        raise ValueError(f"colunas dos quantis diferem do pontual: {feats} contra {colunas_p}")
    modelos = treinar_quantis(dobra, treino, cfg, feats, cfg_q)
    alfas = sorted(modelos)

    def prever(idx):
        x = fixar_na_previsao(dobra.loc[idx, feats], cfg)
        return prever_quantis(modelos, x, dobra.loc[idx, "ref"], cfg)

    q_cal, cruz_cal = prever(cal)
    q_te, cruz_te = prever(teste)
    qhat = q_chapeu(escores_cqr(q_cal[:, 0], q_cal[:, 2], dobra.loc[cal, "y"]), cfg_q.alpha_cqr)
    if not np.isfinite(qhat):
        raise ValueError(
            f"q̂ infinito: a calibração tem {len(cal)} linhas, poucas para alpha_cqr="
            f"{cfg_q.alpha_cqr} (k > n; 5.3). O intervalo não foi calculado."
        )
    y = dobra.loc[teste, "y"].to_numpy()
    numero = previsoes.reindex(dobra.loc[teste, "post_id"]).to_numpy()
    lo_c, hi_c = intervalo(q_te[:, 0], q_te[:, 2], qhat)
    lo_c, hi_c = conter_o_numero(lo_c, hi_c, numero)
    d_te = dobra.loc[teste]
    tab_bruto = _tabela_cobertura(y, q_te[:, 0], q_te[:, 2], d_te, cfg_q)
    tab_cqr = _tabela_cobertura(y, lo_c, hi_c, d_te, cfg_q)
    tot = lambda t: t[t["dimensao"] == "total"].iloc[0]  # noqa: E731
    m = {
        "intervalo_qhat": float(qhat),
        "intervalo_n_cal": len(cal),
        "intervalo_n_teste": len(teste),
        "intervalo_cruzamentos_cal": cruz_cal,
        "intervalo_cruzamentos_teste": cruz_te,
        "intervalo_bruto_cobertura": float(tot(tab_bruto)["cobertura"]),
        "intervalo_bruto_largura": float(tot(tab_bruto)["largura"]),
        "intervalo_cqr_cobertura": float(tot(tab_cqr)["cobertura"]),
        "intervalo_cqr_largura": float(tot(tab_cqr)["largura"]),
    }
    for j, a in enumerate(alfas):
        nome = f"q{round(a * 100)}"
        m[f"intervalo_abaixo_{nome}"] = float(np.mean(y < q_te[:, j]))
        m[f"intervalo_pinball_{nome}"] = _pinball(y, q_te[:, j], a)
    for _, r in tab_cqr[tab_cqr["dimensao"] == "setor"].iterrows():
        m[f"intervalo_cqr_cobertura_{r['grupo']}"] = float(r["cobertura"])
    # o flag (5.4): limiares da calibração, sem kNN e sem extração (dist_knn=None, falhou=[])
    largura_cal = pd.Series(q_cal[:, 2] - q_cal[:, 0] + 2 * qhat)
    ultimo_treino = d.loc[linhas_de_treino(d, treino, cfg), "published_at"].max()
    limiares = limiares_da_calibracao(largura_cal, None, ultimo_treino, cfg_flag)
    motivos = []
    for (_, r), lo, hi in zip(d_te.iterrows(), lo_c, hi_c, strict=True):
        f = confidence_flag(
            float(hi - lo), int(r["n_ref"]), [], None, r["published_at"].to_pydatetime(), limiares
        )
        motivos.append([x.codigo for x in f.motivos])
    tab_flag = avaliar_flag(y, numero, motivos)
    for _, r in tab_flag.iterrows():
        sufixo = "" if r["grupo"] == "sinalizados" else f"_{r['grupo']}"
        m[f"flag_taxa{sufixo}"] = float(r["taxa"])
        if r["n"]:  # um grupo vazio não tem MAE: não é falha
            m[f"flag_mae_{r['grupo']}"] = float(r["mae"])
    return {
        "metricas": {k: float(v) for k, v in m.items() if np.isfinite(v)},
        "nao_finitas": [k for k, v in m.items() if not np.isfinite(v)],
        "tabelas": {"cobertura_bruto": tab_bruto, "cobertura_cqr": tab_cqr, "flag": tab_flag},
        "limiares": limiares,
        "colunas": feats,
        "colunas_pontual": colunas_p,
    }


def registrar_intervalo(run_id: str, iv: dict) -> None:
    """Acrescenta ao MESMO run do teste as métricas `intervalo_*` e `flag_*` e as tabelas."""
    nao_finitas = iv.get("nao_finitas", [])
    if nao_finitas:
        warnings.warn(
            f"métrica não finita fora do registro do intervalo: {', '.join(nao_finitas)}",
            stacklevel=2,
        )
    with mlflow.start_run(run_id=run_id):
        if nao_finitas:
            mlflow.set_tag("metricas_nao_finitas", ",".join(nao_finitas))
        mlflow.log_metrics(iv["metricas"])
        mlflow.log_dict(
            {k: t.to_dict(orient="records") for k, t in iv["tabelas"].items()},
            "intervalo_tabelas.json",
        )
        mlflow.log_dict(iv["limiares"], "intervalo_limiares.json")


def _imprimir_intervalo(iv: dict) -> None:
    m = iv["metricas"]
    print(
        f"\nintervalo (CQR, q̂ {m['intervalo_qhat']:.3f} em log, {m['intervalo_n_cal']} da "
        f"calibração): cobertura {m['intervalo_cqr_cobertura']:.1%} (alvo 80%) · crua "
        f"{m['intervalo_bruto_cobertura']:.1%} · largura {m['intervalo_cqr_largura']:.2f} em log · "
        f"{m['intervalo_n_teste']} posts · cruzamentos {m['intervalo_cruzamentos_teste']}"
    )
    print(iv["tabelas"]["cobertura_cqr"].to_string(index=False, float_format="%.3f"))
    print("\nflag (taxa e MAE por motivo; 5.4):")
    print(iv["tabelas"]["flag"].to_string(index=False, float_format="%.3f"))


def _guardar_falha(erro: BaseException, resultado: dict | None, log=None) -> None:
    """Uma linha em `teste_final.log`: quando, a exceção e, se já existem, as métricas do pontual.

    Os números sobrevivem a uma falha do MLflow ou do disco depois da abertura (o teste não
    reabre). O próprio log falhar não pode mascarar a exceção original.
    """
    log = Path(LOG_FINAL if log is None else log)
    metricas = None
    if resultado is not None:
        metricas = {
            a: {k: float(v) for k, v in r.items() if isinstance(v, int | float | np.number)}
            for a, r in resultado.items()
        }
    agora = datetime.now(UTC).isoformat(timespec="seconds")
    linha = f"{agora} {type(erro).__name__}: {erro}"
    if metricas is not None:
        linha += " | " + json.dumps(metricas, ensure_ascii=False)
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(linha.replace("\n", " ") + "\n")
    except OSError as e:  # sem onde gravar: ao menos o terminal
        print(f"ERRO ao gravar {log}: {e}\n{linha}", file=sys.stderr)


def ja_aberto(registro=REGISTRO) -> list[str]:
    """As linhas do registro de aberturas do teste (vazio se nunca abriu)."""
    registro = Path(registro)
    if not registro.exists():
        return []
    return [x for x in registro.read_text(encoding="utf-8").splitlines() if x.strip()]


def criterio(resultado: dict, cfg_av: dict) -> dict:
    """O critério da #5: temporal com ganho >= +0,03 e IC sem o zero; canal novo, IC sem o zero."""
    t, c = resultado["temporal"], resultado["canal_novo"]
    return {
        "temporal": bool(t["ganho"] >= limites(cfg_av)["ganho_minimo"] and t["ic_ganho_lo"] > 0),
        "canal_novo": bool(c["ic_ganho_lo"] > 0),
    }


def alarmes(resultado: dict, cfg_av: dict) -> list[str]:
    """ "Bom demais, procure vazamento": manchete ou Spearman intra-canal acima do limite."""
    lim = limites(cfg_av)
    avisos = []
    for nome, r in resultado.items():
        for chave, metrica, limite in (
            ("manchete_modelo", "manchete", lim["alarme_manchete"]),
            ("spearman_intra_modelo", "Spearman intra-canal", lim["alarme_spearman"]),
        ):
            if r[chave] > limite:
                avisos.append(
                    f"bom demais, procure vazamento: {metrica} {r[chave]:.3f} > {limite} em {nome}"
                )
    return avisos


def _imprimir(resultado: dict, cfg_av: dict) -> list[str]:
    for nome, r in resultado.items():
        print(
            f"{nome}: manchete modelo {r['manchete_modelo']:.4f} · baseline "
            f"{r['manchete_base']:.4f} · ganho {r['ganho']:+.4f} "
            f"[IC {r['ic_ganho_lo']:+.4f}, {r['ic_ganho_hi']:+.4f}] · linhas {r['linhas']} · "
            f"canais {r['canais']}"
        )
    ok = criterio(resultado, cfg_av)
    minimo = limites(cfg_av)["ganho_minimo"]
    print(
        "\ncritério da #5: "
        f"temporal {'cumprido' if ok['temporal'] else 'NÃO cumprido'} (ganho >= +{minimo}, "
        f"IC sem o zero); canal novo {'cumprido' if ok['canal_novo'] else 'NÃO cumprido'} "
        "(IC sem o zero)"
    )
    avisos = alarmes(resultado, cfg_av)
    for a in avisos:
        print(f"ALARME: {a}")
    return avisos


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m previa.final")
    parser.add_argument("config")
    parser.add_argument("--abrir-o-teste", action="store_true", dest="abrir")
    parser.add_argument("--reabrir", action="store_true")
    args = parser.parse_args(argv)
    if not args.abrir:
        print(
            "recusado: o teste abre uma vez, no fim (aula 2.5, erro comum 4). Confirme que a "
            "config está congelada e rode de novo com --abrir-o-teste."
        )
        return 2
    anteriores = ja_aberto(REGISTRO)
    if anteriores and not args.reabrir:
        print(
            f"recusado: o teste já foi aberto {len(anteriores)} vez(es); a primeira: "
            f"{anteriores[0]}. Olhar o teste de novo o contamina (aula 2.3, erro comum 4); se for "
            "mesmo necessário, passe --reabrir e registre o motivo em docs/DECISOES.md."
        )
        return 2
    # tudo o que pode falhar é lido ANTES de abrir: depois da abertura, um erro gasta o teste
    cfg, cfg_b, cfg_av = ler_config(args.config), ler_config_baselines(), ler_avaliacao()
    limites(cfg_av)
    if cfg.so_amostra:
        raise ValueError(
            "so_amostra=true restringe as linhas avaliadas de outro jeito; o teste final avalia "
            "todo o período de teste. Use uma config sem so_amostra. O teste NÃO foi aberto."
        )
    duracoes = ler_duracoes()
    embeddings = {b: ler_embeddings_do_bloco(b) for b in cfg.pca}
    extras = {e: ler_extra(e) for e in cfg.extras}
    conferir_cobertura(
        caminho_da_base(cfg_av),
        cfg_av,
        {"duracoes": duracoes.to_frame(), **embeddings},
        extras,
    )
    base = carregar_base(final=True, registro=REGISTRO)  # uma vez: escreve o registro
    resultado = None
    try:
        n = len(ja_aberto(REGISTRO))
        resultado = avaliar_final(
            cfg,
            base,
            duracoes,
            cfg_av,
            cfg_b["n_boot"],
            cfg_b["seed"],
            embeddings,
            extras,
            cfg_b["min_posts"],
        )
        # registra ANTES de mostrar: os números ficam gravados antes de alguém vê-los
        run_id = registrar(
            cfg,
            args.config,
            resultado,
            base,
            tags={
                "avaliacao": "final (teste)",
                "abertura": str(n),
                "data_corte": str(cfg_av["teste_inicio"].date()),
                "teste_fim": str(cfg_av["teste_fim"].date()),
            },
            cfg_av=cfg_av,
        )
    except Exception as erro:
        _guardar_falha(erro, resultado)
        raise
    # o intervalo e o flag (M6) vêm depois do pontual já gravado: se quebrarem, o teste aberto
    # não perde o resultado do pontual, e a falha fica numa tag do run (princípio 7)
    try:
        iv = avaliar_intervalo(
            cfg,
            base,
            duracoes,
            cfg_av,
            ler_cfg_intervalo(CONFIGS / "intervalo.yaml"),
            ler_cfg_flag(),
            resultado["temporal"]["previsoes"],
            embeddings=embeddings,
            extras=extras,
            min_posts=cfg_b["min_posts"],
            seed=cfg_b["seed"],
        )
    except Exception as erro:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("intervalo_erro", f"{type(erro).__name__}: {erro}"[:5000])
        except Exception as erro_tag:  # a tag falhar não pode mascarar o erro original
            _guardar_falha(RuntimeError(f"{erro!r}; e a tag falhou: {erro_tag!r}"), resultado)
        raise
    registrar_intervalo(run_id, iv)
    avisar_importancia_dominante(resultado)
    _imprimir(resultado, cfg_av)
    _imprimir_intervalo(iv)
    print(f"\nrun {run_id} · abertura {n} · commit {commit_atual()} · base {hash_base(base)}")
    print(f"comando: uv run python -m previa.final {args.config} --abrir-o-teste")
    return 0


if __name__ == "__main__":
    sys.exit(main())
