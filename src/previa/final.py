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
import sys
from pathlib import Path

import pandas as pd

from previa.baselines import ler_config as ler_config_baselines
from previa.coletor import DADOS
from previa.dados import carregar_base
from previa.experimento import (
    avisar_importancia_dominante,
    commit_atual,
    hash_base,
    ler_config,
    ler_duracoes,
    ler_embeddings_do_bloco,
    ler_extra,
    registrar,
    rodar,
)
from previa.splits import canal_novo_no_futuro, ler_avaliacao, split_treino_cal_teste

REGISTRO = DADOS / "logs" / "teste_aberto.log"
CHAVES_FINAL = ("ganho_minimo", "cobertura_min_extra", "alarme_manchete", "alarme_spearman")


def limites(cfg_av: dict) -> dict:
    """Os três limites de `final:` em configs/avaliacao.yaml; sem a chave, erro (sem padrão)."""
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
    `duplicate_group` do arquivo, sem curtidas. Duração e embeddings (calculados para todo post)
    exigem 100%; cada extra, `final.cobertura_min_extra` (NaN sem mídia é legítimo). A cobertura
    de cada extra é sempre impressa. Falhou: `ValueError`, e o teste não é aberto.
    """
    minimo = limites(cfg_av)["cobertura_min_extra"]
    ids = pd.read_parquet(caminho_base, columns=["post_id", "published_at", "duplicate_group"])
    datas = ids["published_at"]
    na_janela = (datas >= cfg_av["teste_inicio"]) & (datas < cfg_av["teste_fim"])
    canonicos = ids.loc[na_janela & (ids["duplicate_group"] == ids["post_id"]), "post_id"]
    n = len(canonicos)
    faltam = {k: int((~canonicos.isin(b.index)).sum()) for k, b in obrigatorios.items()}
    faltam = {k: v for k, v in faltam.items() if v}
    ruins = []
    for nome, bloco in extras.items():
        tem = int(canonicos.isin(bloco.dropna(how="all").index).sum())
        pct = tem / n if n else 0.0
        print(f"cobertura do extra {nome}: {tem}/{n} ({pct:.1%}) nos posts do teste")
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

    TODO(M6): a cobertura do intervalo no teste entra aqui, depois da calibração.
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
    avisar_importancia_dominante(resultado)
    _imprimir(resultado, cfg_av)
    print(f"\nrun {run_id} · abertura {n} · commit {commit_atual()} · base {hash_base(base)}")
    print(f"comando: uv run python -m previa.final {args.config} --abrir-o-teste")
    return 0


if __name__ == "__main__":
    sys.exit(main())
