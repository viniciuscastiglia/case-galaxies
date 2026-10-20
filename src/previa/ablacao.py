"""Ablação por blocos (aulas 2.6 e 4.8): quanto cada bloco de features vale, e com que segurança.

Os três desenhos da 2.6, rodados juntos, porque um sozinho engana:

- cumulativa (a escada do briefing): os blocos entram um por vez, na ordem da config, e cada
  degrau é medido contra o anterior; o primeiro, contra a mediana madura do canal. Quem entra
  primeiro leva o crédito (4.8, erro comum 6);
- todos menos um: o completo contra o completo sem cada bloco, o que não depende da ordem;
- controle negativo: o completo contra o completo + colunas de puro ruído, o tamanho do acaso
  nesta avaliação (2.6, erro comum 2).

Cada variante sai da config do modelo completo mudando só as features, os PCAs e os extras: o
mesmo split, os mesmos hiperparâmetros e a mesma seed (4.8, erro comum 2). Cada linha da tabela é
o ganho de B sobre A nas duas réguas da #27, com IC 95% do bootstrap pareado por canal (2.6, erros
comuns 3 e 4). Quem fica é decidido pelas hipóteses (H2 a H9), não por esta tabela sozinha.

Uso: uv run python -m previa.ablacao configs/ablacao.yaml
"""

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from previa.baselines import ler_config as ler_config_baselines
from previa.coletor import DADOS
from previa.dados import carregar_base
from previa.experimento import (
    Config,
    commit_atual,
    ganhos,
    hash_base,
    ler_config,
    ler_duracoes,
    ler_embeddings_do_bloco,
    ler_extra,
    preparar,
    registrar,
    rodar,
    tabela_de_previsoes,
)
from previa.splits import ler_avaliacao

BASELINE = "mediana madura"
CAMPOS = ("features", "pca", "extras")


def _casa(entrada: str, padrao: str) -> bool:
    """`hist_*` pega toda entrada da config que começa com `hist_`; senão, o nome exato."""
    return entrada == padrao or (padrao.endswith("*") and entrada.startswith(padrao[:-1]))


def separar_blocos(completo: Config, blocos: dict) -> dict:
    """Cada bloco com as suas entradas da config do completo (features, PCAs e extras).

    Falha alto se um padrão não acha nada, se uma entrada cai em dois blocos ou se sobra entrada
    fora dos blocos: a escada tem de terminar no completo, e um nome errado faria o "menos um"
    igual ao completo sem ninguém perceber.
    """
    saida, dono = {}, {}
    for nome, b in blocos.items():
        if desconhecidas := set(b) - set(CAMPOS):
            raise ValueError(f"bloco {nome}: chave desconhecida {sorted(desconhecidas)}")
        if not any(b.get(c) for c in CAMPOS):
            raise ValueError(f"bloco {nome} vazio: sem features, PCA nem extras")
        for padrao in b.get("features", []):
            if not any(_casa(e, padrao) for e in completo.features):
                raise ValueError(f"bloco {nome}: {padrao} não está nas features do completo")
        faltam = [f"pca {p}" for p in b.get("pca", []) if p not in completo.pca]
        faltam += [f"extra {e}" for e in b.get("extras", []) if e not in completo.extras]
        if faltam:
            raise ValueError(f"bloco {nome}: {faltam} não estão no completo")
        parte = {
            "features": [e for e in completo.features if any(_casa(e, p) for p in b["features"])]
            if "features" in b
            else [],
            "pca": list(b.get("pca", [])),
            "extras": list(b.get("extras", [])),
        }
        for campo, itens in parte.items():
            for item in itens:
                if (campo, item) in dono:
                    raise ValueError(f"{item} está nos blocos {dono[campo, item]} e {nome}")
                dono[campo, item] = nome
        saida[nome] = parte
    todas = [("features", e) for e in completo.features]
    todas += [("pca", p) for p in completo.pca] + [("extras", e) for e in completo.extras]
    if sobra := [item for campo, item in todas if (campo, item) not in dono]:
        raise ValueError(f"fora de qualquer bloco: {sobra} (a escada não terminaria no completo)")
    if len(saida) < 2:
        raise ValueError("a ablação precisa de pelo menos 2 blocos")
    return saida


def montar(completo: Config, partes: list[dict], nome: str, descricao: str, ruido=0) -> Config:
    """O completo só com os blocos de `partes`, na ordem da config do completo."""
    junta = {c: {item for p in partes for item in p[c]} for c in CAMPOS}
    features = [e for e in completo.features if e in junta["features"]]
    return replace(
        completo,
        nome=nome,
        descricao=descricao,
        features=features + [f"ruido_{k}" for k in range(1, ruido + 1)],
        pca={b: n for b, n in completo.pca.items() if b in junta["pca"]},
        extras=[e for e in completo.extras if e in junta["extras"]],
        ruido=ruido,
    )


def desenho(completo: Config, blocos: dict, ruido=2):
    """As configs únicas e as comparações da tabela, cada uma como (desenho, bloco, A, B).

    Duas variantes com as mesmas colunas rodam uma vez só (o "sem o último bloco" é o penúltimo
    degrau da escada).
    """
    partes = separar_blocos(completo, blocos)
    nomes = list(partes)
    configs, chave_de = {}, {}

    def guardar(nome, cfg):
        chave = (tuple(cfg.features), tuple(cfg.pca.items()), tuple(cfg.extras), cfg.ruido)
        if chave not in chave_de:
            chave_de[chave] = nome
            configs[nome] = cfg
        return chave_de[chave]

    comparacoes, anterior = [], BASELINE
    for k in range(1, len(nomes) + 1):
        nome = "completo" if k == len(nomes) else "escada-" + "+".join(nomes[:k])
        desc = f"ablação, cumulativa: {' + '.join(nomes[:k])}"
        atual = guardar(nome, montar(completo, [partes[n] for n in nomes[:k]], nome, desc))
        comparacoes.append(("cumulativa", nomes[k - 1], anterior, atual))
        anterior = atual
    for n in nomes:
        resto = [partes[m] for m in nomes if m != n]
        cfg = montar(completo, resto, f"sem-{n}", f"ablação: completo sem {n}")
        nome = guardar(f"sem-{n}", cfg)
        comparacoes.append(("todos menos um", n, nome, "completo"))
    if ruido:
        desc = f"ablação, controle negativo: completo + {ruido} colunas de ruído"
        cfg = montar(completo, list(partes.values()), "completo+ruido", desc, ruido)
        nome = guardar("completo+ruido", cfg)
        comparacoes.append(("controle negativo", "ruido", "completo", nome))
    return configs, comparacoes


def ablar(
    completo: Config,
    blocos: dict,
    base,
    duracoes,
    cfg_av,
    n_boot=2000,
    seed=0,
    embeddings=None,
    extras=None,
    so_avaliar=None,
    min_posts=3,
    ruido=2,
):
    """Roda cada variante no runner e devolve (tabela, configs, resultados)."""
    configs, comparacoes = desenho(completo, blocos, ruido)
    extras = extras or {}
    resultados = {
        nome: rodar(
            cfg,
            base,
            duracoes,
            cfg_av,
            n_boot,
            seed,
            embeddings,
            {e: extras[e] for e in cfg.extras},  # só os da variante (a guarda do runner)
            so_avaliar,
            min_posts,
        )
        for nome, cfg in configs.items()
    }
    d = preparar(base, duracoes, cfg_av, min_posts=min_posts)
    prev = {nome: tabela_de_previsoes(r) for nome, r in resultados.items()}
    primeiro = comparacoes[0][3]  # a mediana madura, nas linhas e nos folds do primeiro degrau
    p_base = d.set_index("post_id")["p_base"]
    prev[BASELINE] = prev[primeiro].assign(p=prev[primeiro]["post_id"].map(p_base))
    linhas = []
    for qual, bloco, a, b in comparacoes:
        for avaliacao, g in ganhos(d, prev[a], prev[b], n_boot, seed).items():
            linhas.append(
                {"desenho": qual, "bloco": bloco, "a": a, "b": b, "avaliacao": avaliacao, **g}
                | {"n_features_b": resultados[b][avaliacao]["n_features"]}
            )
    return pd.DataFrame(linhas), configs, resultados


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("uso: python -m previa.ablacao configs/ablacao.yaml", file=sys.stderr)
        return 2
    caminho = Path(argv[0])
    cfg_abl = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    completo, cfg_b = ler_config(cfg_abl["completo"]), ler_config_baselines()
    base = carregar_base()  # sem o período de teste
    embeddings = {b: ler_embeddings_do_bloco(b) for b in completo.pca}
    extras = {e: ler_extra(e) for e in completo.extras}
    amostra = None
    if completo.so_amostra:
        amostra = set(pd.read_csv(DADOS / "amostra_multimodal.csv")["post_id"])
    tabela, configs, resultados = ablar(
        completo,
        cfg_abl["blocos"],
        base,
        ler_duracoes(),
        ler_avaliacao(),
        cfg_b["n_boot"],
        cfg_b["seed"],
        embeddings,
        extras,
        amostra,
        cfg_b["min_posts"],
        cfg_abl.get("ruido", 2),
    )
    tags = {"ablacao": caminho.stem}
    run_de = {
        nome: registrar(cfg, f"{caminho}#{nome}", resultados[nome], base, tags)
        for nome, cfg in configs.items()
    }
    tabela["run_a"], tabela["run_b"] = tabela["a"].map(run_de), tabela["b"].map(run_de)
    destino = DADOS / "ablacao" / f"{caminho.stem}.csv"
    destino.parent.mkdir(exist_ok=True)
    tabela.to_csv(destino, index=False)
    print(tabela.drop(columns=["a", "run_a", "run_b"]).round(4).to_string(index=False))
    print(f"\n{len(configs)} runs · commit {commit_atual()} · base {hash_base(base)} -> {destino}")
    print(f"comando: uv run python -m previa.ablacao {caminho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
