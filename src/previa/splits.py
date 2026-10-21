"""Splits por data e por canal (aula 2.3).

Todas as funções devolvem rótulos do índice do DataFrame e cortam por DATA, nunca pela posição da
linha: o `TimeSeriesSplit` do scikit-learn corta por posição e, num painel ordenado por canal,
testa o passado com o futuro sem avisar (erro comum 2).

As datas vêm de configs/avaliacao.yaml (decisões #2 e #3 de 15/10/2026, em docs/DECISOES.md):
treino (01/07/2021 até cal_inicio - gap) -> gap de 30 dias -> calibração (fev–mar/2026) -> teste
(01/04 a 14/09/2026). O walk-forward de desenvolvimento roda só dentro do bloco de treino.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from previa.caminhos import CONFIGS

DATAS = ("teste_inicio", "treino_inicio", "cal_inicio", "teste_fim")


def _utc(data) -> pd.Timestamp:
    """published_at está em UTC: datas sem fuso viram 00h UTC, para a comparação não quebrar."""
    data = pd.Timestamp(data)
    return data.tz_localize("UTC") if data.tzinfo is None else data.tz_convert("UTC")


def ler_avaliacao(caminho=CONFIGS / "avaliacao.yaml") -> dict:
    """configs/avaliacao.yaml com as datas já em UTC."""
    cfg = yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))
    for chave in DATAS:
        cfg[chave] = _utc(str(cfg[chave]))
    cfg["walk_forward"]["inicio"] = _utc(str(cfg["walk_forward"]["inicio"]))
    return cfg


# --- portadas da aula 2.3 (passo 2), com as colunas da base mínima ---


def split_temporal(df, data_corte, gap_dias=0):
    """Treino: antes de (corte - gap). Teste: a partir do corte."""
    corte = _utc(data_corte)
    treino = df.index[df["published_at"] < corte - pd.Timedelta(days=gap_dias)]
    teste = df.index[df["published_at"] >= corte]
    return treino, teste


def walk_forward(df, n_folds=4, gap_dias=30, *, inicio_teste):
    """Janela expansiva: o futuro é fatiado em n_folds janelas por DATA.

    A última janela fecha na data mais recente de `df`: no Prévia, use `folds_de_desenvolvimento`,
    que passa só o bloco de treino. `inicio_teste` vai por nome: na ordem da aula, uma chamada
    posicional como `walk_forward(df, 4, 30)` não quebraria (4 vira uma data de 1970).
    """
    fim = df["published_at"].max() + pd.Timedelta(days=1)
    bordas = pd.date_range(_utc(inicio_teste), fim, periods=n_folds + 1)
    for ini, fim_janela in zip(bordas[:-1], bordas[1:], strict=True):
        treino = df.index[df["published_at"] < ini - pd.Timedelta(days=gap_dias)]
        teste = df.index[(df["published_at"] >= ini) & (df["published_at"] < fim_janela)]
        yield treino, teste


def split_por_canal(df, frac=0.25, seed=0):
    """Sorteia canais inteiros para o teste: nenhum canal aparece dos dois lados."""
    canais = np.sort(df["channel_id"].unique())
    rng = np.random.default_rng(seed)
    teste_canais = rng.choice(canais, size=round(frac * len(canais)), replace=False)
    no_teste = df["channel_id"].isin(teste_canais)
    return df.index[~no_teste], df.index[no_teste]


# --- M1 (Vinicius): os testes-contrato estão em tests/test_splits.py ---


def split_treino_cal_teste(df, cfg):
    """Três blocos por data, na ordem treino | gap | calibração | teste (aula 2.3, passo 6).

    - treino: de cfg["treino_inicio"] (inclusive) até cfg["cal_inicio"] - cfg["gap_dias"] dias;
    - calibração: de cfg["cal_inicio"] (inclusive) até cfg["teste_inicio"];
    - teste: de cfg["teste_inicio"] (inclusive) até cfg["teste_fim"].
    Os fins são exclusivos. Devolve (treino, cal, teste) como rótulos do índice, como as
    funções acima.
    """
    data = df["published_at"]
    fim_treino = cfg["cal_inicio"] - pd.Timedelta(days=cfg["gap_dias"])
    treino = df.index[(data >= cfg["treino_inicio"]) & (data < fim_treino)]
    cal = df.index[(data >= cfg["cal_inicio"]) & (data < cfg["teste_inicio"])]
    teste = df.index[(data >= cfg["teste_inicio"]) & (data < cfg["teste_fim"])]
    return treino, cal, teste


def canal_novo_no_futuro(df, treino, teste, n_grupos=4, seed=0):
    """ "Canal novo no futuro" (aula 2.3): o cliente que o modelo nunca viu, publicando depois.

    Recebe os blocos de tempo (`treino` e `teste`, rótulos do índice) e divide os canais de `df`
    em `n_grupos` grupos sorteados com `seed`. Para cada grupo g, um par (treino_g, teste_g):
    - treino_g: as linhas de `treino` cujo canal NÃO está em g;
    - teste_g: as linhas de `teste` cujo canal está em g.
    Cada canal é "novo" exatamente uma vez. Os posts antigos dos canais de g saem do treino, mas
    continuam no `df` para alimentar as features de histórico deles.
    Devolve uma lista com os n_grupos pares.
    """
    canais = np.sort(df["channel_id"].unique())
    rng = np.random.default_rng(seed)
    grupos = np.array_split(rng.permutation(canais), n_grupos)

    pares = []
    for canais_g in grupos:
        no_grupo = df.index[df["channel_id"].isin(canais_g)]
        treino_g = treino.difference(no_grupo)
        teste_g = teste.intersection(no_grupo)
        pares.append((treino_g, teste_g))
    return pares


# --- em volta da M1 (Claude) ---


def folds_de_desenvolvimento(df, cfg):
    """Os folds de walk-forward do desenvolvimento, só dentro do bloco de treino (erro comum 4).

    É a entrada dos experimentos: com a base inteira, o `walk_forward` sozinho testaria o último
    fold no gap e na calibração. Um fold sem nenhum post no teste é erro de configuração.
    """
    treino, _, _ = split_treino_cal_teste(df, cfg)
    wf = cfg["walk_forward"]
    folds = list(
        walk_forward(df.loc[treino], wf["n_folds"], cfg["gap_dias"], inicio_teste=wf["inicio"])
    )
    vazios = [k for k, (_, teste) in enumerate(folds, start=1) if len(teste) == 0]
    if vazios:
        raise ValueError(
            f"fold(s) {vazios} sem nenhum post no teste: confira walk_forward.inicio e n_folds "
            "em configs/avaliacao.yaml"
        )
    return folds
