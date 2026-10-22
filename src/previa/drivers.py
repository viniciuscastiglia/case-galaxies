"""Os `top_drivers` da explicação (M8; aulas 5.5, 5.6 e 6.3, Tool 4).

O SHAP fica em log (`contribuicoes`); aqui ele vira o contrato que o explicador consome:
`{feature, valor, contribuicao_log, multiplicador, direcao}`. O multiplicador é e^φ "em relação ao
Short típico do canal" (soma em log, conversão só no fim; 5.6, erros comuns 1 e 2). Os PCs de um
embedding viram um fator só (5.6, erro comum 5). O contexto (`hist_*`, setor, idade do canal...)
fica fora: o marketing não o muda e ele se cancela entre dois Shorts do mesmo canal (6.3). Com
`aprovados` (#36, F7), só falam os blocos que a ablação aprovou. São associações do modelo, não
causas (5.6, erro comum 6). `direcao` usa as palavras da aula ("aumenta" / "reduz") e as
contribuições com |φ| abaixo de `min_abs` caem antes do corte em `k` (5.6). Detectores (`det_*`)
ficam individuais. Sondas (`sonda_*`) aparecem com o próprio nome só se foram validadas no setor do
pedido (`sondas_validadas`; 4.4, erro comum 5); as outras somam no bloco da capa. A aditividade
(base + soma = previsão) é conferida em `contribuicoes` (erro comum 7).
"""

from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from previa.caminhos import CONFIGS
from previa.contribuicoes import agrupar_blocos, contribuicoes


def carregar_cfg(caminho=None) -> dict:
    """`configs/explicacao.yaml`: k, padrões do contexto e blocos legíveis."""
    caminho = caminho or CONFIGS / "explicacao.yaml"
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def _expandir(padroes, colunas) -> list[str]:
    return [c for c in colunas if any(fnmatch(c, p) for p in padroes)]


def _valor(v):
    """O valor da coluna em tipo Python puro; NaN vira None."""
    if hasattr(v, "item"):
        v = v.item()
    return None if isinstance(v, float) and np.isnan(v) else v


def top_drivers(
    modelo,
    x: pd.DataFrame,
    fundo: pd.DataFrame | None,
    cfg: dict,
    aprovados: set[str] | None = None,
    setor: str | None = None,
) -> list[dict]:
    """Os `k` fatores que mais mexeram na previsão de `x` (uma linha), por |φ| em log.

    Com `setor` e `cfg["sondas_validadas"]`, a sonda fora da lista do setor soma no bloco
    `cfg["bloco_das_sondas_nao_validadas"]`; sem um dos dois, cada sonda fala com o próprio nome.
    """
    if len(x) != 1:
        raise ValueError(
            f"top_drivers explica uma linha por vez; veio {len(x)}. Passe x.iloc[[i]]."
        )
    valores, _ = contribuicoes(modelo, x, fundo)  # o assert de aditividade fica lá
    blocos = {}
    for nome, padroes in cfg["blocos"].items():
        cols = _expandir(padroes, x.columns)
        if cols:
            blocos[nome] = cols
    if setor is not None and "sondas_validadas" in cfg:
        validadas = set(cfg["sondas_validadas"].get(setor, []))
        soltas = [c for c in x.columns if c.startswith("sonda_") and c not in validadas]
        if soltas:
            capa = cfg["bloco_das_sondas_nao_validadas"]
            blocos[capa] = blocos.get(capa, []) + soltas
    agrupado = agrupar_blocos(valores, blocos).iloc[0]
    contexto = set(_expandir(cfg["contexto"], x.columns))
    saida = []
    for nome, phi in agrupado.items():
        if nome in contexto or (aprovados is not None and nome not in aprovados):
            continue
        phi = float(phi)
        if abs(phi) < cfg["min_abs"]:  # desprezível: não entra (5.6, top_drivers)
            continue
        saida.append(
            {
                "feature": nome,
                "valor": None if nome in blocos else _valor(x.iloc[0][nome]),
                "contribuicao_log": round(phi, 3),
                "multiplicador": round(float(np.exp(phi)), 2),
                "direcao": "aumenta" if phi > 0 else "reduz",
            }
        )
    saida.sort(key=lambda d: abs(d["contribuicao_log"]), reverse=True)
    return saida[: cfg["k"]]
