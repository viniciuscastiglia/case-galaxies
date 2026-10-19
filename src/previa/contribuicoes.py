"""Contribuições SHAP da previsão (Fase 5, encanamento; aula 5.6).

Sem fundo: o TreeSHAP nativo do LightGBM (`pred_contrib=True`), os mesmos valores do
`TreeExplainer` no modo `tree_path_dependent`, sem dependência extra. Com fundo (a referência que
a #16 vai escolher, por exemplo os posts recentes do canal, 5.6): o `TreeExplainer` intervencional
do shap 0.52. Nos dois casos, a conta tem de fechar, base + soma = previsão, e o assert fica no
código (erro comum 7). `agrupar_blocos` soma os componentes de PCA de um embedding num fator só
(erro comum 5). Os multiplicadores e a direção (`top_drivers`) são a M8.
"""

import numpy as np
import pandas as pd


def contribuicoes(modelo, x: pd.DataFrame, fundo: pd.DataFrame | None = None, tol=1e-4):
    """SHAP de cada feature de cada linha de `x` (escala log) e o valor base."""
    if fundo is None:
        c = modelo.predict(x, pred_contrib=True)
        valores, base = c[:, :-1], float(c[0, -1])
    else:
        import shap

        explainer = shap.TreeExplainer(modelo, data=fundo, feature_perturbation="interventional")
        exp = explainer(x)
        valores, base = exp.values, float(np.ravel(exp.base_values)[0])
    previsao = modelo.predict(x)
    if not np.allclose(valores.sum(axis=1) + base, previsao, atol=tol):
        raise AssertionError("SHAP não fecha: base + soma das contribuições != previsão")
    return pd.DataFrame(valores, index=x.index, columns=x.columns), base


def agrupar_blocos(valores: pd.DataFrame, blocos: dict[str, list[str]]) -> pd.DataFrame:
    """Soma as colunas de cada bloco (16 PCs do título -> "título"); a conta continua fechando."""
    saida = {nome: valores[cols].sum(axis=1) for nome, cols in blocos.items()}
    usadas = {c for cols in blocos.values() for c in cols}
    saida |= {c: valores[c] for c in valores.columns if c not in usadas}
    return pd.DataFrame(saida, index=valores.index)
