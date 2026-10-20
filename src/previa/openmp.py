"""Guarda do OpenMP no processo que junta torch e LightGBM (troubleshooting do hub; conflitos #16).

Medido em 20/10, no Mac M4 do projeto: com torch (sentence-transformers) e LightGBM no mesmo
processo, o primeiro `encode` do torch dá segfault (exit 139) em qualquer ordem de import, também
só prevendo com um booster salvo; importar o LightGBM primeiro, a dica do troubleshooting, não
basta. Com `OMP_NUM_THREADS=1` antes dos imports, os dois convivem.

O treino e a extração continuam em processos separados, com todas as threads (aula 4.8). Só o
processo que serve a tool (a função, o grafo e o servidor MCP), que embute o título e a capa e
prevê com o GBM, chama `um_runtime()` antes de qualquer import pesado.
"""

import os
import sys


def um_runtime() -> None:
    """Fixa uma thread de OpenMP; erro alto se já for tarde (torch ou LightGBM carregados)."""
    carregados = [m for m in ("torch", "lightgbm") if m in sys.modules]
    if carregados and os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError(
            f"{', '.join(carregados)} já foi importado sem OMP_NUM_THREADS=1: chame "
            "previa.openmp.um_runtime() antes dos imports (senão, segfault no 1º encode do torch)"
        )
    os.environ["OMP_NUM_THREADS"] = "1"
