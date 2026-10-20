"""Testes da guarda do OpenMP (troubleshooting do hub; conflitos #16).

Cada caso roda num Python novo: a guarda depende do que já foi importado no processo.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

HUB = Path.home() / ".cache/huggingface/hub"
MINILM = HUB / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"


def _python(codigo: str, **env) -> subprocess.CompletedProcess:
    ambiente = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"} | env
    return subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, env=ambiente, timeout=240
    )


def test_fixa_uma_thread_antes_dos_imports():
    r = _python(
        "from previa.openmp import um_runtime; um_runtime(); import os; "
        "print(os.environ['OMP_NUM_THREADS'])"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "1"


def test_erro_alto_se_o_lightgbm_ja_carregou_sem_a_guarda():
    # falha nunca silenciosa: tarde demais para fixar as threads, o segfault viria no 1º encode
    r = _python("import lightgbm\nfrom previa.openmp import um_runtime\num_runtime()")
    assert r.returncode != 0 and "lightgbm" in r.stderr and "OMP_NUM_THREADS" in r.stderr


def test_com_a_variavel_ja_definida_o_import_anterior_passa():
    r = _python(
        "import lightgbm\nfrom previa.openmp import um_runtime\num_runtime()", OMP_NUM_THREADS="1"
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(
    sys.platform != "darwin" or not MINILM.exists(),
    reason="o segfault é do macOS; precisa do MiniLM em cache",
)
def test_torch_e_lightgbm_no_mesmo_processo_com_a_guarda():
    # 20/10, Mac M4: sem a guarda, exit 139 no 1º encode do torch, em qualquer ordem de import
    codigo = """
from previa.openmp import um_runtime
um_runtime()
import lightgbm as lgb
import numpy as np
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
m.encode(["O AUDI RS6 MAIS EXTREMO"] * 8)
x = np.random.default_rng(0).normal(size=(2000, 10))
b = lgb.train({"objective": "regression", "verbosity": -1}, lgb.Dataset(x, x[:, 0]), 20)
b.predict(x[:1])
m.encode(["de novo"])
print("ok")
"""
    r = _python(codigo, HF_HUB_OFFLINE="1")
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip().endswith("ok")
