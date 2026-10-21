"""Nenhum teste toca nos dados reais (auditoria F18; aulas 6.1, 6.5, 6.6).

`previa.caminhos` lê `PREVIA_DADOS` e `GALAXIES_ARTEFATOS` na importação, e os defaults de função
(`arquivo=DADOS / "cota.json"`) ficam ligados nesse momento. Por isso as variáveis são definidas
aqui, no topo, antes de qualquer import do `previa`, e sempre por atribuição: um valor que já
estivesse no ambiente do Vinicius (apontando para `data/`) nunca pode valer nos testes.

O MLflow segue a mesma regra. Sem endereço no ambiente, ele usa `./mlflow.db` e `./mlruns` do cwd,
que são o repositório e os runs lacrados; e `set_tracking_uri` e `set_experiment` escrevem
MLFLOW_TRACKING_URI e MLFLOW_EXPERIMENT_ID no `os.environ`, então um endereço ou um id deixado por
um teste valeria para o seguinte (MLflow 3.16; auditoria, revisão final). A raiz dos artefatos é a
`_MLFLOW_SERVER_ARTIFACT_ROOT`, a variável que o próprio MLflow lê para o padrão do cliente.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

PASTA_TESTES = Path(tempfile.mkdtemp(prefix="previa-testes-")).resolve()
PASTAS = (PASTA_TESTES / "data", PASTA_TESTES / "artefatos")
os.environ["PREVIA_DADOS"] = str(PASTAS[0])
os.environ["GALAXIES_ARTEFATOS"] = str(PASTAS[1])
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{PASTA_TESTES / 'mlflow.db'}"
os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = str(PASTA_TESTES / "mlruns")
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

atexit.register(shutil.rmtree, PASTA_TESTES, ignore_errors=True)


@pytest.fixture(autouse=True)
def pastas_de_teste_vazias():
    """Cada teste começa com `data/` e `artefatos/` de teste vazios: o que um teste escreve em
    `caminhos.DADOS` não vaza para o seguinte. Só apaga dentro da pasta temporária."""
    for pasta in PASTAS:
        assert pasta.parent == PASTA_TESTES and PASTA_TESTES.name.startswith("previa-testes-")
        shutil.rmtree(pasta, ignore_errors=True)
        pasta.mkdir(parents=True)
    yield
