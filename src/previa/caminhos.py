"""Caminhos absolutos do projeto (auditoria F18; aulas 6.1, 6.5 e 6.6).

O servidor MCP sobe de qualquer diretório (o cliente MCP decide o `cwd`), então nenhum caminho
pode ser relativo. A raiz sai da localização deste arquivo; dados e artefatos aceitam uma variável
de ambiente, que é lida na importação (aula 6.1, erro comum 3: o caminho precisa estar definido
antes da primeira chamada). Os testes definem as variáveis no `tests/conftest.py`, antes de
importar o `previa`, para nunca tocar nos dados reais.
"""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]  # src/previa/caminhos.py -> raiz do repo
DADOS = Path(os.environ.get("PREVIA_DADOS", RAIZ / "data"))
CONFIGS = RAIZ / "configs"
ARTEFATOS = Path(os.environ.get("GALAXIES_ARTEFATOS", RAIZ / "artefatos"))  # nome da 6.1 e 6.6
