"""Caminhos absolutos do projeto (auditoria F18; aulas 6.1, 6.5 e 6.6).

O servidor MCP sobe de qualquer diretório (o cliente MCP decide o `cwd`), então nenhum caminho
pode ser relativo. A raiz sai da localização deste arquivo; dados e artefatos aceitam uma variável
de ambiente, que é lida na importação (aula 6.1, erro comum 3: o caminho precisa estar definido
antes da primeira chamada). Variável vazia vale como ausente e valor relativo é erro. Os testes
definem as variáveis no `tests/conftest.py`, antes de importar o `previa`, para nunca tocar nos
dados reais.
"""

import os
from pathlib import Path


def _caminho_absoluto(variavel: str, padrao: Path) -> Path:
    """O caminho da variável de ambiente, ou o padrão se ela não existir ou estiver vazia.

    `export X=` (ou `X=` no `.env`) deixa a variável vazia, e `Path("")` seria o cwd: vazio vale
    como ausente. Um valor relativo falha aqui, na importação, e não vira um caminho que muda com
    o `cwd` do cliente MCP (aula 6.1, erro comum 3; falha nunca silenciosa, princípio 7).
    """
    valor = os.environ.get(variavel) or str(padrao)
    caminho = Path(valor)
    if not caminho.is_absolute():
        raise ValueError(
            f"{variavel}={valor!r} não é um caminho absoluto: o servidor sobe de qualquer "
            f"diretório. Use um caminho completo (ex.: {padrao}) ou apague a variável"
        )
    return caminho


RAIZ = Path(__file__).resolve().parents[2]  # src/previa/caminhos.py -> raiz do repo
DADOS = _caminho_absoluto("PREVIA_DADOS", RAIZ / "data")
CONFIGS = RAIZ / "configs"
ARTEFATOS = _caminho_absoluto("GALAXIES_ARTEFATOS", RAIZ / "artefatos")  # nome da 6.1 e 6.6
