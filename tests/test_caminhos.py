"""Caminhos absolutos e variáveis de ambiente (auditoria F18; aulas 6.1, 6.5 e 6.6).

O servidor MCP sobe de outro diretório: nenhum caminho pode depender do `cwd`. Os subprocessos
testam isso de verdade, porque os caminhos são lidos na importação do `previa`.
"""

import os
import subprocess
import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
VARIAVEIS = ("PREVIA_DADOS", "GALAXIES_ARTEFATOS")


def _rodar(codigo, cwd, **env_extra):
    """Roda `codigo` num Python novo, em `cwd`, sem as variáveis que o conftest definiu."""
    env = {k: v for k, v in os.environ.items() if k not in VARIAVEIS}
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", codigo], cwd=cwd, env=env, capture_output=True, text=True
    )


def test_caminhos_valem_de_outro_diretorio(tmp_path):
    codigo = "from previa import caminhos, splits; print(caminhos.CONFIGS); splits.ler_avaliacao()"
    r = _rodar(codigo, tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(RAIZ_REPO / "configs")


def test_variavel_dos_artefatos_vale(tmp_path):
    alvo = tmp_path / "x"
    codigo = (
        "from previa import caminhos, modelo; print(caminhos.ARTEFATOS); print(modelo.ARTEFATOS)"
    )
    r = _rodar(codigo, tmp_path, GALAXIES_ARTEFATOS=str(alvo))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(alvo), str(alvo)]


def test_variavel_dos_dados_vale(tmp_path):
    alvo = tmp_path / "d"
    codigo = (
        "from previa import caminhos, coletor, dados; "
        "print(caminhos.DADOS); print(coletor.DADOS); print(dados.DADOS)"
    )
    r = _rodar(codigo, tmp_path, PREVIA_DADOS=str(alvo))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(alvo)] * 3


def test_sem_variaveis_os_padroes_sao_da_raiz_do_repo(tmp_path):
    codigo = "from previa import caminhos; print(caminhos.DADOS); print(caminhos.ARTEFATOS)"
    r = _rodar(codigo, tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(RAIZ_REPO / "data"), str(RAIZ_REPO / "artefatos")]


def test_testes_nunca_usam_os_dados_reais():
    from previa import caminhos

    assert caminhos.DADOS != caminhos.RAIZ / "data"
    assert caminhos.ARTEFATOS != caminhos.RAIZ / "artefatos"
    assert caminhos.DADOS.is_absolute() and caminhos.ARTEFATOS.is_absolute()
