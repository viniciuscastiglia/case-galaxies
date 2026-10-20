"""Carrega a base de treino para os experimentos, com a trava do período de teste.

Aulas 1.2, 2.3 e 2.5: o teste é separado no dia 1 e usado uma vez, no fim. Fora da avaliação
final, `carregar_base` não devolve nada publicado a partir de `teste_inicio`
(configs/avaliacao.yaml). Abrir o teste exige `final=True`, avisa e deixa um registro em
data/logs/teste_aberto.log: é a prova de quantas vezes o teste foi olhado.

A base é a cópia congelada que `base` aponta no config (aula 2.5, erro comum 2): a coleta diária
reexporta a `base_minima` com as curtidas novas, e a mesma config daria outro número no dia
seguinte. Sem a chave, o erro é explícito: nunca cai na base viva em silêncio.
"""

import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from previa.coletor import DADOS

CONFIGS = Path("configs")


def carregar_base(
    final: bool = False,
    caminho_base=None,
    caminho_config=CONFIGS / "avaliacao.yaml",
    registro=DADOS / "logs" / "teste_aberto.log",
) -> pd.DataFrame:
    """Base mínima para desenvolver (final=False) ou para a avaliação final (final=True)."""
    config = yaml.safe_load(Path(caminho_config).read_text(encoding="utf-8"))
    if caminho_base is None:
        caminho_base = DADOS / config["base"]  # um caminho absoluto no config vale como está
    teste_inicio = pd.Timestamp(str(config["teste_inicio"]), tz="UTC")
    base = pd.read_parquet(caminho_base)
    if not final:
        return base[base["published_at"] < teste_inicio].reset_index(drop=True)
    registro = Path(registro)
    registro.parent.mkdir(parents=True, exist_ok=True)
    with registro.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} teste aberto ({caminho_base})\n")
    warnings.warn(
        f"avaliação final: o período de teste (desde {teste_inicio.date()}) foi aberto. "
        "Daqui em diante, nenhuma decisão de modelo pode olhar para ele.",
        UserWarning,
        stacklevel=2,
    )
    return base
