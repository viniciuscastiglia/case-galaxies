"""Carrega a base de treino para os experimentos, com a trava do período de teste.

Aulas 1.2, 2.3 e 2.5: o teste é separado no dia 1 e usado uma vez, no fim. Fora da avaliação
final, `carregar_base` não devolve nada publicado a partir de `teste_inicio`
(configs/avaliacao.yaml). Abrir o teste exige `final=True`, avisa e deixa um registro em
data/logs/teste_aberto.log: é a prova de quantas vezes o teste foi olhado.

A base é a cópia congelada que `base` aponta no config (aula 2.5, erro comum 2): a coleta diária
reexporta a `base_minima` com as curtidas novas, e a mesma config daria outro número no dia
seguinte. Sem a chave, o erro é explícito: nunca cai na base viva em silêncio.
"""

import argparse
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from previa.caminhos import CONFIGS
from previa.coletor import DADOS


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
    if config.get("exige_teste_aberto", False):
        # a v2 gasta os meses de teste da v1 (#33): só depois da avaliação final
        aberturas = (
            Path(registro).read_text(encoding="utf-8").strip() if Path(registro).exists() else ""
        )
        if not aberturas:
            raise ValueError(
                "esta config (base v2) usa os meses de teste da v1 para treino e calibração, e o "
                "teste final ainda não foi aberto (registro vazio): rode antes "
                "`python -m previa.final`"
            )
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


def congelar(origem: Path, destino: Path) -> str:
    """Congela a base mínima da coleta diária numa cópia com nome (aula 2.5, erro comum 2).

    Recusa sobrescrever um arquivo que já existe (a v1 é a base dos runs das H1 a H6, decisão #33),
    valida o esquema mínimo (princípio 6) antes de copiar e devolve o `hash_base` da cópia.
    """
    from previa.esquema import BaseShorts
    from previa.experimento import hash_base  # import tardio: o experimento importa este módulo

    origem, destino = Path(origem), Path(destino)
    if destino.exists():
        raise FileExistsError(
            f"{destino.name} já existe e não é sobrescrito: escolha outro nome (ex.: base_v3)"
        )
    base = pd.read_parquet(origem)
    BaseShorts.validate(base, lazy=True)  # erro aqui: corrija na coleta, não na cópia
    shutil.copyfile(origem, destino)
    return hash_base(pd.read_parquet(destino))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Base congelada (decisão #33)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("congelar", help="copia data/base_minima.parquet para o nome dado")
    c.add_argument("nome", help="ex.: base_v2.parquet (dentro de data/)")
    args = ap.parse_args(argv)
    h = congelar(DADOS / "base_minima.parquet", DADOS / args.nome)
    print(f"{args.nome} congelada; hash_base {h}")


if __name__ == "__main__":
    main()
