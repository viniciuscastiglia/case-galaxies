"""Auditoria das datas das features (aula 1.7, exercício; `docs/AUDITORIA_FEATURES.md`).

Para cada feature, a coluna de data em que o valor foi observado; `auditar_datas` devolve a
fração das linhas em que ela foi observada **depois** de `published_at`. Fração maior que zero
pede uma decisão: remover, substituir ou manter com ablação e registro no data card.

`uv run python -m previa.auditoria` audita as features do runner na base de desenvolvimento.
"""

import sys

import pandas as pd


def auditar_datas(df: pd.DataFrame, observado_em: dict[str, str]) -> pd.Series:
    """Fração das linhas em que cada feature foi observada DEPOIS da publicação (1.7)."""
    return pd.Series(
        {feat: float((df[col] > df["published_at"]).mean()) for feat, col in observado_em.items()},
        name="observada_depois",
    ).sort_values(ascending=False, kind="stable")


def mapa_do_previa(features: list[str]) -> dict[str, str]:
    """Quando cada feature do runner é observada (a auditoria de 12/10, em código)."""
    mapa = {}
    for f in features:
        if f.startswith(("hist_", "tendencia")):
            mapa[f] = "hist_ultimo_ts"  # o último post maduro usado
        elif f in ("dias_desde_ultimo", "posts_30d"):
            mapa[f] = "cad_ultimo_ts"  # o último post anterior, de qualquer idade
        elif f.startswith(("txt_", "titulo_")):
            mapa[f] = "collected_at"  # o título é a versão do dia da coleta
        else:  # calendário, duração e setor: conhecidos no instante da publicação
            mapa[f] = "published_at"
    return mapa


def main() -> int:
    from previa.dados import carregar_base
    from previa.experimento import ler_config, ler_duracoes, preparar
    from previa.splits import ler_avaliacao

    cfg = ler_config("configs/experimentos/opcao_a_manuais.yaml")
    d = preparar(carregar_base(), ler_duracoes(), ler_avaliacao(), cfg.lacuna_dias)
    d["cad_ultimo_ts"] = d["published_at"] - pd.to_timedelta(d["dias_desde_ultimo"], "D")
    d = d[d["linha"]]
    print(auditar_datas(d, mapa_do_previa(cfg.features)).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
