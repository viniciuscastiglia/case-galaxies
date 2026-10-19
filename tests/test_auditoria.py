"""Teste do `auditar_datas` (aula 1.7, exercício): a fração das linhas em que cada feature foi
observada depois da publicação. O caso é o da solução da aula."""

import pandas as pd

from previa.auditoria import auditar_datas


def test_caso_da_aula():
    df = pd.DataFrame(
        {
            "published_at": pd.to_datetime(
                ["2025-03-10T17:00Z", "2026-08-06T22:00Z", "2026-09-10T22:00Z"]
            ),
            "collected_at": pd.Timestamp("2026-09-28T12:00Z"),
            "ultimo_post_anterior": pd.to_datetime(
                ["2025-03-06T17:00Z", "2026-08-02T22:00Z", "2026-09-06T22:00Z"]
            ),
        }
    )
    mapa = {
        "inscritos_canal": "collected_at",
        "titulo_e_descricao": "collected_at",
        "mediana_anteriores": "ultimo_post_anterior",
        "hora_local": "published_at",
    }
    r = auditar_datas(df, mapa)
    assert r.to_dict() == {
        "inscritos_canal": 1.0,
        "titulo_e_descricao": 1.0,
        "mediana_anteriores": 0.0,
        "hora_local": 0.0,
    }
