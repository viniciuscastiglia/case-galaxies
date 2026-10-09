"""Base sintética do módulo 2 da trilha (o mesmo gerador em todas as aulas, 2.1 a 2.6).

Cópia fiel do `gerar_base()` das aulas, para os exercícios importarem em vez de colarem:
    from previa.sintetico import gerar_base
Os efeitos do conteúdo são conhecidos, o que permite checar se a avaliação conta a verdade.
NÃO são dados reais.
"""

import numpy as np
import pandas as pd


def gerar_base(n_canais=60, seed=42):
    """Shorts sintéticos: nível próprio por canal + tendência lenta + ruído."""
    rng = np.random.default_rng(seed)
    efeito_rosto = {"moda": 0.15, "beleza": 0.35, "automotivo": 0.05}
    partes = []
    for c in range(n_canais):
        setor = ["moda", "beleza", "automotivo"][c % 3]
        inscritos = int(np.exp(np.clip(rng.normal(11.3, 1.3), 8, 15.5)))
        nivel = 0.85 * np.log(inscritos) - 4.2 + rng.normal(0, 0.55)
        tendencia = np.cumsum(rng.normal(rng.normal(0, 0.03), 0.09, 32))  # por mês
        inicio = rng.integers(0, 400) if rng.random() < 0.85 else rng.integers(700, 900)
        n = max(8, int((960 - inicio) / rng.uniform(6, 20)))  # 1 post a cada 6-20 dias
        dia = np.sort(rng.integers(inicio, 960, n))
        hora = np.where(rng.random(n) < 0.4, rng.integers(18, 23, n), rng.integers(7, 24, n))
        duracao = np.clip(rng.normal(38 if setor == "automotivo" else 28, 12, n), 8, 90).round()
        rosto = rng.random(n) < (0.35 if setor == "automotivo" else 0.7)
        texto_tela = rng.random(n) < 0.5
        cortes_10s = np.clip(rng.normal(2.5, 1.0, n), 0, 6).round(1)
        pergunta = rng.random(n) < 0.3
        gancho = np.clip(rng.normal(5, 2, n), 0, 10).round()  # nota do VLM, 0-10
        conteudo = (
            0.09 * (gancho - 5)
            + efeito_rosto[setor] * rosto
            + 0.15 * texto_tela
            + 0.05 * (cortes_10s - 2.5)
            - 0.05 * (duracao - 30) / 10
            + 0.06 * pergunta
            + 0.12 * ((hora >= 18) & (hora <= 22))
        )
        viral = rng.random(n) < 0.03
        y = (
            nivel
            + tendencia[dia // 30]
            + conteudo
            + rng.normal(0, 0.7, n)
            + viral * rng.normal(1.5, 0.4, n)
        )
        curtidas = np.maximum(0, np.round(np.expm1(y)))
        regime = np.where(dia >= 455, 1.7, 1.0)  # replays contam desde 31/03/2025
        views = np.round((curtidas + 1) * np.exp(rng.normal(np.log(35), 0.35, n)) * regime)
        partes.append(
            pd.DataFrame(
                {
                    "canal": f"c{c:02d}",
                    "setor": setor,
                    "inscritos": inscritos,
                    "data_pub": pd.Timestamp("2024-01-01")
                    + pd.to_timedelta(dia, "D")
                    + pd.to_timedelta(hora, "h"),
                    "hora": hora,
                    "duracao_s": duracao,
                    "rosto": rosto,
                    "texto_tela": texto_tela,
                    "cortes_10s": cortes_10s,
                    "titulo_pergunta": pergunta,
                    "legenda_chars": np.clip(rng.normal(90, 45, n), 0, None).round(),
                    "gancho_vlm": gancho,
                    "curtidas": curtidas,
                    "views": views,
                    "comentarios": rng.poisson(curtidas * 0.02),
                }
            )
        )
    df = pd.concat(partes, ignore_index=True)
    df["y"] = np.log1p(df["curtidas"])
    return df.sort_values(["data_pub", "canal"]).reset_index(drop=True)
