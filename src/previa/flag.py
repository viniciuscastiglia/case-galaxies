"""O sinal de baixa confiança do contrato (aula 5.4; decisão #14).

Regras explícitas, cada uma com um código estável e uma frase para o marketing. O flag só
descreve: o número e o intervalo seguem na resposta (5.4, "Não esconda o número"). Os limiares
saem da calibração, uma vez, e vão no pacote do modelo (EC2); nunca são recalculados por chamada.

`setor_desconhecido` não é usado: o contrato recusa setor fora de beleza/automotivo (#22).
`fora_da_distribuicao` só existe quando a tool tem a capa e o índice dos comparáveis com o limiar:
a distância (1 - cosseno) ao vizinho maduro mais parecido, contra o p99 da calibração do pacote,
calculado uma vez em `previa.indice` (decisão #14).
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from previa.caminhos import CONFIGS
from previa.schemas import ConfidenceFlag, Motivo


def carregar_cfg(caminho=None) -> dict:
    """`configs/flag.yaml`: percentis e limites das regras."""
    caminho = caminho or CONFIGS / "flag.yaml"
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def _br(x: float) -> str:
    """Número no formato brasileiro, uma casa (5.4: `_br`)."""
    return f"{x:.1f}".replace(".", ",")


def _br2(x: float) -> str:
    """Duas casas: a distância de cosseno vive entre 0 e 1, e uma casa só esconderia a diferença."""
    return f"{x:.2f}".replace(".", ",")


def limiares_da_calibracao(
    largura_log_cal: pd.Series,
    dist_knn_cal: pd.Series | None,
    ultimo_treino,
    cfg: dict,
) -> dict:
    """Os limiares, calculados na CALIBRAÇÃO (nunca no treino, EC3; nunca no caso, EC2).

    Devolve um dict JSON-serializável para o pacote do modelo.
    """
    dist = None
    if dist_knn_cal is not None and len(dist_knn_cal.dropna()) > 0:
        dist = float(np.nanpercentile(dist_knn_cal.dropna(), cfg["percentil_knn"]))
    return {
        "largura_log": float(np.percentile(largura_log_cal.dropna(), cfg["percentil_largura"])),
        "dist_knn": dist,
        "min_posts": int(cfg["min_posts"]),
        "max_dias": int(cfg["max_dias"]),
        "max_dias_passado": int(cfg["max_dias_passado"]),
        "ultimo_treino": pd.Timestamp(ultimo_treino).date().isoformat(),
    }


def _lista(itens: list[str]) -> str:
    if len(itens) <= 1:
        return "".join(itens)
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def confidence_flag(
    largura_log: float,
    posts_maduros: int,
    extracao_falhou: list[str],
    dist_knn: float | None,
    data_planejada: datetime,
    limiares: dict,
    curtidas_ocultas: int = 0,
) -> ConfidenceFlag:
    """As regras da 5.4; cada motivo diz o que resolveria, quando existe resolução.

    `curtidas_ocultas`: posts que já seriam maduros, mas sem curtidas (o canal as esconde, F10).
    Eles não contam como maduros; com `canal_novo` ligado, o texto diz isso, para o canal não
    parecer novo sem explicação. O código continua `canal_novo` (os 6 códigos estáveis da 5.4).
    """
    L = limiares
    motivos: list[Motivo] = []

    def motivo(codigo, texto):
        motivos.append(Motivo(codigo=codigo, texto=texto))

    if largura_log > L["largura_log"]:
        motivo(
            "intervalo_largo",
            f"o limite de cima é {_br(np.exp(largura_log))} vezes o de baixo "
            f"(em 80% dos casos, fica abaixo de {_br(np.exp(L['largura_log']))} vezes)",
        )
    if posts_maduros < L["min_posts"]:
        faltam = L["min_posts"] - posts_maduros
        texto = f"o canal tem só {posts_maduros} posts maduros (publicados há mais de 30 dias); "
        if (
            curtidas_ocultas
        ):  # publicar mais não resolve: os novos também viriam com as curtidas ocultas
            texto += (
                f"{curtidas_ocultas} posts têm as curtidas ocultas e não entram na conta; com as "
                "curtidas deles enviadas em channel_stats, a estimativa do canal melhora"
            )
        else:
            texto += f"com mais {faltam} posts publicados, a estimativa do canal melhora"
        motivo("canal_novo", texto)
    if extracao_falhou:
        motivo(
            "extracao_falhou",
            f"não foi possível analisar {_lista(extracao_falhou)}; a previsão usa só o que "
            f"foi extraído; reenvie {_lista(extracao_falhou)} para análise",
        )
    elif dist_knn is not None and L.get("dist_knn") is not None and dist_knn > L["dist_knn"]:
        motivo(
            "fora_da_distribuicao",
            f"o criativo é diferente de tudo que o modelo viu "
            f"(distância {_br2(dist_knn)}, o normal vai até {_br2(L['dist_knn'])})",
        )
    plan = data_planejada.date() if isinstance(data_planejada, datetime) else data_planejada
    dias = (plan - date.fromisoformat(L["ultimo_treino"])).days
    if dias > L["max_dias"] or dias < -L["max_dias_passado"]:
        sentido = "depois" if dias > 0 else "antes"
        motivo(
            "data_fora_do_periodo",
            f"a publicação está a {abs(dias)} dias {sentido} do fim dos dados do modelo; "
            "com um modelo retreinado perto dessa data, a previsão fica mais firme",
        )
    return ConfidenceFlag(baixa_confianca=bool(motivos), motivos=motivos)


def avaliar_flag(y, pred, motivos_por_linha: list[list[str]]) -> pd.DataFrame:
    """O teste que dá sentido ao flag (5.4, EC1): MAE dos sinalizados contra os demais.

    `motivos_por_linha`: os códigos de cada linha (lista vazia = não sinalizada). A taxa de
    sinalização vai na coluna `taxa` (decisão #14: 15% a 35%).
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if not (len(y) == len(pred) == len(motivos_por_linha)):
        raise ValueError(
            f"tamanho diferente: y={len(y)}, pred={len(pred)}, motivos={len(motivos_por_linha)}"
        )
    erro = np.abs(y - pred)
    sinal = np.array([len(m) > 0 for m in motivos_por_linha])
    n = len(y)

    def linha(grupo, mascara):
        k = int(mascara.sum())
        return {
            "grupo": grupo,
            "n": k,
            "taxa": k / n if n else np.nan,
            "mae": float(erro[mascara].mean()) if k else np.nan,
        }

    linhas = [linha("sinalizados", sinal), linha("nao_sinalizados", ~sinal)]
    codigos = sorted({c for m in motivos_por_linha for c in m})
    for c in codigos:
        linhas.append(linha(c, np.array([c in m for m in motivos_por_linha])))
    return pd.DataFrame(linhas)
