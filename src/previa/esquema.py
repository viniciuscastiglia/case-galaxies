"""Esquema mínimo do briefing, em código (aula 1.8): Pydantic valida cada registro e pandera
valida a tabela de treino. Rode a validação a cada exportação."""

from typing import Literal

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from previa.coletor import DADOS, consultar_rotulados, ler_canais

SETORES = ["moda", "beleza", "automotivo"]


class ChannelStats(BaseModel):
    """Estatísticas do canal NO MOMENTO DA COLETA (não da publicação)."""

    subscribers: int | None = Field(default=None, ge=0)  # None = canal oculta inscritos
    video_count: int = Field(ge=0)
    collected_at: AwareDatetime


class Post(BaseModel):
    """Esquema mínimo do briefing, um registro por Short."""

    post_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")  # formato observado dos IDs
    channel_id: str = Field(pattern=r"^UC[A-Za-z0-9_-]{22}$")
    published_at: AwareDatetime
    caption: str  # título + descrição
    media_ref: str = Field(pattern=r"^https://www\.youtube\.com/shorts/")  # link, não arquivo
    metric_target: int | None = Field(ge=0)  # likeCount; None = oculto
    target_age_days: float = Field(ge=30)  # idade no snapshot do rótulo
    channel_stats_at_collection: ChannelStats
    sector: Literal["moda", "beleza", "automotivo"]

    @model_validator(mode="after")
    def coleta_depois_da_publicacao(self):
        if self.channel_stats_at_collection.collected_at <= self.published_at:
            raise ValueError("coleta anterior à publicação: confira fusos e datas")
        return self


class BaseShorts(pa.DataFrameModel):
    """A tabela achatada que vai para o treino (um Short por linha)."""

    post_id: Series[str] = pa.Field(unique=True, str_matches=r"^[A-Za-z0-9_-]{11}$")
    channel_id: Series[str] = pa.Field(str_startswith="UC")
    published_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "us", "tz": "UTC"})
    caption: Series[str]
    media_ref: Series[str] = pa.Field(str_startswith="https://www.youtube.com/shorts/")
    metric_target: Series[pd.Int64Dtype] = pa.Field(ge=0, nullable=True)
    target_age_days: Series[float] = pa.Field(ge=30)
    subscribers_at_collection: Series[pd.Int64Dtype] = pa.Field(ge=0, nullable=True)
    collected_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "us", "tz": "UTC"})
    sector: Series[str] = pa.Field(isin=SETORES)
    # post_id da primeira cópia do mesmo criativo (o próprio post_id quando não há cópia)
    duplicate_group: Series[str] = pa.Field(str_matches=r"^[A-Za-z0-9_-]{11}$")

    @pa.dataframe_check
    def coleta_depois_da_publicacao(cls, df: pd.DataFrame) -> Series[bool]:
        return df["collected_at"] > df["published_at"]

    @pa.dataframe_check
    def no_maximo_10pct_sem_curtidas(cls, df: pd.DataFrame) -> bool:
        # checagem no DataFrame: checks de coluna ignoram os nulos quando nullable=True
        return df["metric_target"].isna().mean() <= 0.10

    class Config:
        strict = True  # coluna extra ou faltando = erro
        coerce = True  # converte tipos quando possível (ex.: "2410" -> 2410)


def grupo_de_duplicatas(rotulados: pd.DataFrame) -> pd.Series:
    """Duplicatas e reuploads (aula 2.2, família 5): título normalizado (caixa e espaços) e
    duração iguais marcam o mesmo criativo, inclusive entre canais. Cada post recebe o
    video_id da primeira cópia publicada. Só marca: a regra (manter uma cópia ou deixar as
    cópias do mesmo lado do split) é decisão do experimento."""
    titulo = rotulados["title"].fillna("").str.lower().str.split().str.join(" ")
    chave = titulo + "|" + rotulados["duration_s"].astype(str)
    ordem = rotulados.assign(chave=chave).sort_values(["published_at", "video_id"])
    primeira = ordem.groupby("chave")["video_id"].first()
    return chave.map(primeira)


def montar_base_minima(
    rotulados: pd.DataFrame, setores: dict[str, str], inscritos: pd.DataFrame
) -> pd.DataFrame:
    """Shorts rotulados (consultar_rotulados) no esquema mínimo, com o setor da planilha e os
    inscritos do dia do rótulo (canais_snapshots). Os inscritos ficam para estratificação e
    análise de erros, fora das features de treino (aula 1.7)."""
    base = rotulados.assign(collected_date=rotulados["collected_at"].dt.date)
    base = base.merge(inscritos, on=["channel_id", "collected_date"], how="left")
    return pd.DataFrame(
        {
            "post_id": base["video_id"],
            "channel_id": base["channel_id"],
            "published_at": base["published_at"],
            "caption": base["title"].fillna("") + "\n" + base["description"].fillna(""),
            "media_ref": "https://www.youtube.com/shorts/" + base["video_id"],
            "metric_target": base["likes"].astype("Int64"),
            "target_age_days": base["age_days"].astype(float),
            "subscribers_at_collection": base["subscribers"].astype("Int64"),
            "collected_at": base["collected_at"],
            "sector": base["channel_id"].map(setores),
            "duplicate_group": grupo_de_duplicatas(base),
        }
    )


def exportar_base_minima(
    con, caminho_canais=DADOS / "canais.csv", destino=DADOS / "base_minima.parquet"
) -> int:
    planilha = pd.read_csv(caminho_canais, dtype=str)
    setores = dict(zip(planilha["channel_id"], planilha["setor"], strict=False))
    inscritos = con.execute(
        "SELECT channel_id, collected_date, subscribers FROM canais_snapshots"
    ).df()
    inscritos["collected_date"] = pd.to_datetime(inscritos["collected_date"]).dt.date
    rotulados = consultar_rotulados(con)
    # só os canais que seguem incluídos: a revisão pode excluir um canal depois da coleta
    rotulados = rotulados[rotulados["channel_id"].isin(set(ler_canais(caminho_canais)))]
    base = montar_base_minima(rotulados, setores, inscritos)
    BaseShorts.validate(base, lazy=True)  # erro aqui: corrija na coleta, não na validação
    base.to_parquet(destino, index=False)
    return len(base)
