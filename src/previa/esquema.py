"""Esquema mínimo do briefing, em código (aula 1.8): Pydantic valida cada registro e pandera
valida a tabela de treino. Rode a validação a cada exportação."""

from typing import Any, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from previa import duplicatas
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


def _em_utc(s: pd.Series) -> bool:
    """A coluna é datetime com fuso UTC, de qualquer unidade (a base real é `us`, a sintética
    pode ser `ns`): compara o dtype com o UTC na unidade da própria coluna, sem coagir."""
    return isinstance(s.dtype, pd.DatetimeTZDtype) and s.dtype == pd.DatetimeTZDtype(
        s.dtype.unit, "UTC"
    )


class TabelaTreino(pa.DataFrameModel):
    """A tabela do `preparar` para o treino: só os invariantes (princípio 6; aulas 1.8 e 3.5).

    Não é estrita: a tabela tem dezenas de colunas (features, baseline, faixas). Cada nulo daqui é
    legítimo e fica de fora das regras: `hist_*` e `hist_ultimo_ts` nos primeiros posts do canal,
    `duracao_s` quando a duração falta. O que não pode: `y` nulo ou negativo, `setor` fora de 0/1,
    `post_id` repetido, histórico que não é anterior à publicação (o teste da M3, princípio 2).
    Valide só as linhas de treino (`validar_tabela_de_treino`): o resto da tabela é passado.
    """

    post_id: Series[str] = pa.Field(unique=True)
    channel_id: Series[str]
    published_at: Series[Any]  # fuso UTC conferido abaixo; a unidade (us ou ns) fica livre
    y: Series[float] = pa.Field(ge=0)  # log1p das curtidas, nunca nulo
    setor: Series[float] = pa.Field(isin=[0.0, 1.0])
    hist_n: Series[int] = pa.Field(ge=0)
    duracao_s: Series[float] = pa.Field(ge=0, nullable=True)
    hist_ultimo_ts: Series[Any] = pa.Field(nullable=True)  # NaT nos primeiros posts do canal

    @pa.check("published_at", "hist_ultimo_ts", name="datetime_em_utc")
    def datetime_em_utc(cls, s: pd.Series) -> bool:
        return _em_utc(s)

    @pa.dataframe_check
    def historico_anterior_a_publicacao(cls, df: pd.DataFrame) -> Series[bool]:
        # checagem no DataFrame: onde há histórico, o último post usado é estritamente anterior
        ultimo = df["hist_ultimo_ts"]
        return ultimo.isna() | (ultimo < df["published_at"])


def validar_tabela_de_treino(d: pd.DataFrame, features: list[str]) -> None:
    """O contrato da tabela de treino nas linhas de treino (`d["linha"]`), antes de qualquer fit.

    Valida a `TabelaTreino` (`lazy=True`: todas as falhas de uma vez, `SchemaErrors`) e confere que
    cada feature existe, é numérica e não tem ±inf, só nas linhas de treino. NaN pode: o LightGBM
    trata como ausente (aula 3.5). Falha nunca silenciosa (princípio 7): `ValueError` com os nomes.
    """
    if "linha" not in d.columns:
        raise ValueError("a tabela de treino não tem a coluna `linha` (rode o preparar)")
    treino = d[d["linha"]]
    TabelaTreino.validate(treino, lazy=True)
    ausentes = [f for f in features if f not in d.columns]
    if ausentes:
        raise ValueError(f"features fora da tabela de treino: {ausentes}")
    texto = [f for f in features if not pd.api.types.is_numeric_dtype(treino[f])]
    if texto:
        raise ValueError(f"features não numéricas na tabela de treino: {texto}")
    infinitas = [
        f for f in features if np.isinf(treino[f].to_numpy(dtype=float, na_value=np.nan)).any()
    ]
    if infinitas:
        raise ValueError(f"features com ±inf na tabela de treino (só NaN é permitido): {infinitas}")


def montar_base_minima(
    rotulados: pd.DataFrame, setores: dict[str, str], inscritos: pd.DataFrame, hashes=None
) -> pd.DataFrame:
    """Shorts rotulados (consultar_rotulados) no esquema mínimo, com o setor da planilha e os
    inscritos do dia do rótulo (canais_snapshots). Os inscritos ficam para estratificação e
    análise de erros, fora das features de treino (aula 1.7). `hashes` são os dHash das
    thumbnails (duplicatas.hashes_das_thumbnails), usados na regra das duplicatas."""
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
            "duplicate_group": duplicatas.grupo_de_duplicatas(base, hashes),
        }
    )


def exportar_base_minima(
    con,
    caminho_canais=DADOS / "canais.csv",
    destino=DADOS / "base_minima.parquet",
    cache=duplicatas.CACHE,
    baixar=None,
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
    # thumbnails só dos posts com cópia de título e duração; o cache baixa só as novas
    hashes = duplicatas.hashes_das_thumbnails(duplicatas.candidatos(rotulados), cache, baixar)
    base = montar_base_minima(rotulados, setores, inscritos, hashes)
    BaseShorts.validate(base, lazy=True)  # erro aqui: corrija na coleta, não na validação
    base.to_parquet(destino, index=False)
    return len(base)
