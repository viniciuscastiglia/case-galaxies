"""Contrato de `prever_criativo`: o que entra e o que sai da tool (A1; aula 6.1).

As regras são do Vinicius (caso à mão de 21/10, `docs/DECISOES.md`, linha "A1"); cada uma
vem como passo em português no comentário, e o teste está em `tests/test_schemas.py`.
O contrato barra na porta o que daria um número errado em silêncio; a lógica da previsão
não mora aqui (o número vem do modelo, princípio 1).
"""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

MAX_DIAS_FUTURO = 90  # 6.1, exercício 1: além disso o histórico do canal estaria velho
MAX_DURACAO_S = 180  # F34: Short tem até 3 minutos (1.5)


def agora() -> datetime:
    """O "agora" do validador dos 90 dias; o teste troca esta função para fixar a data."""
    return datetime.now(UTC)


class _Estrito(BaseModel):
    # Passo 4: campo a mais é recusado (vazamento: não existe antes de publicar), `extra="forbid"`.
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- entrada


class Criativo(_Estrito):
    """O criativo: vídeo (duração e orientação conferidas), imagem (vira a capa) ou só texto."""

    tipo: Literal["video", "imagem", "texto"]
    arquivo: str | None = Field(default=None, description="caminho ou URL; texto não tem arquivo")
    duration_s: float | None = Field(default=None, gt=0)
    vertical: bool | None = None

    @model_validator(mode="after")
    def _confere(self):
        if self.tipo != "texto" and not self.arquivo:
            raise ValueError(
                f"criativo do tipo '{self.tipo}' precisa do arquivo (caminho ou URL); "
                "reenvie com 'arquivo', ou use tipo 'texto' para só a legenda"
            )
        if self.tipo == "video":
            # F34: não é Short (1.5): recusa legível, com o que reenviar.
            if self.duration_s is not None and self.duration_s > MAX_DURACAO_S:
                raise ValueError(
                    f"o vídeo tem {self.duration_s:g} s e o limite é {MAX_DURACAO_S} s: "
                    "o Prévia prevê só YouTube Shorts; reenvie um vídeo de até 180 s"
                )
            if self.vertical is False:
                raise ValueError(
                    "o vídeo é horizontal e o Prévia prevê só Shorts verticais: "
                    "reenvie o vídeo na vertical (vertical=true)"
                )
        return self


class PostDoCanal(_Estrito):
    """Um post do histórico enviado pelo cliente: a data é obrigatória (paridade com o treino)."""

    published_at: AwareDatetime
    likes: int | None = Field(default=None, ge=0)  # None = curtidas ocultas (F10)


class ChannelStats(_Estrito):
    """Canal fora da base: inscritos e posts COM data, que passam pelas mesmas funções do treino."""

    subscribers: int | None = Field(default=None, ge=0)
    posts: list[PostDoCanal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _algo_para_usar(self):
        # Passo 7: sem posts e sem inscritos não há âncora nenhuma.
        if not self.posts and self.subscribers is None:
            raise ValueError(
                "channel_stats vazio: envie os inscritos ('subscribers') ou a lista de posts "
                "('posts', cada um com 'published_at' e 'likes')"
            )
        return self


class PredictionRequest(_Estrito):
    creative: Criativo
    caption: str
    # Passo 2: data sem fuso é recusada (AwareDatetime); a hora local é a do público.
    planned_datetime: AwareDatetime
    # Passo 3: o modelo só viu beleza e automotivo (#22).
    sector: Literal["beleza", "automotivo"]
    # Passo 7: o canal vem da base (`channel_id`) OU como `channel_stats`; exatamente um dos dois.
    channel_id: str | None = None
    channel_stats: ChannelStats | None = None
    cover: str | None = Field(default=None, description="capa enviada pela marca (F11)")
    metric: Literal["likes"] = "likes"
    interval_level: Literal[0.8] = 0.8

    @field_validator("planned_datetime", mode="wrap")
    @classmethod
    def _com_fuso(cls, v, handler):
        try:
            return handler(v)
        except ValidationError as e:
            if any(err["type"] == "timezone_aware" for err in e.errors()):
                raise ValueError(
                    "planned_datetime sem fuso horário: o modelo usa a hora local do público "
                    "(19h em São Paulo é 22h em UTC); reenvie como '2026-10-23T19:00:00-03:00'"
                ) from None
            raise

    @field_validator("planned_datetime")
    @classmethod
    def _ate_90_dias(cls, v: datetime) -> datetime:
        # Passo 5: mais de 90 dias no futuro é recusado. Passo 6: o passado passa (golden set).
        if v - agora() > timedelta(days=MAX_DIAS_FUTURO):
            raise ValueError(
                f"planned_datetime a mais de {MAX_DIAS_FUTURO} dias no futuro: o histórico do "
                "canal estaria velho; reenvie com uma data mais próxima"
            )
        return v

    @model_validator(mode="after")
    def _um_canal(self):
        if (self.channel_id is None) == (self.channel_stats is None):
            raise ValueError(
                "envie apenas um entre 'channel_id' (canal da base) e 'channel_stats' "
                "(inscritos e posts com data); vieram "
                + ("os dois" if self.channel_id is not None else "nenhum dos dois")
            )
        return self


# ---------------------------------------------------------------- saída

Codigo = Literal[
    "intervalo_largo",
    "canal_novo",
    "setor_desconhecido",
    "extracao_falhou",
    "fora_da_distribuicao",
    "data_fora_do_periodo",
]  # os 6 códigos estáveis da aula 5.4


class Motivo(_Estrito):
    codigo: Codigo
    texto: str


class ConfidenceFlag(_Estrito):
    baixa_confianca: bool
    motivos: list[Motivo] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coerente(self):
        if self.baixa_confianca != bool(self.motivos):
            raise ValueError("baixa_confianca deve ser True exatamente quando há motivos no flag")
        return self


class Driver(_Estrito):
    """Os campos de `previa.drivers.top_drivers`."""

    feature: str
    valor: Any = None
    contribuicao_log: float
    multiplicador: float
    direcao: Literal["aumenta", "reduz"]


class Comparavel(_Estrito):
    post_id: str
    published_at: AwareDatetime
    likes: int
    similaridade: float


class Explicacao(_Estrito):
    resumo: str
    fatores: list[str] = Field(default_factory=list)
    ressalvas: list[str] = Field(default_factory=list)


class ModelVersion(_Estrito):
    """O de `previa.modelo.model_version`: de qual modelo veio o número (#37, F23)."""

    versao: str
    run_id: str
    ultimo_treino: str
    base: str
    hash_base: str
    rotulo: str


class PredictionResponse(_Estrito):
    predicted_metric: float = Field(ge=0)
    interval_low: float
    interval_high: float
    relative_to_channel: float  # o multiplicador sobre a âncora
    confidence_flag: ConfidenceFlag
    top_drivers: list[Driver] = Field(max_length=5)
    comparables: list[Comparavel]
    explanation: Explicacao
    features_used: list[str]  # F19
    model_version: ModelVersion

    @model_validator(mode="after")
    def _intervalo(self):
        if self.interval_low > self.predicted_metric:
            raise ValueError("interval_low acima de predicted_metric: intervalo incoerente")
        if self.predicted_metric > self.interval_high:
            raise ValueError("interval_high abaixo de predicted_metric: intervalo incoerente")
        return self
