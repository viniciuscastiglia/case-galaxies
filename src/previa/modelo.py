"""O modelo de uso (aulas 3.4, 6.1 e 6.3; princípio 4 do CLAUDE.md).

Treinado uma vez no bloco de treino (até 30 dias antes da calibração), salvo com uma versão,
carregado uma vez por processo (`lru_cache`, 6.1), e as features de um Short novo montadas com as
MESMAS funções do treino: `montar_features` sobre os posts do canal mais a linha nova (sem y),
`manuais` sobre a legenda e o PCA de cada bloco de embedding ajustado no treino. O teste de
paridade (tests/test_modelo.py) prova que a previsão vê o que o treino viu.

O treino segue o runner (`previa.experimento`): alvo relativo ao canal (05/10), early stopping nos
últimos `val_dias` do treino e refit com o nº de árvores escolhido. A previsão é a saída do modelo
mais a mediana madura do canal (`referencia`), em log1p(curtidas). O intervalo (quantis e CQR)
chega com a M6; até lá, o bundle tem o modelo pontual.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm as lgb  # antes de qualquer torch no processo (hub, troubleshooting)
import numpy as np
import pandas as pd

from previa.caminhos import ARTEFATOS
from previa.experimento import (
    Config,
    aplicar_pca,
    codigo_setor,
    hash_base,
    preparar,
    treinar_gbm,
)
from previa.features import historico_do_canal, montar_features
from previa.texto import manuais

NOVO = "__novo__"


@dataclass
class Bundle:
    modelo: lgb.Booster
    features: list[str]
    pcas: dict
    versao: str
    lacuna_dias: int
    ultimo_treino: str
    arvores: int
    config: dict = field(default_factory=dict)


def treinar_bundle(cfg: Config, base, duracoes, cfg_av, embeddings=None) -> Bundle:
    """O modelo final da config: todas as linhas do bloco de treino, sem a calibração e o teste."""
    if cfg.extras:  # o caminho dos blocos extras na previsão ainda não existe
        raise ValueError(f"extras ainda não suportados no bundle: {cfg.extras}")
    d = preparar(base, duracoes, cfg_av, cfg.lacuna_dias, min_posts_alvo=cfg.min_posts_alvo)
    fim = cfg_av["cal_inicio"] - pd.Timedelta(days=cfg_av["gap_dias"])
    tr = d.index[d["linha"] & (d["published_at"] < fim)]
    dobra, feats, pcas = d.loc[tr].copy(), list(cfg.features), {}
    for bloco, n_comp in cfg.pca.items():
        cols, pca = aplicar_pca(
            embeddings[bloco], d.loc[tr, "post_id"], dobra["post_id"], n_comp, bloco, cfg.seed
        )
        dobra[list(cols.columns)] = cols.to_numpy()
        feats += list(cols.columns)
        pcas[bloco] = pca
    modelo, n = treinar_gbm(dobra, tr, cfg, feats)
    assinatura = f"{feats}|{hash_base(base)}|{n}".encode()
    versao = f"gbm-{cfg.nome}-{date.today():%Y%m%d}-{hashlib.sha1(assinatura).hexdigest()[:7]}"
    ultimo = str(d.loc[tr, "published_at"].max())
    return Bundle(modelo, feats, pcas, versao, cfg.lacuna_dias, ultimo, n, asdict(cfg))


def salvar(bundle: Bundle, destino=ARTEFATOS) -> Path:
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destino / "modelo.joblib")
    return destino / "modelo.joblib"


@lru_cache(maxsize=1)
def carregar(origem=ARTEFATOS) -> Bundle:
    """Lê o modelo do disco UMA vez por processo; as chamadas seguintes vêm do cache (6.1)."""
    return joblib.load(Path(origem) / "modelo.joblib")


def features_de_um_short(bundle: Bundle, posts_canal: pd.DataFrame, novo: dict) -> pd.DataFrame:
    """As features de um Short ainda não publicado, numa linha, na ordem de `bundle.features`.

    `posts_canal`: os posts do canal (channel_id, published_at UTC, y = log1p das curtidas, NaN se
    ocultas), de qualquer idade: a lacuna de maturação é aplicada aqui, como no treino.
    `novo`: published_at (com fuso), caption, sector, duration_s e `embeddings` (bloco -> vetor).
    """
    canal = posts_canal["channel_id"].iloc[0] if len(posts_canal) else "canal_novo"
    quando = pd.Timestamp(novo["published_at"]).tz_convert("UTC")
    linha = pd.DataFrame({"channel_id": [canal], "published_at": [quando], "y": [np.nan]})
    todas = pd.concat([posts_canal[["channel_id", "published_at", "y"]], linha.set_axis([NOVO])])
    f = montar_features(todas, bundle.lacuna_dias).loc[[NOVO]]
    duracao = novo.get("duration_s")
    f["duracao_s"] = np.nan if duracao is None else float(duracao)
    f["setor"] = codigo_setor(pd.Series([novo["sector"]], index=f.index))
    f = f.join(manuais(pd.Series([novo.get("caption", "")], index=f.index)))
    for bloco, pca in bundle.pcas.items():
        vetor = (novo.get("embeddings") or {}).get(bloco)
        comps = (
            pca.transform(np.asarray(vetor, float)[None, :])[0]
            if vetor is not None
            else np.full(pca.n_components_, np.nan)  # sem embedding: o GBM trata como ausente
        )
        for k, v in enumerate(comps):
            f[f"{bloco}_pc{k:02d}"] = v
    for k, v in bundle.config.get("fixar", {}).items():  # #21: controles fixados na previsão
        f[k] = v
    return f[bundle.features]


def referencia(posts_canal: pd.DataFrame, quando) -> tuple[float, int]:
    """A mediana madura do canal (log1p, 30 dias) em `quando` e quantos posts ela usa.

    É a âncora do alvo relativo (05/10; 1.6): a previsão é essa mediana mais a saída do modelo.
    """
    linha = pd.DataFrame(
        {"channel_id": ["canal"], "published_at": [pd.Timestamp(quando).tz_convert("UTC")]},
    ).assign(y=np.nan)
    posts = posts_canal[["published_at", "y"]].assign(channel_id="canal")
    h = historico_do_canal(pd.concat([posts, linha.set_axis([NOVO])]), 30).loc[NOVO]
    return float(h["hist_mediana"]), int(h["hist_n"])


MIN_POSTS_ANCORA = 3  # #20 (2.1): com menos, o baseline nem existe, e o modelo nunca foi avaliado


def prever_log(bundle: Bundle, x: pd.DataFrame, ref=None, n_ref=None) -> float:
    """log1p das curtidas previstas para a linha `x` (o único lugar de onde sai o número).

    No alvo relativo, `ref` e `n_ref` vêm de `referencia`: sem âncora, ou com menos de 3 posts
    maduros, não há número (a avaliação só cobre esse regime; o flag de confiança é a 5.4).
    """
    r = float(bundle.modelo.predict(x)[0])
    if bundle.config.get("alvo", "relativo") != "relativo":
        return r
    if ref is None or np.isnan(ref):
        raise ValueError("alvo relativo: o canal não tem post maduro para ancorar a previsão")
    if n_ref is None or n_ref < MIN_POSTS_ANCORA:
        raise ValueError(
            f"o canal tem {n_ref} posts maduros; o modelo só foi avaliado com "
            f"{MIN_POSTS_ANCORA} posts ou mais (#20)"
        )
    return r + float(ref)
