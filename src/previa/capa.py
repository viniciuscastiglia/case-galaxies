"""As features da capa de um criativo novo, na hora da previsão (F17 do lado da previsão; F11).

Paridade treino/uso (princípio 4; aula 4.4): a capa passa pelas MESMAS funções que geraram as
colunas do treino: o recorte 9:16 (`previa.visual.preparar_thumb`), o encoder do SigLIP, as sondas
relativas com as frases de `configs/multimodal.yaml` e a `aparencia` do `previa.midia`.

O encoder real é o `previa.visual.Siglip`. Quem serve a tool chama `previa.openmp.um_runtime()`
antes dos imports (LightGBM e torch no mesmo processo; CLAUDE.md, OpenMP). Este módulo não importa
torch: o encoder entra como argumento.

Cache pelo conteúdo (6.3, Tool 1: idempotência e cache): a chave é o SHA-256 dos bytes da imagem e a
versão do pacote, não o caminho. A mesma capa reenviada (o A/B de um canal, uma nova tentativa do
agente) não passa de novo pelo SigLIP; um arquivo trocado no mesmo caminho é recalculado. O cache é
do processo, com até `MAX_CACHE` capas, e devolve cópias (quem muda o resultado não muda o cache).
"""

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from previa.amostra import ler_config
from previa.midia import aparencia
from previa.visual import preparar_thumb, sondas_relativas

BLOCOS_DA_CAPA = ("sondas_thumb", "aparencia_thumb")
MAX_CACHE = 256
_CACHE: OrderedDict = OrderedDict()


def limpar_cache_da_capa() -> None:
    """Esvazia o cache das capas (os testes começam com ele vazio)."""
    _CACHE.clear()


def _copia(valores: dict, emb: np.ndarray) -> tuple[dict, dict, list[str]]:
    return dict(valores), {"thumb": np.array(emb, dtype=float, copy=True)}, []


def extras_da_capa(caminho_imagem: Path | None, bundle, encoder) -> tuple[dict, dict, list[str]]:
    """(valores das colunas extras da capa, `{"thumb": vetor}` para o PCA, blocos ausentes).

    Sem imagem (F11), nenhum valor, `{"thumb": None}` e os dois blocos como ausentes: o
    `features_de_um_short` os transforma em NaN e o motivo vai para o flag. Imagem que não abre
    levanta (princípio 7); coluna do pacote que a capa não sabe calcular também.
    """
    if caminho_imagem is None:
        return {}, {"thumb": None}, list(BLOCOS_DA_CAPA)
    caminho = Path(caminho_imagem)
    if not caminho.is_file():
        raise FileNotFoundError(f"capa não encontrada: {caminho}")
    chave = (hashlib.sha256(caminho.read_bytes()).hexdigest(), getattr(bundle, "versao", ""))
    if chave in _CACHE:
        _CACHE.move_to_end(chave)
        return _copia(*_CACHE[chave])
    imagem = preparar_thumb(caminho)  # o mesmo recorte 9:16 do treino
    emb = encoder.embed_imagens([imagem])
    sondas = ler_config()["siglip"]["sondas"]
    rel = sondas_relativas(emb, encoder.embed_textos(list(sondas.values())))[0]
    valores = {f"sonda_{n}_thumb": float(v) for n, v in zip(sondas, rel, strict=True)}
    valores |= {f"thumb_{k}": float(v) for k, v in aparencia(np.asarray(imagem)).items()}
    esperadas = {c for b in BLOCOS_DA_CAPA for c in bundle.colunas_extras.get(b, [])}
    if faltam := sorted(esperadas - set(valores)):
        raise ValueError(f"o pacote espera colunas da capa que não sei calcular: {faltam}")
    _CACHE[chave] = (valores, np.asarray(emb[0], float))
    if len(_CACHE) > MAX_CACHE:
        _CACHE.popitem(last=False)  # a mais antiga sai primeiro
    return _copia(*_CACHE[chave])
