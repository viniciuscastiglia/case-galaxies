"""Embeddings das thumbnails com SigLIP 2 (decisão #12; aula 4.4).

Roda num processo próprio, nunca junto do LightGBM (torch + LightGBM no mesmo processo dá
segfault no macOS; hub, troubleshooting). Cada thumbnail passa pelo recorte 9:16 do
`previa.thumbs` e vira um vetor de 768 dimensões com norma 1, salvo em lotes de parquet
(`data/emb_thumb_v<versao>/`): o que já está salvo nunca é recalculado (aula 4.1). Ao lado, as
sondas relativas (cosseno com cada frase menos a média das frases na mesma imagem, 4.4), que
entram no modelo como colunas nomeadas. O PCA não mora aqui: ele é ajustado só no treino de cada
fold, no runner (4.4, erro comum 3).

`uv run python -m previa.visual thumbs` embeda as thumbnails já baixadas e refaz as sondas;
`uv run python -m previa.visual aparencia` mede brilho, contraste, saturação e colorfulness da
faixa 9:16 de cada capa, com a mesma `aparencia` dos frames da amostra (4.3).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from previa.amostra import ler_config
from previa.coletor import DADOS
from previa.midia import aparencia
from previa.thumbs import PASTA as PASTA_THUMBS
from previa.thumbs import recortar_9_16


class Siglip:
    """O SigLIP 2 congelado, como na 4.4 (transformers 5: o vetor sai em `.pooler_output`)."""

    def __init__(self, ckpt: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.modelo = AutoModel.from_pretrained(ckpt).eval()
        self.proc = AutoProcessor.from_pretrained(ckpt)

    def _normalizar(self, saida) -> np.ndarray:
        return self.torch.nn.functional.normalize(saida.pooler_output, dim=-1).numpy()

    def embed_imagens(self, imagens: list) -> np.ndarray:
        with self.torch.inference_mode():
            entradas = self.proc(images=imagens, return_tensors="pt")
            return self._normalizar(self.modelo.get_image_features(**entradas))

    def embed_textos(self, textos: list[str]) -> np.ndarray:
        # SigLIP: texto minúsculo e padding fixo de 64 tokens (4.4, erro comum 2)
        with self.torch.inference_mode():
            entradas = self.proc(
                text=[t.lower() for t in textos],
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            return self._normalizar(self.modelo.get_text_features(**entradas))


def preparar_thumb(caminho: Path) -> Image.Image:
    return Image.fromarray(recortar_9_16(np.asarray(Image.open(caminho).convert("RGB"))))


def carregar_embeddings(pasta: Path) -> pd.DataFrame:
    """Todos os lotes salvos: uma linha por vídeo (índice video_id), uma coluna por dimensão."""
    lotes = sorted(Path(pasta).glob("lote_*.parquet"))
    if not lotes:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in lotes]).set_index("video_id")


def embedar_thumbs(ids, pasta_thumbs: Path, destino: Path, encoder, lote=256, log=print) -> dict:
    """Embeda as thumbnails de `ids` que ainda não estão em `destino`, em lotes."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    feitos = set(carregar_embeddings(destino).index)
    faltam = [v for v in ids if v not in feitos and (Path(pasta_thumbs) / f"{v}.jpg").exists()]
    resumo = {
        "novos": len(faltam),
        "pulados": sum(v in feitos for v in ids),
        "sem_thumb": sum(
            v not in feitos and not (Path(pasta_thumbs) / f"{v}.jpg").exists() for v in ids
        ),
    }
    n_lote = len(list(destino.glob("lote_*.parquet")))
    for i in range(0, len(faltam), lote):
        bloco = faltam[i : i + lote]
        emb = encoder.embed_imagens(
            [preparar_thumb(Path(pasta_thumbs) / f"{v}.jpg") for v in bloco]
        )
        df = pd.DataFrame(
            emb.astype(np.float32), columns=[f"e{k:03d}" for k in range(emb.shape[1])]
        )
        df.insert(0, "video_id", bloco)
        arquivo = destino / f"lote_{n_lote:05d}.parquet"
        tmp = arquivo.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, arquivo)  # atômico: nunca fica um lote pela metade
        n_lote += 1
        log(f"{i + len(bloco)}/{len(faltam)} thumbnails embedadas")
    return resumo


def aparencia_thumbs(ids, pasta_thumbs: Path) -> pd.DataFrame:
    """Cor e brilho de cada capa (4.3), na faixa 9:16 que o encoder vê; sem capa, sem linha."""
    linhas = {}
    for v in ids:
        caminho = Path(pasta_thumbs) / f"{v}.jpg"
        if caminho.exists():
            linhas[v] = aparencia(np.asarray(preparar_thumb(caminho)))
    t = pd.DataFrame.from_dict(linhas, orient="index").add_prefix("thumb_")
    t.index.name = "video_id"
    return t


def sondas_relativas(emb_img: np.ndarray, emb_txt: np.ndarray) -> np.ndarray:
    """Similaridade com cada sonda menos a média das sondas na mesma imagem (4.4)."""
    sim = emb_img @ emb_txt.T  # cosseno: tudo tem norma 1
    return sim - sim.mean(axis=1, keepdims=True)


def main(argv=None) -> int:
    acao = (argv or sys.argv[1:] or ["thumbs"])[0]
    if acao not in ("thumbs", "aparencia"):
        print("uso: python -m previa.visual thumbs|aparencia", file=sys.stderr)
        return 2
    cfg = ler_config()["siglip"]
    if acao == "aparencia":
        ids = sorted(p.stem for p in PASTA_THUMBS.glob("*.jpg"))
        t = aparencia_thumbs(ids, PASTA_THUMBS)
        t.reset_index().to_parquet(DADOS / f"aparencia_thumb_v{cfg['versao']}.parquet", index=False)
        print(f"{len(t)} capas com brilho, contraste, saturação e colorfulness", flush=True)
        return 0
    destino = DADOS / f"emb_thumb_v{cfg['versao']}"
    ids = sorted(p.stem for p in PASTA_THUMBS.glob("*.jpg"))
    encoder = Siglip(cfg["ckpt"])
    print(embedar_thumbs(ids, PASTA_THUMBS, destino, encoder, cfg["lote"]), flush=True)
    emb = carregar_embeddings(destino)
    nomes = list(cfg["sondas"])
    rel = sondas_relativas(emb.to_numpy(), encoder.embed_textos(list(cfg["sondas"].values())))
    sondas = pd.DataFrame(rel, index=emb.index, columns=[f"sonda_{n}_thumb" for n in nomes])
    sondas.reset_index().to_parquet(DADOS / f"sondas_thumb_v{cfg['versao']}.parquet", index=False)
    print(f"{len(emb)} thumbnails com embedding e {len(nomes)} sondas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
