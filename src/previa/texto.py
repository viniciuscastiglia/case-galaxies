"""Features de texto do título (decisão #9; aula 3.6).

Só o título, a primeira linha da `caption` (título + descrição): a descrição padrão dos canais
vira um identificador de canal (3.6, erro comum 4). Dois blocos:

- manuais (`features_de_texto`): forma do título e termos dos dois setores; baratas, rodam no
  runner;
- embeddings multilíngues (MiniLM, 384 dimensões): num processo próprio, com torch, e em cache
  por `video_id` + hash do título (erro comum 3). O PCA de 16 é ajustado só no treino de cada
  fold, no runner (erro comum 1).

O título vem da API na data da coleta e pode ter sido editado depois da publicação (auditoria
das features, 2.2: risco temporal brando; vai para o data card).

`uv run python -m previa.texto embeddings` codifica os títulos da base que ainda não estão no cache.
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from previa.coletor import DADOS

EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿⬀-⯿️]")
CTA = re.compile(r"link na bio|comenta|salva|segue|compartilha|marca (um|uma|alguém)", re.I)
PT = {
    "de",
    "do",
    "da",
    "que",
    "com",
    "para",
    "por",
    "em",
    "um",
    "uma",
    "você",
    "não",
    "sobre",
    "isso",
}
EN = {"the", "and", "with", "for", "to", "of", "in", "you", "is", "this", "my", "how"}
TERMOS_SETOR = {  # termos dos dois setores (#9); contados juntos: o setor já é uma feature
    "maquiagem", "make", "pele", "batom", "base", "skincare", "cabelo", "unha", "unhas",
    "perfume", "rímel", "sérum", "protetor", "hidratante", "sombra", "delineado",
    "carro", "carros", "motor", "pneu", "pneus", "roda", "rodas", "freio", "câmbio", "suv",
    "sedan", "picape", "moto", "km", "oficina", "revisão", "elétrico", "híbrido",
}  # fmt: skip


def titulo_de(caption: pd.Series) -> pd.Series:
    return caption.fillna("").str.split("\n", n=1).str[0]


def features_de_texto(texto: str) -> dict:
    """As colunas manuais da 3.6, mais os termos do setor."""
    letras = [c for c in texto if c.isalpha()]
    palavras = re.findall(r"\w+", texto.lower())
    return {
        "txt_caracteres": len(texto),
        "txt_palavras": len(palavras),
        "txt_hashtags": len(re.findall(r"#\w+", texto)),
        "txt_emojis": len(EMOJI.findall(texto)),
        "txt_pergunta": int("?" in texto),
        "txt_cta": int(bool(CTA.search(texto))),
        "txt_numeros": len(re.findall(r"\d+", texto)),
        "txt_frac_maiusculas": sum(c.isupper() for c in letras) / max(len(letras), 1),
        "txt_ingles": int(len(EN & set(palavras)) > len(PT & set(palavras))),
        "txt_termos_setor": sum(p in TERMOS_SETOR for p in palavras),
    }


def manuais(caption: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([features_de_texto(t) for t in titulo_de(caption)], index=caption.index)


def _chave(video_id: str, texto: str) -> str:
    return f"{video_id}:{hashlib.sha1(texto.encode()).hexdigest()[:12]}"


def embeddings_com_cache(ids, textos, encoder, arquivo: Path) -> pd.DataFrame:
    """Um vetor por título; só codifica o que não está no cache (chave: video_id + hash)."""
    arquivo = Path(arquivo)
    chaves = [_chave(i, t) for i, t in zip(ids, textos, strict=True)]
    cache = pd.read_parquet(arquivo) if arquivo.exists() else None
    conhecidas = set() if cache is None else set(cache.index)
    faltam = [(k, t) for k, t in zip(chaves, textos, strict=True) if k not in conhecidas]
    if faltam:
        vet = encoder.encode([t for _, t in faltam], batch_size=64, normalize_embeddings=True)
        cols = [f"t{k:03d}" for k in range(vet.shape[1])]
        novos = pd.DataFrame(
            np.asarray(vet, np.float32), index=[k for k, _ in faltam], columns=cols
        )
        cache = novos if cache is None else pd.concat([cache, novos])
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(arquivo)
    return cache.loc[chaves].set_axis(list(ids))


def ler_embeddings(arquivo: Path) -> pd.DataFrame:
    """O cache inteiro com índice video_id (a versão mais recente de cada título)."""
    e = pd.read_parquet(arquivo)
    e.index = e.index.str.split(":").str[0]
    return e[~e.index.duplicated(keep="last")]


def main(argv=None) -> int:
    import yaml
    from sentence_transformers import SentenceTransformer

    if (argv or sys.argv[1:] or ["embeddings"])[0] != "embeddings":
        print("uso: python -m previa.texto embeddings", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(Path("configs/features.yaml").read_text(encoding="utf-8"))["texto"]
    base = pd.read_parquet(DADOS / "base_minima.parquet", columns=["post_id", "caption"])
    arquivo = DADOS / f"emb_titulo_v{cfg['versao']}.parquet"
    encoder = SentenceTransformer(cfg["modelo"], device="cpu")
    e = embeddings_com_cache(
        base["post_id"].tolist(), titulo_de(base["caption"]).tolist(), encoder, arquivo
    )
    print(f"{len(e)} títulos com embedding ({e.shape[1]} dimensões) -> {arquivo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
