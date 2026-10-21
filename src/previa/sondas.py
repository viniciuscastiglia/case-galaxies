"""Validação das sondas de texto do SigLIP nas capas contra rótulos baratos (aula 4.4; F26).

Uma sonda é uma frase ("um close do rosto de uma pessoa") cuja similaridade com a imagem, menos a
média das sondas na mesma imagem, vira uma coluna de features. Antes de ler a H7, a 4.4 pede para
validar cada sonda: o AUC dela contra um rótulo de "o conceito aparece?", e ficar com as que passam
de ~0,7 (exercício "valide as suas sondas"). A 4.4 pede 30 frames rotulados à mão; aqui o rótulo
barato é um campo da rubrica do VLM (previa.vlm) nas capas da amostra, e só 4 das 10 sondas têm um
campo parecido: as outras 6 ficam "sem rótulo" e a validação delas, para depois.

Limitações, que entram no README: (1) o rótulo é do vídeo inteiro ("aparece um rosto em algum
momento") e a sonda é da capa, então um vídeo com rosto só no meio conta como positivo mesmo que a
capa não mostre rosto; o ruído está no rótulo, e por isso o AUC é um piso do que a sonda faz.
(2) O VLM erra também; ele é um rótulo barato, não a verdade. (3) "falando" é medida contra
`tem_rosto`, o campo mais próximo: quem fala aparece, mas aparecer não é falar.

Só pandas e sklearn: nada de torch aqui (LightGBM e torch no mesmo processo dão segfault no Mac).

Uso: uv run python -m previa.sondas
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from previa.amostra import ler_config
from previa.coletor import DADOS

COLUNAS = ["sonda", "rotulo", "n", "positivos", "auc", "passa"]


def validar_sondas(
    sondas: pd.DataFrame, rotulos: pd.DataFrame, pares: dict[str, str], auc_min: float
) -> pd.DataFrame:
    """AUC de cada sonda da capa contra um campo bool da rubrica; uma linha por par.

    `sondas`: índice post_id, colunas `sonda_<nome>_thumb` (maior = o conceito mais presente).
    `rotulos`: índice post_id, colunas bool do VLM. `pares`: nome da sonda -> campo do rótulo.
    Só os post_ids dos dois lados entram (`n` conta a interseção). `passa` é `auc >= auc_min`:
    a direção importa, uma sonda com AUC 0,1 aponta para o lado errado e não passa. Rótulo de
    uma classe só não tem AUC: `auc` e `passa` ficam NaN, sem exceção.
    """
    for nome, campo in pares.items():
        if f"sonda_{nome}_thumb" not in sondas.columns:
            raise ValueError(f"a sonda 'sonda_{nome}_thumb' não está na tabela de sondas")
        if campo not in rotulos.columns:
            raise ValueError(f"o campo '{campo}' (par da sonda '{nome}') não está nos rótulos")
    if not (sondas.index.is_unique and rotulos.index.is_unique):
        raise ValueError("post_id repetido nas sondas ou nos rótulos")
    comuns = sondas.index.intersection(rotulos.index)
    if len(comuns) == 0:
        raise ValueError("nenhum post_id em comum entre as sondas e os rótulos")
    linhas = []
    for nome, campo in pares.items():
        score = sondas.loc[comuns, f"sonda_{nome}_thumb"]
        rotulo = rotulos.loc[comuns, campo]
        if score.isna().any() or rotulo.isna().any():
            raise ValueError(f"NaN na sonda '{nome}' ou no campo '{campo}' dos post_ids em comum")
        y = rotulo.astype(bool)
        auc = roc_auc_score(y, score) if y.nunique() == 2 else np.nan
        linhas.append(
            {
                "sonda": nome,
                "rotulo": campo,
                "n": len(comuns),
                "positivos": int(y.sum()),
                "auc": float(auc),
                "passa": np.nan if np.isnan(auc) else bool(auc >= auc_min),
            }
        )
    r = pd.DataFrame(linhas, columns=COLUNAS)
    r["passa"] = r["passa"].astype(object)  # True, False ou NaN (sem AUC)
    return r


def ler_rubrica(caminho) -> pd.DataFrame:
    """A rubrica por post_id, uma linha por vídeo (a última vence, como em `experimento`)."""
    with open(caminho, encoding="utf-8") as f:
        linhas = [json.loads(x) for x in f if x.strip()]
    return pd.DataFrame(linhas).drop_duplicates("post_id", keep="last").set_index("post_id")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        print("uso: python -m previa.sondas", file=sys.stderr)
        return 2
    cfg = ler_config()
    sig = cfg["siglip"]
    val = sig["validacao"]
    sondas = pd.read_parquet(DADOS / f"sondas_thumb_v{sig['versao']}.parquet")
    sondas = sondas.set_index("video_id")
    rotulos = ler_rubrica(DADOS / "vlm" / f"rubrica_v{cfg['vlm']['versao']}.jsonl")
    r = validar_sondas(sondas, rotulos, val["pares"], val["auc_min"])
    destino = DADOS / f"sondas_auc_v{sig['versao']}.csv"
    r.to_csv(destino, index=False)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    passam = r.loc[r["passa"] == True, "sonda"].tolist()  # noqa: E712 (NaN != True)
    print(f"\npassam de {val['auc_min']}: {', '.join(passam) or 'nenhuma'}")
    todas = [c.removeprefix("sonda_").removesuffix("_thumb") for c in sondas.columns]
    sem_rotulo = [n for n in todas if n not in val["pares"]]
    print(f"sem rótulo barato ({len(sem_rotulo)}): {', '.join(sem_rotulo)}")
    print(f"-> {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
