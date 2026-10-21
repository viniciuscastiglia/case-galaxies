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
`tem_rosto`, o campo mais próximo: quem fala aparece, mas aparecer não é falar. (4) O prompt do VLM
define `close_up_produto` por setor (em beleza, frasco, batom ou paleta; em automotivo, detalhe do
carro: farol, roda, painel), então um par pode trazer um `setor`: `produto_beleza` só vale na
beleza e `roda` só no automotivo; com os dois setores juntos, o AUC cai para perto de 0,5 sem que a
sonda esteja errada (revisão final). Os pares sem `setor` valem nos dois.

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

COLUNAS = ["sonda", "rotulo", "setor", "n", "positivos", "auc", "passa"]
SEM_FILTRO = "todos"


def _normalizar_par(nome: str, par) -> tuple[str, str | None]:
    """(campo do rótulo, setor ou None) de um par: `campo` ou `{rotulo: campo, setor: setor}`."""
    if isinstance(par, str):
        return par, None
    if not isinstance(par, dict) or "rotulo" not in par or not set(par) <= {"rotulo", "setor"}:
        raise ValueError(
            f"par da sonda '{nome}': use o nome do campo ou {{rotulo: campo, setor: setor}} "
            f"(chaves aceitas: rotulo, setor); veio {par!r}"
        )
    return par["rotulo"], par.get("setor")


def validar_sondas(
    sondas: pd.DataFrame,
    rotulos: pd.DataFrame,
    pares: dict,
    auc_min: float,
    setores: pd.Series | None = None,
) -> pd.DataFrame:
    """AUC de cada sonda da capa contra um campo bool da rubrica; uma linha por par.

    `sondas`: índice post_id, colunas `sonda_<nome>_thumb` (maior = o conceito mais presente).
    `rotulos`: índice post_id, colunas bool do VLM. `pares`: nome da sonda -> campo do rótulo, ou
    -> `{rotulo: campo, setor: setor}` para medir só nos posts daquele setor (o prompt do VLM
    define o campo por setor); `setores` é o setor de cada post_id e só é exigido por quem usa
    `setor`. Só os post_ids dos dois lados (e do setor, quando há filtro) entram: `n` conta essa
    interseção e a coluna `setor` traz o filtro, "todos" quando não há. `passa` é `auc >= auc_min`:
    a direção importa, uma sonda com AUC 0,1 aponta para o lado errado e não passa. Rótulo de
    uma classe só não tem AUC: `auc` e `passa` ficam NaN, sem exceção.
    """
    normais = {nome: _normalizar_par(nome, par) for nome, par in pares.items()}
    for nome, (campo, setor) in normais.items():
        if f"sonda_{nome}_thumb" not in sondas.columns:
            raise ValueError(f"a sonda 'sonda_{nome}_thumb' não está na tabela de sondas")
        if campo not in rotulos.columns:
            raise ValueError(f"o campo '{campo}' (par da sonda '{nome}') não está nos rótulos")
        if setor is not None and setores is None:
            raise ValueError(f"a sonda '{nome}' pede o setor '{setor}', mas faltam os setores")
    if not (sondas.index.is_unique and rotulos.index.is_unique):
        raise ValueError("post_id repetido nas sondas ou nos rótulos")
    if setores is not None and not setores.index.is_unique:
        raise ValueError("post_id repetido nos setores")
    comuns = sondas.index.intersection(rotulos.index)
    if len(comuns) == 0:
        raise ValueError("nenhum post_id em comum entre as sondas e os rótulos")
    linhas = []
    for nome, (campo, setor) in normais.items():
        ids = comuns
        if setor is not None:
            ids = comuns[setores.reindex(comuns).to_numpy() == setor]
            if len(ids) == 0:
                raise ValueError(f"nenhum post do setor '{setor}' em comum (sonda '{nome}')")
        score = sondas.loc[ids, f"sonda_{nome}_thumb"]
        rotulo = rotulos.loc[ids, campo]
        if score.isna().any() or rotulo.isna().any():
            raise ValueError(f"NaN na sonda '{nome}' ou no campo '{campo}' dos post_ids em comum")
        y = rotulo.astype(bool)
        auc = roc_auc_score(y, score) if y.nunique() == 2 else np.nan
        linhas.append(
            {
                "sonda": nome,
                "rotulo": campo,
                "setor": setor or SEM_FILTRO,
                "n": len(ids),
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


def ressalvas(sondas: pd.DataFrame, rotulos: pd.DataFrame, pares: dict) -> list[str]:
    """As ressalvas que acompanham a tabela, uma linha cada (limitações 1 e 3 do módulo)."""
    linhas = ["AUC é piso (rótulo do vídeo, sonda da capa)"]
    if "falando" in pares:
        campo, _ = _normalizar_par("falando", pares["falando"])
        linhas.append(f"falando é medida contra {campo} (proxy)")
    comuns = sondas.index.intersection(rotulos.index)
    linhas.append(f"{len(rotulos)} rótulos, {len(sondas)} capas, {len(comuns)} em comum")
    return linhas


def ler_setores() -> pd.Series:
    """post_id -> setor, da amostra multimodal (a #12 lê só ids, canal, data e setor: nenhuma
    curtida, então o período de teste não é aberto)."""
    amostra = pd.read_csv(DADOS / "amostra_multimodal.csv", usecols=["post_id", "sector"])
    return amostra.drop_duplicates("post_id").set_index("post_id")["sector"]


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
    r = validar_sondas(sondas, rotulos, val["pares"], val["auc_min"], ler_setores())
    destino = DADOS / f"sondas_auc_v{sig['versao']}.csv"
    r.to_csv(destino, index=False)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    passam = r.loc[r["passa"] == True, "sonda"].tolist()  # noqa: E712 (NaN != True)
    print(f"\npassam de {val['auc_min']}: {', '.join(passam) or 'nenhuma'}")
    todas = [c.removeprefix("sonda_").removesuffix("_thumb") for c in sondas.columns]
    sem_rotulo = [n for n in todas if n not in val["pares"]]
    print(f"sem rótulo barato ({len(sem_rotulo)}): {', '.join(sem_rotulo)}")
    for linha in ressalvas(sondas, rotulos, val["pares"]):
        print(f"ressalva: {linha}")
    print(f"-> {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
