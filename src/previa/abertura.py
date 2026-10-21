"""Abertura dos resultados lacrados (decisão #36; aulas 2.4, 2.5, 2.6 e 4.8).

Lê as hipóteses H2 a H11 e a ablação já rodadas e dá o veredito de cada uma de uma vez, pela regra
que o `HIPOTESES.md` escreveu antes de rodar (2.5: o que estava escrito não muda):

- a régua da #27 (`veredito_27`): o tratamento fica se melhorar a manchete OU o erro em curtidas
  (`mae_log`), com o IC 95% pareado por canal sem o zero, SEM piorar a outra (o IC da outra não
  fica inteiro do lado ruim). Cada hipótese escolhe a variante da régua na `configs/abertura.yaml`;
- a leitura da ablação (`veredito_36`): as três leituras da 2.6 lado a lado (degrau da escada,
  todos menos um, controle de ruído). Só é "argumento" para a explicação o bloco sólido nas
  três: escada e remoção com IC sem o zero, as duas acima do efeito do ruído (2.6, erros comuns
  2 a 4; 4.8, erro comum 6).

A H1 (o teste final) não entra: abre à parte, com `carregar_base(final=True)`.

SÓ O CONTROLADOR RODA, COM O VINICIUS PRESENTE: este script revela os resultados lacrados.
Uso: uv run python -m previa.abertura
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from previa.baselines import ler_config as ler_config_baselines
from previa.caminhos import CONFIGS, DADOS
from previa.experimento import EXPERIMENTO, comparar

REGUAS = ("padrao", "mae_log", "nao_piorar", "ilustracao")
AVALIACOES = ("walk_forward", "canal_novo")
LEITURAS = {"cumulativa": "escada", "todos menos um": "remocao", "controle negativo": "ruido"}
METRICAS = [
    "ganho",
    "ic_ganho_lo",
    "ic_ganho_hi",
    "dif_mae_log",
    "ic_dif_mae_lo",
    "ic_dif_mae_hi",
]


def run_da_config(caminho: str) -> str:
    """O run FINISHED de desenvolvimento mais recente com a tag `config` igual ao caminho.

    Só `avaliacao = desenvolvimento (2025)`: o run do teste final e o do modelo de uso têm a mesma
    config e não podem ser o da abertura. Não lê métrica nenhuma.
    """
    mlflow.set_experiment(EXPERIMENTO)
    seguro = str(caminho).replace("'", "''")
    achados = mlflow.search_runs(
        experiment_names=[EXPERIMENTO],
        filter_string=(
            f"tags.config = '{seguro}' and attributes.status = 'FINISHED' "
            "and tags.avaliacao = 'desenvolvimento (2025)'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not achados:
        raise LookupError(
            f"nenhum run com a tag config = {caminho}: rode "
            f"`uv run python -m previa.experimento {caminho}` antes da abertura"
        )
    return achados[0].info.run_id


def _melhora_manchete(g) -> bool:
    return g["ic_ganho_lo"] > 0  # ganho > 0 é B ordenando melhor


def _melhora_mae(g) -> bool:
    return g["ic_dif_mae_hi"] < 0  # dif_mae_log < 0 é B errando menos o número


def _piora_manchete(g) -> bool:
    return g["ic_ganho_hi"] < 0  # o IC inteiro do lado ruim


def _piora_mae(g) -> bool:
    return g["ic_dif_mae_lo"] > 0


def veredito_27(g, regra: str) -> bool:
    """True se o tratamento fica, numa avaliação (`g`: a coluna do `comparar` dessa avaliação).

    padrao: melhora a manchete ou o mae_log, sem piorar a outra (#27). mae_log: decide o mae_log,
    sem piorar a manchete (H8, H9, H11). nao_piorar: nenhuma das duas com o IC inteiro do lado
    ruim (H6 no walk-forward). ilustracao: nunca fica (H3, lacuna 0).
    """
    if regra == "padrao":
        melhora = _melhora_manchete(g) or _melhora_mae(g)
        return bool(melhora and not _piora_manchete(g) and not _piora_mae(g))
    if regra == "mae_log":
        return bool(_melhora_mae(g) and not _piora_manchete(g))
    if regra == "nao_piorar":
        return bool(not _piora_manchete(g) and not _piora_mae(g))
    if regra == "ilustracao":
        return False
    raise ValueError(f"régua desconhecida: {regra!r}; use uma de {REGUAS}")


def _efeito(g, regua: str) -> float:
    """O efeito no lado bom: ganho na manchete; menos a diferença do mae_log."""
    return float(g["ganho"]) if regua == "manchete" else -float(g["dif_mae_log"])


def _sem_zero(g, regua: str) -> bool:
    return bool(_melhora_manchete(g) if regua == "manchete" else _melhora_mae(g))


def argumento_unico(g, rui) -> bool:
    """A regra de "argumento" da #36 para uma leitura só (`g`), contra o controle de ruído (`rui`).

    Numa mesma régua (manchete ou mae_log): IC pareado sem o zero no lado bom e efeito (ponto)
    acima do efeito do ruído em valor absoluto (2.6: o tamanho do acaso nesta avaliação).
    """
    return any(
        _sem_zero(g, r) and _efeito(g, r) > abs(_efeito(rui, r)) for r in ("manchete", "mae_log")
    )


def _argumento(esc, rem, rui) -> bool:
    """Sólido nas três leituras: a mesma régua vale para a escada e para a remoção."""
    return any(
        _sem_zero(esc, r)
        and _sem_zero(rem, r)
        and _efeito(esc, r) > abs(_efeito(rui, r))
        and _efeito(rem, r) > abs(_efeito(rui, r))
        for r in ("manchete", "mae_log")
    )


def _leituras(avaliacao, bloco, esc, rem, rui, argumento) -> list[dict]:
    return [
        {"avaliacao": avaliacao, "bloco": bloco, "leitura": leitura}
        | {m: linha[m] for m in METRICAS}
        | {k: linha.get(k) for k in ("run_a", "run_b")}
        | {"argumento": "sim" if argumento else "não"}
        for leitura, linha in (("escada", esc), ("remocao", rem), ("ruido", rui))
    ]


def veredito_36(tabela: pd.DataFrame) -> pd.DataFrame:
    """As três leituras de cada bloco, por avaliação, e se ele é argumento (#36).

    `argumento` = "sim" só se, numa mesma régua (manchete ou mae_log), o degrau da escada e a
    remoção têm IC pareado sem o zero no lado bom e os dois efeitos passam o do controle de ruído
    (em valor absoluto, no ponto estimado). O que fica no modelo é da #27 da hipótese, não desta
    tabela (por isso não há "sai" aqui). Uma linha por bloco, avaliação e leitura.
    """
    linhas = []
    for avaliacao, t in tabela.groupby("avaliacao", sort=False):
        rui = _ruido(t, avaliacao)
        for bloco in t[t["desenho"] == "cumulativa"]["bloco"]:
            esc = t[(t["desenho"] == "cumulativa") & (t["bloco"] == bloco)]
            rem = t[(t["desenho"] == "todos menos um") & (t["bloco"] == bloco)]
            if rem.empty:
                raise ValueError(f"bloco {bloco} sem a remoção (todos menos um) em {avaliacao}")
            esc, rem = esc.iloc[0], rem.iloc[0]
            linhas += _leituras(avaliacao, bloco, esc, rem, rui, _argumento(esc, rem, rui))
    return pd.DataFrame(linhas)


def _ruido(t, avaliacao):
    ruido = t[t["desenho"] == "controle negativo"]
    if ruido.empty:
        raise ValueError(
            f"a tabela da ablação não tem o controle negativo em {avaliacao}: sem ele não "
            "se sabe o tamanho do acaso (2.6, erro comum 2); rode previa.ablacao com ruido > 0"
        )
    return ruido.iloc[0]


def abrir(cfg_abertura: dict, comparar_fn=comparar) -> pd.DataFrame:
    """Uma tabela com o veredito de cada hipótese e de cada bloco da ablação.

    Linhas `tipo == "hipotese"`: uma por hipótese e avaliação, com o veredito da régua daquela
    avaliação e o `veredito_final` (todas as avaliações passam e a que ela `exige` também).
    Linhas `tipo == "bloco"`: as três leituras da ablação, com o veredito da #36.
    """
    cfg_b = ler_config_baselines()
    comps = cfg_abertura["comparacoes"]
    brutas = {}
    for nome, c in comps.items():
        if desconhecidas := set(c["regras"].values()) - set(REGUAS):
            raise ValueError(f"{nome}: régua desconhecida {sorted(desconhecidas)}")
        run_a, run_b = run_da_config(c["controle"]), run_da_config(c["tratamento"])
        r = comparar_fn(run_a, run_b, cfg_b["n_boot"], cfg_b["seed"], cfg_b["min_posts"])
        for avaliacao in c["regras"]:
            if avaliacao not in r.columns:
                raise ValueError(f"{nome}: o comparar não trouxe a avaliação {avaliacao}")
        brutas[nome] = (run_a, run_b, r)

    veredito_de = {}  # (nome, avaliação) -> bool

    def final(nome, vistos=()) -> bool:
        if nome in vistos:
            raise ValueError(f"`exige` em ciclo: {(*vistos, nome)}")
        c = comps[nome]
        ok = all(veredito_de[nome, a] for a in c["regras"])
        if exige := c.get("exige"):
            ok = ok and final(exige, (*vistos, nome))
        return ok

    linhas = []
    for nome, c in comps.items():
        run_a, run_b, r = brutas[nome]
        for avaliacao, regra in c["regras"].items():
            veredito_de[nome, avaliacao] = veredito_27(r[avaliacao], regra)
    for nome, c in comps.items():
        run_a, run_b, r = brutas[nome]
        for avaliacao, regra in c["regras"].items():
            linhas.append(
                {
                    "tipo": "hipotese",
                    "id": nome,
                    "avaliacao": avaliacao,
                    "regra": regra,
                    "controle": c["controle"],
                    "tratamento": c["tratamento"],
                    "run_a": run_a,
                    "run_b": run_b,
                    "veredito": "fica" if veredito_de[nome, avaliacao] else "não fica",
                    "veredito_final": "fica" if final(nome) else "não fica",
                }
                | {m: r.loc[m, avaliacao] for m in METRICAS}
            )
    destino = DADOS / cfg_abertura["ablacao"]
    if not destino.exists():
        raise FileNotFoundError(f"{destino} não existe: rode `uv run python -m previa.ablacao`")
    ablacao = pd.read_csv(destino)
    blocos = veredito_36(ablacao)
    mapa = cfg_abertura["blocos"]
    if faltam := set(blocos["bloco"]) - set(mapa):
        raise ValueError(f"blocos da ablação sem hipótese no `blocos` da config: {sorted(faltam)}")
    # bloco que está no mapa mas não na ablação (H10, o último): a escada e a remoção são a mesma
    # comparação (controle -> tratamento da hipótese); o ruído é o da ablação
    extra = []
    for bloco in set(mapa) - set(blocos["bloco"]):
        nome = mapa[bloco]
        if not nome:
            raise ValueError(f"bloco {bloco} fora da ablação e sem hipótese: nada a abrir")
        run_a, run_b, r = brutas[nome]
        for avaliacao in comps[nome]["regras"]:
            g = r[avaliacao].to_dict() | {"run_a": run_a, "run_b": run_b}
            rui = _ruido(ablacao[ablacao["avaliacao"] == avaliacao], avaliacao)
            extra += _leituras(avaliacao, bloco, g, g, rui, argumento_unico(g, rui))
    blocos = pd.concat([blocos, pd.DataFrame(extra)], ignore_index=True)
    blocos["modelo"] = blocos["bloco"].map(
        lambda b: (
            f"{mapa[b]}: {'fica' if final(mapa[b]) else 'não fica'}"
            if mapa[b]
            else "base da Opção A (sem hipótese)"
        )
    )
    blocos = blocos.assign(tipo="bloco", id=blocos["bloco"])
    return pd.concat([pd.DataFrame(linhas), blocos], ignore_index=True)


def _tabela_md(df: pd.DataFrame) -> str:
    cabeca = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    corpo = "".join(
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in linha) + " |\n"
        for linha in df.itertuples(index=False)
    )
    return cabeca + corpo


NOTAS = """## Como ler
- `modelo` vem da #27 da hipótese do bloco (veredito_final); `argumento` = "sim" só se o bloco é
  sólido nas três leituras (escada, remoção, acima do ruído), #36.
- A barra do ruído usa o ponto estimado do controle negativo, em valor absoluto (2.6).
- H10 (detectores) é o último bloco: a escada e a remoção são a mesma comparação (h7_thumb ->
  h10_detectores); o ruído é o da ablação.
- Os ids dos runs (run_a, run_b) estão em cada linha: confira à vista, por exemplo, se o H7 é o
  run f405ee37.
- **Escolhas de leitura que o Vinicius confirma na abertura:** (a) H11 exigida nos dois cenários
  (walk-forward e canal novo); (b) H8 e H9 decididas pelo mae_log, com a manchete como guarda
  (não piorar); (c) a barra do ruído no ponto estimado.
"""


def em_markdown(r: pd.DataFrame) -> str:
    """O relatório da abertura: hipóteses (régua da #27) e blocos (as três leituras, #36)."""
    h = r[r["tipo"] == "hipotese"].drop(
        columns=["tipo", "controle", "tratamento", "bloco", "leitura"], errors="ignore"
    )
    b = r[r["tipo"] == "bloco"][
        ["bloco", "avaliacao", "leitura", *METRICAS, "run_a", "run_b", "modelo", "argumento"]
    ]
    return (
        "# Abertura dos resultados lacrados (#27 e #36)\n\n"
        "## Hipóteses (régua da #27, `HIPOTESES.md`)\n\n"
        f"{_tabela_md(h.dropna(axis=1, how='all'))}\n"
        "## Blocos da ablação (escada, remoção e ruído lado a lado, #36)\n\n"
        f"{_tabela_md(b)}\n"
        f"{NOTAS}"
    )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    caminho = Path(argv[0]) if argv else CONFIGS / "abertura.yaml"
    cfg = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    r = abrir(cfg)
    texto = em_markdown(r)
    destino = DADOS / "logs" / "abertura.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\n-> {destino}\ncomando: uv run python -m previa.abertura")
    return 0


if __name__ == "__main__":
    sys.exit(main())
