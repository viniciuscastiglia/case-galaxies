"""Testes-contrato dos splits por data (aula 2.3; decisões #2 e #3 de 15/10).

Para a M1, estes testes são o enunciado: `split_treino_cal_teste` e `canal_novo_no_futuro`
estão certas quando tudo aqui passa. As datas são literais conferidos à mão, nunca calculados
pelo código testado.
"""

import pandas as pd
import pytest

from previa.dados import carregar_base
from previa.splits import (
    canal_novo_no_futuro,
    folds_de_desenvolvimento,
    ler_avaliacao,
    split_por_canal,
    split_temporal,
    split_treino_cal_teste,
    walk_forward,
)

AVALIACAO = """\
teste_inicio: "2026-04-01"
treino_inicio: "2021-07-01"
gap_dias: 30
cal_inicio: "2026-02-01"
teste_fim: "2026-09-15"
walk_forward:
  n_folds: 4
  inicio: "2025-01-01"
canal_novo:
  n_grupos: 4
  seed: 0
"""

# um post em cada fronteira das decisões de 15/10, com o bloco esperado (None = nenhum)
FRONTEIRAS = [
    ("pre_shorts", "UCa", "2021-06-30T23:59:00Z", None),  # antes do Shorts no Brasil (#2)
    ("ini_treino", "UCa", "2021-07-01T00:00:00Z", "treino"),
    ("meio_trein", "UCb", "2024-05-10T12:00:00Z", "treino"),
    ("fim_treino", "UCa", "2026-01-01T23:59:00Z", "treino"),  # cal_inicio - 30 dias = 02/01
    ("gap_inicio", "UCb", "2026-01-02T00:00:00Z", None),
    ("gap_fim000", "UCa", "2026-01-31T23:59:00Z", None),
    ("cal_inicio", "UCb", "2026-02-01T00:00:00Z", "cal"),
    ("cal_fim000", "UCa", "2026-03-31T23:59:00Z", "cal"),
    ("tst_inicio", "UCb", "2026-04-01T00:00:00Z", "teste"),  # a fronteira já é teste
    ("tst_fim000", "UCa", "2026-09-14T23:59:00Z", "teste"),  # último Short maduro
    ("imaturo000", "UCb", "2026-09-15T00:00:00Z", None),
]


@pytest.fixture
def cfg(tmp_path):
    caminho = tmp_path / "avaliacao.yaml"
    caminho.write_text(AVALIACAO)
    return ler_avaliacao(caminho)


@pytest.fixture
def fronteiras():
    return pd.DataFrame(
        {
            "post_id": [p for p, _, _, _ in FRONTEIRAS],
            "channel_id": [c for _, c, _, _ in FRONTEIRAS],
            "published_at": pd.to_datetime([q for _, _, q, _ in FRONTEIRAS]),
        }
    )


def base_densa(n_canais=8, passo_dias=5):
    """Um post por canal a cada `passo_dias`, de 2021 a out/2026, em ordem de CANAL (o formato
    que engana o TimeSeriesSplit, aula 2.3, passo 5)."""
    datas = pd.date_range("2021-01-01", "2026-10-01", freq=f"{passo_dias}D", tz="UTC")
    partes = [
        pd.DataFrame({"channel_id": f"UC{c:02d}", "published_at": datas}) for c in range(n_canais)
    ]
    return pd.concat(partes, ignore_index=True)


def ids(df, rotulos):
    return set(df.loc[rotulos, "post_id"])


# --- configuração ---


def test_ler_avaliacao_devolve_datas_em_utc(cfg):
    assert cfg["cal_inicio"] == pd.Timestamp("2026-02-01", tz="UTC")
    assert cfg["walk_forward"]["inicio"] == pd.Timestamp("2025-01-01", tz="UTC")
    assert cfg["gap_dias"] == 30
    assert cfg["canal_novo"] == {"n_grupos": 4, "seed": 0}


# --- portadas da aula 2.3 (Claude) ---


def test_split_temporal_deixa_o_gap_fora_dos_dois_lados(fronteiras):
    treino, teste = split_temporal(fronteiras, pd.Timestamp("2026-02-01", tz="UTC"), gap_dias=30)
    assert ids(fronteiras, treino) == {"pre_shorts", "ini_treino", "meio_trein", "fim_treino"}
    assert ids(fronteiras, teste) == {
        "cal_inicio",
        "cal_fim000",
        "tst_inicio",
        "tst_fim000",
        "imaturo000",
    }  # gap_inicio e gap_fim000 não estão em lado nenhum


def test_walk_forward_avanca_no_tempo_com_gap():
    df = base_densa()
    inicio = pd.Timestamp("2025-01-01", tz="UTC")
    folds = list(walk_forward(df, n_folds=4, gap_dias=30, inicio_teste=inicio))
    assert len(folds) == 4
    anterior_fim = inicio
    for treino, teste in folds:
        ini, fim = df.loc[teste, "published_at"].min(), df.loc[teste, "published_at"].max()
        assert ini >= anterior_fim  # janelas de teste em ordem e sem sobreposição
        assert df.loc[treino, "published_at"].max() < ini - pd.Timedelta(days=30)
        anterior_fim = fim
    tamanhos = [len(tr) for tr, _ in folds]
    assert tamanhos == sorted(tamanhos)  # janela expansiva: o treino só cresce


def test_walk_forward_exige_inicio_teste_por_nome():
    """Na ordem da aula, walk_forward(df, 4, 30) ligava inicio_teste=4 (1970) sem erro."""
    with pytest.raises(TypeError):
        list(walk_forward(base_densa(), 4, 30))


def test_folds_de_desenvolvimento_ficam_dentro_do_treino(cfg):
    """Com a base inteira, o 4º fold testava no gap e na calibração (erro comum 4)."""
    df = base_densa()
    treino, _, _ = split_treino_cal_teste(df, cfg)
    folds = folds_de_desenvolvimento(df, cfg)
    assert len(folds) == 4
    for treino_k, teste_k in folds:
        assert len(teste_k) > 0
        assert set(treino_k) | set(teste_k) <= set(treino)
        assert df.loc[teste_k, "published_at"].min() >= cfg["walk_forward"]["inicio"]


def test_fold_vazio_e_erro_de_configuracao(cfg):
    cfg["walk_forward"]["inicio"] = pd.Timestamp("2027-01-01", tz="UTC")  # depois dos dados
    with pytest.raises(ValueError, match="sem nenhum post"):
        folds_de_desenvolvimento(base_densa(), cfg)


def test_split_por_canal_nao_deixa_canal_dos_dois_lados():
    df = base_densa()
    treino, teste = split_por_canal(df, frac=0.25, seed=0)
    canais_treino = set(df.loc[treino, "channel_id"])
    canais_teste = set(df.loc[teste, "channel_id"])
    assert canais_treino.isdisjoint(canais_teste)
    assert len(canais_teste) == 2  # 25% de 8 canais


# --- M1: split_treino_cal_teste (Vinicius) ---


def test_blocos_seguem_as_datas_decididas(fronteiras, cfg):
    treino, cal, teste = split_treino_cal_teste(fronteiras, cfg)
    assert ids(fronteiras, treino) == {"ini_treino", "meio_trein", "fim_treino"}
    assert ids(fronteiras, cal) == {"cal_inicio", "cal_fim000"}
    assert ids(fronteiras, teste) == {"tst_inicio", "tst_fim000"}


def test_blocos_nao_se_sobrepoem_e_respeitam_o_gap(cfg):
    df = base_densa()
    treino, cal, teste = split_treino_cal_teste(df, cfg)
    datas = df["published_at"]
    assert set(treino).isdisjoint(cal) and set(cal).isdisjoint(teste)
    assert datas[treino].max() < datas[cal].min() - pd.Timedelta(days=30)  # erro comum 3
    assert datas[cal].max() < datas[teste].min()  # treino -> gap -> calibração -> teste


def test_corta_por_data_e_nao_pela_posicao_da_linha(fronteiras, cfg):
    embaralhada = fronteiras.sample(frac=1, random_state=0)  # erro comum 2
    for original, mexida in zip(
        split_treino_cal_teste(fronteiras, cfg),
        split_treino_cal_teste(embaralhada, cfg),
        strict=True,
    ):
        assert set(original) == set(mexida)


def test_desenvolvimento_tem_teste_vazio(tmp_path, fronteiras):
    """A trava (aula 2.3 e 2.5): com carregar_base() o teste não existe, e o split não quebra."""
    (tmp_path / "avaliacao.yaml").write_text(AVALIACAO)
    fronteiras.to_parquet(tmp_path / "base.parquet", index=False)
    dev = carregar_base(
        caminho_base=tmp_path / "base.parquet",
        caminho_config=tmp_path / "avaliacao.yaml",
        registro=tmp_path / "teste_aberto.log",
    )
    treino, cal, teste = split_treino_cal_teste(dev, ler_avaliacao(tmp_path / "avaliacao.yaml"))
    assert len(teste) == 0
    assert ids(dev, cal) == {"cal_inicio", "cal_fim000"}


# --- M1: canal_novo_no_futuro (Vinicius) ---


@pytest.fixture
def grupos(cfg):
    df = base_densa()
    treino, _, teste = split_treino_cal_teste(df, cfg)
    return df, treino, teste, canal_novo_no_futuro(df, treino, teste, n_grupos=4, seed=0)


def test_canal_novo_nunca_aparece_no_treino(grupos):
    df, _, _, folds = grupos
    assert len(folds) == 4
    for treino_g, teste_g in folds:
        assert len(teste_g) > 0
        assert set(df.loc[treino_g, "channel_id"]).isdisjoint(df.loc[teste_g, "channel_id"])


def test_canal_novo_respeita_o_tempo(grupos):
    _, treino, teste, folds = grupos
    for treino_g, teste_g in folds:
        assert set(treino_g) <= set(treino)  # só datas do bloco de treino
        assert set(teste_g) <= set(teste)  # só datas do bloco de teste


def test_cada_canal_e_novo_exatamente_uma_vez(grupos):
    df, _, teste, folds = grupos
    vistos = [set(df.loc[teste_g, "channel_id"]) for _, teste_g in folds]
    assert sum(len(v) for v in vistos) == len(set().union(*vistos))  # grupos disjuntos
    assert set().union(*vistos) == set(df.loc[teste, "channel_id"])  # todos passam pelo teste


def test_canal_novo_reproduz_com_a_mesma_seed(grupos):
    df, treino, teste, folds = grupos
    de_novo = canal_novo_no_futuro(df, treino, teste, n_grupos=4, seed=0)
    assert [set(t) for _, t in folds] == [set(t) for _, t in de_novo]
