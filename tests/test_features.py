"""Testes das features do canal e do tempo (M3; aulas 3.5 e 2.2; princípios 2 e 4 do CLAUDE.md).

O histórico só usa posts maduros (publicados ao menos 30 dias antes) e com curtidas. Os valores
esperados são literais conferidos à mão; os dois testes de vazamento são os da 3.5.
"""

import numpy as np
import pandas as pd
import pytest

from previa.baselines import mediana_madura
from previa.features import (
    cadencia,
    calendario,
    historico_do_canal,
    montar_features,
    quantis_do_canal,
)

DIA_0 = pd.Timestamp("2025-01-01", tz="UTC")


def _canal(dias, ys, canal="UCa"):
    return pd.DataFrame(
        {
            "channel_id": canal,
            "published_at": DIA_0 + pd.to_timedelta(dias, "D"),
            "y": np.asarray(ys, float),
        }
    )


def _aleatoria(n=300, seed=0):
    """Dois canais, posts em datas sorteadas ao longo de 2 anos, alguns sem curtidas."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "channel_id": rng.choice(["UCa", "UCb"], n),
            "published_at": DIA_0 + pd.to_timedelta(rng.uniform(0, 730, n), "D"),
            "y": rng.normal(7, 1.5, n),
        }
    )
    d.loc[rng.random(n) < 0.05, "y"] = np.nan
    return d


def test_historico_caso_a_mao():
    # dias 0, 10, 40, 50 (sem curtidas) e 80; maduro = publicado até 30 dias antes, com y
    d = _canal([0, 10, 40, 50, 80], [1.0, 3.0, 2.0, np.nan, 5.0])
    h = historico_do_canal(d, lacuna_dias=30)
    assert h["hist_n"].tolist() == [0, 0, 2, 2, 3]
    assert h["hist_mediana"].tolist()[2:] == [2.0, 2.0, 2.0]  # [1, 3] e [1, 3, 2]
    assert h["hist_media"].iloc[4] == pytest.approx(2.0)
    assert h["hist_desvio"].iloc[2] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert h["hist_mediana"].iloc[:2].isna().all()  # sem histórico maduro ainda
    # o último post usado pelo de dia 80 é o de dia 40 (o de dia 50 não tem curtidas)
    assert h["hist_ultimo_ts"].iloc[4] == DIA_0 + pd.Timedelta(days=40)


def test_historico_30_dias_entra_29_nao():
    d = _canal([0, 29, 30], [4.0, 1.0, 1.0])
    h = historico_do_canal(d, lacuna_dias=30)
    assert h["hist_n"].tolist() == [0, 0, 1]
    assert h["hist_mediana"].iloc[2] == 4.0


def test_historico_lacuna_zero_nao_usa_o_proprio_post_nem_simultaneos():
    d = _canal([0, 5, 5, 9], [1.0, 2.0, 3.0, 4.0])
    h = historico_do_canal(d, lacuna_dias=0)
    assert h["hist_n"].tolist() == [0, 1, 1, 3]
    assert h["hist_mediana"].iloc[3] == 2.0


def test_historico_janelas_recentes_e_tendencia():
    d = _canal([0, 1, 2, 3, 4, 5, 6, 100], [1, 2, 3, 4, 5, 6, 7, 0])
    h = historico_do_canal(d, lacuna_dias=30).iloc[-1]
    assert h["hist_med5"] == 5.0  # mediana de 3..7
    assert h["hist_med20"] == 4.0  # mediana de 1..7
    assert h["tendencia"] == 1.0


def test_historico_um_canal_nao_ve_o_outro():
    d = pd.concat([_canal([0], [9.0], "UCa"), _canal([40], [1.0], "UCb")], ignore_index=True)
    assert historico_do_canal(d, lacuna_dias=30)["hist_n"].tolist() == [0, 0]


def test_ultimo_post_usado_e_anterior():
    d = _aleatoria()
    h = historico_do_canal(d, lacuna_dias=30)
    usado = h["hist_ultimo_ts"].notna()
    limite = d.loc[usado, "published_at"] - pd.Timedelta(days=30)
    assert (h.loc[usado, "hist_ultimo_ts"] <= limite).all()


CORTE = DIA_0 + pd.Timedelta(days=400)


def _perturbacao(gerar, d, data=CORTE):
    """Teste da 3.5: embaralha y dos posts a partir de `data`; as features de antes não mudam."""
    antes = gerar(d)
    mexido = d.copy()
    futuro = mexido["published_at"] >= data
    mexido.loc[futuro, "y"] = np.random.default_rng(0).permutation(mexido.loc[futuro, "y"])
    depois = gerar(mexido)
    ate = d["published_at"] <= data
    num = antes.select_dtypes("number").columns
    return np.allclose(antes.loc[ate, num], depois.loc[ate, num], equal_nan=True)


def test_perturbar_o_futuro_nao_muda_o_passado():
    d = _aleatoria()
    assert _perturbacao(lambda x: historico_do_canal(x, lacuna_dias=30), d)
    assert _perturbacao(lambda x: montar_features(x, lacuna_dias=30), d)


def test_a_versao_ingenua_e_pega():
    def ingenua(x):
        return x.groupby("channel_id")["y"].transform("median").to_frame("mediana_do_canal")

    assert not _perturbacao(ingenua, _aleatoria())


def test_ordem_das_linhas_nao_importa():
    d = _aleatoria()
    embaralhada = d.sample(frac=1, random_state=1)
    a = montar_features(d, lacuna_dias=30)
    b = montar_features(embaralhada, lacuna_dias=30).loc[d.index]
    pd.testing.assert_frame_equal(a, b)


def test_paridade_com_a_mediana_madura_do_baseline():
    d = _aleatoria()
    h = historico_do_canal(d, lacuna_dias=30)
    m = mediana_madura(d, dias=30, min_posts=3)
    ok = h["hist_n"] >= 3
    assert ok.sum() > 100
    np.testing.assert_allclose(h.loc[ok, "hist_mediana"], m[ok])


def test_cadencia_usa_todos_os_posts_anteriores():
    # cadência não lê curtidas: entra o post sem y e não há lacuna; simultâneos não se veem
    d = _canal([0, 5, 5, 20], [1.0, np.nan, 2.0, 3.0])
    c = cadencia(d)
    assert c["dias_desde_ultimo"].isna().iloc[0]
    assert c["dias_desde_ultimo"].tolist()[1:] == [5.0, 5.0, 15.0]
    assert c["posts_30d"].tolist() == [0, 1, 1, 3]


def test_calendario_hora_local_feriado_e_idade():
    d = pd.DataFrame(
        {
            "channel_id": ["UCa", "UCa", "UCa"],
            "published_at": pd.to_datetime(
                ["2025-03-02T01:00:00Z", "2025-03-04T15:00:00Z", "2025-04-21T12:00:00Z"]
            ),
            "y": [1.0, 2.0, 3.0],
        }
    )
    c = calendario(d)
    assert c["hora_local"].tolist() == [22, 12, 9]  # UTC-3
    assert c["dia_semana_local"].iloc[0] == 5  # 01/03/2025, sábado, em Brasília
    assert c["feriado"].tolist() == [0, 1, 1]  # terça de Carnaval e Tiradentes
    assert c["idade_canal_dias"].tolist() == [0, 2, 50]


# --- quantis do canal (decisão #7, aula 1.6): a régua das faixas, não uma feature do modelo ---

QUANTIS = ["faixa_p25", "faixa_p75", "faixa_p95"]


def test_quantis_do_canal_caso_a_mao():
    # 10 posts maduros (dias 0 a 9, y = 1..10); um post sem curtidas no dia 20 não conta
    dias = [*range(10), 20, 38, 39]
    ys = [*range(1, 11), np.nan, 0.0, 0.0]
    q = quantis_do_canal(_canal(dias, ys), lacuna_dias=30, min_posts=10)
    assert list(q.columns) == QUANTIS
    assert q.iloc[:11].isna().all().all()  # os 10 primeiros e o post do dia 20: sem 10 maduros
    # dia 38: o post do dia 9 tem 29 dias e não entra; sobram 9 maduros, menos que min_posts
    assert q.iloc[11].isna().all()
    # dia 39: o post do dia 9 tem 30 dias e entra; y = 1..10, interpolação linear do pandas
    assert q.iloc[12].tolist() == pytest.approx([3.25, 7.75, 9.55])


def test_quantis_posts_recentes_entram_so_depois_dos_30_dias():
    # os mesmos 10 maduros e, nos dias 38 e 39, dois posts com y = 0; o do dia 100 já os vê:
    # 12 posts (0, 0, 1..10) e o p25 cai de 3,25 para 1,75 (a prova de que entraram)
    dias = [*range(10), 38, 39, 100]
    ys = [*range(1, 11), 0.0, 0.0, 5.0]
    q = quantis_do_canal(_canal(dias, ys))
    esperado = np.quantile([*range(1, 11), 0.0, 0.0], [0.25, 0.75, 0.95])
    assert q.iloc[12].tolist() == pytest.approx(esperado.tolist())
    assert q.iloc[12]["faixa_p25"] == pytest.approx(1.75)


def test_quantis_min_posts_e_configuravel():
    d = _canal([0, 1, 2, 40], [1.0, 2.0, 3.0, 9.0])
    assert quantis_do_canal(d, min_posts=4).iloc[3].isna().all()
    assert quantis_do_canal(d, min_posts=3).iloc[3].tolist() == pytest.approx([1.5, 2.5, 2.9])


def test_quantis_um_canal_nao_ve_o_outro():
    d = pd.concat(
        [_canal(list(range(10)), list(range(1, 11)), "UCa"), _canal([39], [0.0], "UCb")],
        ignore_index=True,
    )
    assert quantis_do_canal(d).iloc[10].isna().all()  # UCb não tem maduros, mesmo com os de UCa


def test_quantis_batem_com_np_quantile_dos_maduros():
    d = _aleatoria()
    q = quantis_do_canal(d, lacuna_dias=30, min_posts=10)
    com = q.dropna()
    assert len(com) > 100
    for i in com.index[::25]:
        post = d.loc[i]
        maduros = d[
            (d["channel_id"] == post["channel_id"])
            & d["y"].notna()
            & (d["published_at"] <= post["published_at"] - pd.Timedelta(days=30))
        ]["y"]
        assert len(maduros) >= 10
        esperado = np.quantile(maduros, [0.25, 0.75, 0.95])
        assert com.loc[i, QUANTIS].tolist() == pytest.approx(esperado.tolist())


def test_quantis_perturbar_o_futuro_nao_muda():
    d = _aleatoria()
    assert _perturbacao(lambda x: quantis_do_canal(x, lacuna_dias=30), d)
    # e o teste não é vazio: há quantis calculados antes do corte
    antes = quantis_do_canal(d, lacuna_dias=30)
    assert antes.loc[d["published_at"] <= CORTE].notna().all(axis=1).sum() > 50


def test_quantis_ordem_das_linhas_nao_importa():
    d = _aleatoria()
    a = quantis_do_canal(d)
    b = quantis_do_canal(d.sample(frac=1, random_state=1)).loc[d.index]
    pd.testing.assert_frame_equal(a, b)


def test_quantis_nao_sao_feature_do_modelo():
    from previa.features import FEATURES

    assert not set(QUANTIS) & set(FEATURES)
    assert not set(QUANTIS) & set(montar_features(_aleatoria()).columns)
