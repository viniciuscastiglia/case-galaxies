"""Testes da fonte do histórico do canal na previsão (Tool 2; aulas 1.5 e 6.3), sem chave: o
FakeYT imita a API no formato real."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from previa.coletor import Cliente
from previa.historico import historico_api, historico_congelado, posts_de_itens

from fakes import CANAL_ID, FakeYT, pagina, video

AGORA = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)


def _yt():
    itens = {
        "v1": video("v1", "2026-08-01T22:00:05Z", "PT22S"),
        "v2": video("v2", "2026-09-01T22:00:05Z", "PT40S", likes=None),  # curtidas ocultas
        "v3": video("v3", "2026-09-05T22:00:05Z", "PT15M"),  # vídeo longo: não é Short
    }
    return FakeYT({"UU" + CANAL_ID[2:]: {"p0": pagina(list(itens))}}, itens)


def test_posts_de_itens_so_shorts_com_y_em_log():
    p = posts_de_itens(list(_yt().itens.values()), AGORA)
    assert p["post_id"].tolist() == ["v1", "v2"]
    assert p.loc[0, "y"] == np.log1p(2410) and np.isnan(p.loc[1, "y"])
    assert str(p["published_at"].dt.tz) == "UTC" and p["caption"].iloc[0].startswith("1 produto")


def test_historico_api_usa_o_cache_do_dia(tmp_path):
    yt = _yt()
    cli = Cliente(yt, arquivo=None)
    a = historico_api(cli, CANAL_ID, agora=AGORA, pasta=tmp_path)
    chamadas = len(yt.chamadas)
    b = historico_api(cli, CANAL_ID, agora=AGORA, pasta=tmp_path)
    assert len(yt.chamadas) == chamadas  # o segundo pedido do dia não chama a API
    pd.testing.assert_frame_equal(a, b)


def test_historico_congelado_do_golden(tmp_path):
    from previa.coletor import parse_video

    linhas = [parse_video(it, AGORA) for it in _yt().itens.values()]
    pd.DataFrame(linhas).to_parquet(tmp_path / "historico.parquet")
    h = historico_congelado(CANAL_ID, tmp_path / "historico.parquet")
    assert h["post_id"].tolist() == ["v1", "v2"]
