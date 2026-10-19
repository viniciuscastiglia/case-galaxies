"""Testes do golden set congelado (decisão #4; aulas 6.6 e 2.3): canais de fora da base, sorteados
por estrato com seed fixa, 5 Shorts maduros do período de teste por canal. Sem API: a busca dos
Shorts de um canal é falsa."""

import pandas as pd

from previa.golden import candidatos, escolher_shorts, sortear

INICIO, FIM = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-09-15", tz="UTC")
COLETA = pd.Timestamp("2026-10-19", tz="UTC")


def _shorts(canal, datas, likes=100, short=True):
    pub = pd.to_datetime(datas, utc=True)
    return pd.DataFrame(
        {
            "video_id": [f"{canal}{k:03d}" for k in range(len(datas))],
            "channel_id": canal,
            "published_at": pub,
            "likes": likes,
            "is_short": short,
            "age_days": (COLETA - pub).total_seconds() / 86_400,
        }
    )


def test_candidatos_sao_as_reservas_revisadas_de_fora_da_base():
    canais = pd.DataFrame(
        {
            "channel_id": ["UCa", "UCb", "UCc", "UCnjK9YWSbd0AAjHhM3vAu7A"],
            "motivo": [
                "reserva: passou na revisão de 08/10",
                "reserva sem revisão: sem vaga",
                "reserva: passou na revisão",
                "reserva: marca de cabelo",
            ],
            "setor": "beleza",
            "tipo": "marca",
        }
    )
    c = candidatos(canais, canais_da_base={"UCc"})
    assert c["channel_id"].tolist() == ["UCa"]  # sem revisão, na base e a Salon Line saem


def test_escolhe_os_5_mais_recentes_maduros_do_periodo():
    datas = ["2026-03-20", "2026-04-02", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"]
    datas += ["2026-09-10", "2026-09-30"]  # 30/09: fora do período (e imaturo)
    s = _shorts("UCa", datas)
    s.loc[3, "likes"] = None  # curtidas ocultas: fora
    sel = escolher_shorts(s, INICIO, FIM)
    assert sel["published_at"].dt.strftime("%m-%d").tolist() == [
        "04-02",
        "05-01",
        "07-01",
        "08-01",
        "09-10",
    ]


def test_sorteio_por_estrato_pula_canal_sem_5_shorts():
    cands = pd.DataFrame(
        {
            "channel_id": ["UCa", "UCb", "UCc", "UCd"],
            "setor": ["beleza", "beleza", "beleza", "automotivo"],
            "tipo": ["marca", "marca", "criador", "criador"],
        }
    )
    bons = ["2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01"]
    shorts = {
        "UCa": _shorts("UCa", bons),
        "UCb": _shorts("UCb", bons[:3]),  # só 3: não é elegível
        "UCc": _shorts("UCc", bons),
        "UCd": _shorts("UCd", bons),
    }
    estratos = {("beleza", "marca"): 1, ("beleza", "criador"): 1, ("automotivo", "criador"): 1}
    escolhidos, log = sortear(cands, estratos, shorts.__getitem__, INICIO, FIM, seed=0)
    assert set(escolhidos) == {"UCa", "UCc", "UCd"}
    assert all(len(v) == 5 for v in escolhidos.values())
    assert len(log) >= 3  # um registro por canal tentado
