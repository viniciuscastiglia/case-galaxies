"""Testes do modelo de uso (M4 final; aulas 6.1 e 6.3; princípio 4 do CLAUDE.md).

O teste central é a paridade: as features que `features_de_um_short` monta para um Short "novo",
só com os posts anteriores do canal, são as mesmas que o treino calculou para ele.
"""

import hashlib
import json

import mlflow
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from test_experimento import mlflow_temporario  # noqa: F401

from previa.caminhos import CONFIGS
from previa.esquema import validar_tabela_de_treino
from previa.experimento import ler_config, preparar
from previa.intervalo import escores_cqr
from previa.intervalo import q_chapeu as calcular_q_chapeu
from previa.modelo import (
    Bundle,
    carregar,
    features_de_um_short,
    main,
    model_version,
    prever_intervalo,
    prever_log,
    referencia,
    registrar_bundle,
    salvar,
    treinar_bundle,
)
from previa.splits import ler_avaliacao

AVALIACAO = """\
teste_inicio: "2026-04-01"
treino_inicio: "2021-07-01"
gap_dias: 30
cal_inicio: "2025-10-01"
teste_fim: "2026-09-15"
walk_forward:
  n_folds: 4
  inicio: "2025-01-01"
canal_novo:
  n_grupos: 2
  seed: 0
"""

CONFIG = """\
nome: teste
hipotese: H1
descricao: GBM pequeno
features: [hist_n, hist_mediana, tendencia, dias_desde_ultimo, posts_30d, hora_local,
           feriado, idade_canal_dias, duracao_s, setor, txt_caracteres, txt_cta]
max_arvores: 60
paciencia: 10
lgbm: {objective: regression, learning_rate: 0.1, num_leaves: 7, min_data_in_leaf: 20}
pca: {titulo: 2}
"""


@pytest.fixture
def cfg_av(tmp_path):
    (tmp_path / "avaliacao.yaml").write_text(AVALIACAO)
    return ler_avaliacao(tmp_path / "avaliacao.yaml")


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "exp.yaml").write_text(CONFIG)
    return ler_config(tmp_path / "exp.yaml")


@pytest.fixture
def base():
    """4 canais, um post a cada 5 dias de 2021-05 a 2026-01, com título, duração e embedding."""
    rng = np.random.default_rng(0)
    linhas = []
    for c in range(4):
        datas = pd.date_range("2021-05-01", "2026-01-31", freq="5D", tz="UTC")
        datas = datas + pd.to_timedelta(rng.integers(0, 24, len(datas)), "h")
        for k, t in enumerate(datas):
            pid = f"c{c}p{k:05d}".ljust(11, "x")
            dur = int(rng.integers(10, 60))
            curt = int(np.expm1(5 + 0.3 * c + 0.02 * dur + rng.normal(0, 0.5)))
            titulo = "Comenta aqui! " * int(rng.integers(0, 2)) + f"Short {k} do canal {c}"
            linhas.append((pid, f"UC{c}", t, curt, "beleza" if c < 2 else "automotivo", pid, dur))
            linhas[-1] += (titulo + "\ndescrição padrão", 10_000 * (c + 1) ** 2)
    cols = ["post_id", "channel_id", "published_at", "metric_target", "sector"]
    cols += ["duplicate_group", "duration_s", "caption", "subscribers_at_collection"]
    return pd.DataFrame(linhas, columns=cols)


def _partes(base):
    dur = base.set_index("post_id")["duration_s"]
    emb = pd.DataFrame(np.random.default_rng(1).normal(size=(len(base), 5)), index=base["post_id"])
    return base.drop(columns="duration_s"), dur, {"titulo": emb}


def test_paridade_treino_uso(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    d = preparar(b, dur, cfg_av, cfg.lacuna_dias)
    alvo = d.index[d["linha"] & (d["channel_id"] == "UC1")][-40]  # um post do fim de 2025
    linha = d.loc[alvo]
    anteriores = d[(d["channel_id"] == "UC1") & (d["published_at"] < linha["published_at"])]
    novo = {
        "published_at": linha["published_at"],
        "caption": linha["caption"],
        "sector": linha["sector"],
        "duration_s": float(dur[linha["post_id"]]),
        "embeddings": {"titulo": emb["titulo"].loc[linha["post_id"]].to_numpy()},
    }
    x = features_de_um_short(bundle, anteriores[["channel_id", "published_at", "y"]], novo)
    sem_pca = [f for f in cfg.features if f != "setor"]
    np.testing.assert_allclose(
        x[sem_pca].to_numpy(float)[0], d.loc[alvo, sem_pca].to_numpy(float), equal_nan=True
    )
    assert x["setor"].iloc[0] == linha["setor"]
    esperado = bundle.pcas["titulo"].transform(novo["embeddings"]["titulo"][None, :])[0]
    np.testing.assert_allclose(x[["titulo_pc00", "titulo_pc01"]].to_numpy()[0], esperado)


def test_salva_carrega_e_preve_igual(base, cfg, cfg_av, tmp_path):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    salvar(bundle, tmp_path / "artefatos")
    carregado = carregar(tmp_path / "artefatos")
    novo = {
        "published_at": pd.Timestamp("2026-01-20T22:00Z"),
        "caption": "Comenta aqui! Título novo",
        "sector": "beleza",
        "duration_s": 30.0,
        "embeddings": {"titulo": np.zeros(5)},
    }
    posts = preparar(b, dur, cfg_av)[["channel_id", "published_at", "y"]]
    posts = posts[posts["channel_id"] == "UC0"]
    x = features_de_um_short(carregado, posts, novo)
    ref, n_ref = referencia(posts, novo["published_at"])
    assert n_ref >= 10
    assert prever_log(carregado, x, ref) == pytest.approx(prever_log(bundle, x, ref))
    assert prever_log(bundle, x, ref) == pytest.approx(bundle.modelo.predict(x)[0] + ref)
    with pytest.raises(ValueError, match="sem âncora"):
        prever_log(bundle, x, float("nan"))
    with pytest.raises(ValueError, match="degrau"):
        prever_log(bundle, x, ref, "chute")
    for degrau in ("inscritos", "mediana_do_setor"):  # cold start (#34): dá número, sem recusar
        assert prever_log(bundle, x, 6.0, degrau) == pytest.approx(
            bundle.modelo.predict(x)[0] + 6.0
        )
    assert carregado.versao == bundle.versao and bundle.versao.startswith("gbm-teste-")


def test_bundle_treina_so_ate_o_fim_do_treino(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    assert pd.Timestamp(bundle.ultimo_treino) < cfg_av["cal_inicio"] - pd.Timedelta(days=30)


def test_referencia_bate_com_a_do_treino(base, cfg, cfg_av):
    b, dur, _ = _partes(base)
    d = preparar(b, dur, cfg_av)
    alvo = d.index[d["linha"] & (d["channel_id"] == "UC2")][-10]
    linha = d.loc[alvo]
    anteriores = d[(d["channel_id"] == "UC2") & (d["published_at"] < linha["published_at"])]
    ref, n_ref = referencia(anteriores[["published_at", "y"]], linha["published_at"])
    assert ref == pytest.approx(linha["ref"]) and n_ref == linha["n_ref"]


FEATURES_PRODUCAO = """\
nome: producao
hipotese: H1
descricao: a lista da opcao_a_texto
features: [hist_n, hist_mediana, hist_media, hist_desvio, hist_med5, hist_med20, tendencia,
           dias_desde_ultimo, posts_30d, hora_local, dia_semana_local, feriado, idade_canal_dias,
           duracao_s, setor, txt_caracteres, txt_palavras, txt_hashtags, txt_emojis, txt_pergunta,
           txt_cta, txt_numeros, txt_frac_maiusculas, txt_ingles, txt_termos_setor]
max_arvores: 30
paciencia: 5
lgbm: {objective: regression, learning_rate: 0.1, num_leaves: 7, min_data_in_leaf: 20}
pca: {titulo: 2}
"""


def test_paridade_com_a_lista_de_producao_e_historico_completo(base, cfg_av, tmp_path):
    (tmp_path / "prod.yaml").write_text(FEATURES_PRODUCAO)
    cfg = ler_config(tmp_path / "prod.yaml")
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    d = preparar(b, dur, cfg_av)
    alvo = d.index[d["linha"] & (d["channel_id"] == "UC3")][-60]
    linha = d.loc[alvo]
    novo = {
        "published_at": linha["published_at"],
        "caption": linha["caption"],
        "sector": linha["sector"],
        "duration_s": float(dur[linha["post_id"]]),
        "embeddings": {"titulo": emb["titulo"].loc[linha["post_id"]].to_numpy()},
    }
    canal = d[d["channel_id"] == "UC3"][["channel_id", "published_at", "y"]]
    so_antes = canal[canal["published_at"] < linha["published_at"]]
    x_antes = features_de_um_short(bundle, so_antes, novo)
    x_tudo = features_de_um_short(bundle, canal, novo)  # inclui o próprio post e os seguintes
    nao_pca = [f for f in bundle.features if not f.startswith("titulo_pc")]
    np.testing.assert_allclose(
        x_antes[nao_pca].to_numpy(float)[0], d.loc[alvo, nao_pca].to_numpy(float), equal_nan=True
    )
    pd.testing.assert_frame_equal(x_antes, x_tudo)


def _extra(b):
    """Um bloco extra de 2 colunas por post_id (como o `vlm`), determinístico."""
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {"ex_a": rng.normal(size=len(b)), "ex_b": rng.normal(size=len(b))},
        index=pd.Index(b["post_id"], name="post_id"),
    )


def test_bundle_treina_com_extras(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    cfg.extras = ["meu"]
    cfg.features = [*cfg.features, "ex_*"]
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb, extras={"meu": _extra(b)})
    assert {"ex_a", "ex_b"} <= set(bundle.features) and "ex_*" not in bundle.features
    assert bundle.extras == ["meu"]
    cfg.features = [*cfg.features[:-1], "zz_*"]  # um curinga sem coluna: não vira controle calado
    with pytest.raises(ValueError, match="zz_"):
        treinar_bundle(cfg, b, dur, cfg_av, emb, extras={"meu": _extra(b)})


def _novo_com_extras(d, dur, emb, linha, extras):
    return {
        "published_at": linha["published_at"],
        "caption": linha["caption"],
        "sector": linha["sector"],
        "duration_s": float(dur[linha["post_id"]]),
        "embeddings": {"titulo": emb["titulo"].loc[linha["post_id"]].to_numpy()},
        "extras": extras,
    }


def test_paridade_treino_uso_com_extras(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    cfg.extras = ["meu"]
    cfg.features = [*cfg.features, "ex_*"]
    ext = _extra(b)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb, extras={"meu": ext})
    d = preparar(b, dur, cfg_av, cfg.lacuna_dias, extras={"meu": ext})
    alvo = d.index[d["linha"] & (d["channel_id"] == "UC1")][-40]
    linha = d.loc[alvo]
    anteriores = d[(d["channel_id"] == "UC1") & (d["published_at"] < linha["published_at"])]
    novo = _novo_com_extras(d, dur, emb, linha, ext.loc[linha["post_id"]].to_dict())
    x = features_de_um_short(bundle, anteriores[["channel_id", "published_at", "y"]], novo)
    assert list(x.columns) == bundle.features
    np.testing.assert_allclose(
        x[["ex_a", "ex_b"]].to_numpy(float)[0], d.loc[alvo, ["ex_a", "ex_b"]].to_numpy(float)
    )


def test_extra_ausente_levanta_a_menos_que_o_bloco_venha_declarado(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    cfg.extras = ["meu"]
    cfg.features = [*cfg.features, "ex_*"]
    ext = _extra(b)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb, extras={"meu": ext})
    d = preparar(b, dur, cfg_av, cfg.lacuna_dias, extras={"meu": ext})
    linha = d.loc[d.index[d["linha"] & (d["channel_id"] == "UC1")][-40]]
    posts = d[(d["channel_id"] == "UC1") & (d["published_at"] < linha["published_at"])]
    posts = posts[["channel_id", "published_at", "y"]]
    sem_coluna = _novo_com_extras(d, dur, emb, linha, {"ex_a": 1.0})  # falta ex_b
    with pytest.raises(ValueError, match="ex_b"):
        features_de_um_short(bundle, posts, sem_coluna)
    sem_bloco = _novo_com_extras(d, dur, emb, linha, {})
    with pytest.raises(ValueError, match="meu"):
        features_de_um_short(bundle, posts, sem_bloco)
    sem_bloco["extras_ausentes"] = {"meu": "vídeo sem áudio"}  # declarado: NaN e o motivo
    x = features_de_um_short(bundle, posts, sem_bloco)
    assert x[["ex_a", "ex_b"]].isna().all(axis=None)
    assert x.attrs["extracao_falhou"] == {"meu": "vídeo sem áudio"}
    fora = _novo_com_extras(d, dur, emb, linha, {"ex_a": 1.0})
    fora["extras_ausentes"] = {"outro": "x"}  # declarar outro bloco não perdoa a coluna que falta
    with pytest.raises(ValueError, match="ex_b"):
        features_de_um_short(bundle, posts, fora)


def test_bundle_valida_a_tabela_de_treino(base, cfg, cfg_av, monkeypatch):
    # F15 (princípio 6; aulas 1.8 e 3.5): o modelo de uso confere a mesma tabela que o runner,
    # com as features da config e antes de o PCA criar as colunas dos blocos
    import previa.modelo as modelo

    chamadas = []

    def espiao(d, features, lacuna_dias=None):
        chamadas.append((list(features), lacuna_dias))
        return validar_tabela_de_treino(d, features, lacuna_dias)

    monkeypatch.setattr(modelo, "validar_tabela_de_treino", espiao)
    b, dur, emb = _partes(base)
    treinar_bundle(cfg, b, dur, cfg_av, emb)
    assert chamadas == [(cfg.features, cfg.lacuna_dias)]
    assert not any(c.startswith("titulo_pc") for c in chamadas[0][0])


def test_bundle_recusa_a_tabela_que_quebra_o_contrato(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    dur.loc[b["post_id"].iloc[100]] = -1.0  # 2022-09: uma duração negativa numa linha de treino
    with pytest.raises(pa.errors.SchemaErrors, match="duracao_s"):
        treinar_bundle(cfg, b, dur, cfg_av, emb)


def _sha(caminho) -> str:
    return hashlib.sha256(open(caminho, "rb").read()).hexdigest()


@pytest.mark.usefixtures("mlflow_temporario")
def test_registrar_bundle_poe_o_run_no_model_version(base, cfg, cfg_av, tmp_path):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    assert bundle.run_id is None
    run_id = registrar_bundle(bundle, cfg_av, tmp_path / "exp.yaml", destino=tmp_path / "art")
    mv = model_version(bundle)
    assert mv["run_id"] == run_id and run_id[:8] in mv["rotulo"]
    assert mv["versao"] in mv["rotulo"] and mv["ultimo_treino"][:10] in mv["rotulo"]
    assert mv["hash_base"] and mv["hash_base"] != ""
    assert carregar(tmp_path / "art").run_id == run_id
    tags = mlflow.get_run(run_id).data.tags
    assert tags["papel"] == "modelo_de_uso" and tags["hash_base"] == mv["hash_base"]
    assert {"git_commit", "data_corte", "config"} <= set(tags)
    # o corte do modelo de uso é o fim do treino (cal_inicio - gap), não o início do walk-forward
    assert tags["data_corte"] == "2025-09-01"


@pytest.mark.usefixtures("mlflow_temporario")
def test_artefato_do_run_e_o_arquivo_salvo(base, cfg, cfg_av, tmp_path):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    destino = tmp_path / "art"
    run_id = registrar_bundle(bundle, cfg_av, tmp_path / "exp.yaml", destino=destino)
    baixado = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="modelo.joblib", dst_path=str(tmp_path / "baixado")
    )
    assert _sha(baixado) == _sha(destino / "modelo.joblib")


def test_model_version_json_ao_lado_do_pacote(base, cfg, cfg_av, tmp_path):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    bundle.run_id = "abcdef1234567890"
    salvar(bundle, tmp_path / "art")
    gravado = json.loads((tmp_path / "art" / "model_version.json").read_text())
    assert gravado == model_version(bundle)
    assert gravado["rotulo"].startswith(bundle.versao + " · run abcdef12 · treino até ")


@pytest.mark.usefixtures("mlflow_temporario")
def test_registrar_bundle_que_falha_deixa_a_pasta_intocada(
    base, cfg, cfg_av, tmp_path, monkeypatch
):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    destino = tmp_path / "art"
    registrar_bundle(bundle, cfg_av, tmp_path / "exp.yaml", destino=destino)
    antes = {n: (destino / n).read_bytes() for n in ("modelo.joblib", "model_version.json")}
    outro = treinar_bundle(cfg, b, dur, cfg_av, emb)
    outro.versao = "gbm-outro"

    def quebra(*a, **k):
        raise RuntimeError("mlflow fora do ar")

    monkeypatch.setattr(mlflow, "log_artifact", quebra)
    with pytest.raises(RuntimeError, match="mlflow fora do ar"):
        registrar_bundle(outro, cfg_av, tmp_path / "exp.yaml", destino=destino)
    assert outro.run_id is None
    assert {n: (destino / n).read_bytes() for n in antes} == antes
    assert sorted(p.name for p in destino.iterdir()) == sorted(antes)  # sem lixo temporário


def test_model_version_de_pacote_antigo_pede_o_registro(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    bundle = treinar_bundle(cfg, b, dur, cfg_av, emb)
    del bundle.__dict__["run_id"]  # um joblib antigo não tem o atributo (cai no padrão da classe)
    with pytest.raises(ValueError, match="registrar_bundle"):
        model_version(bundle)


@pytest.mark.usefixtures("mlflow_temporario")
def test_main_treina_com_a_base_da_config_de_avaliacao(base, cfg, tmp_path, monkeypatch, capsys):
    import previa.modelo as modelo
    from previa.experimento import hash_base

    arq = tmp_path / "base_propria.parquet"
    base.to_parquet(arq)
    aval = tmp_path / "aval.yaml"
    aval.write_text(AVALIACAO + f'base: "{arq}"\n')
    cfg_yaml = tmp_path / "exp.yaml"
    cfg_yaml.write_text(CONFIG)
    b, dur, emb = _partes(base)
    monkeypatch.setattr(modelo, "ler_duracoes", lambda: dur)
    monkeypatch.setattr(modelo, "ler_embeddings_do_bloco", lambda bloco: emb[bloco])
    monkeypatch.setattr(modelo, "ARTEFATOS", tmp_path / "art")
    destino = tmp_path / "art"
    assert main(["treinar", str(cfg_yaml), str(aval)]) == 0
    mv = json.loads((destino / "model_version.json").read_text())
    assert mv["base"] == str(arq)
    cortada = base[base["published_at"] < pd.Timestamp("2026-04-01", tz="UTC")]
    assert mv["hash_base"] == hash_base(cortada.reset_index(drop=True))
    assert mv["rotulo"] in capsys.readouterr().out


def _bundle(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    return treinar_bundle(cfg, b, dur, cfg_av, emb), (b, dur, emb)


def test_bundle_guarda_quantis_qchapeu_limiares_e_ancora(base, cfg, cfg_av):
    bundle, _ = _bundle(base, cfg, cfg_av)
    assert sorted(bundle.quantis) == [0.1, 0.5, 0.9]
    assert np.isfinite(bundle.q_chapeu)
    assert bundle.calibracao["inicio"] == "2025-10-01"
    assert bundle.calibracao["fim"] == "2026-04-01"
    assert set(bundle.limiares_flag) >= {"largura_log", "min_posts", "ultimo_treino"}
    assert bundle.limiares_flag["ultimo_treino"] == "2025-09-01"
    assert len(bundle.ancora["coef"]) == 2 and set(bundle.ancora["mediana_setor"]) == {
        "beleza",
        "automotivo",
    }
    json.dumps(bundle.limiares_flag)  # serializável
    json.dumps(bundle.ancora)


def test_q_chapeu_vem_so_da_calibracao(base, cfg, cfg_av):
    """Recalcula o q̂ à mão: só linhas da calibração, com o modelo do pacote."""
    from previa.experimento import aplicar_pca
    from previa.intervalo import ler_config as ler_cfg_q
    from previa.intervalo import prever_quantis

    alfa = ler_cfg_q(CONFIGS / "intervalo.yaml").alpha_cqr
    bundle, (b, dur, emb) = _bundle(base, cfg, cfg_av)
    d = preparar(b, dur, cfg_av, cfg.lacuna_dias)
    ids = d.index[
        d["linha"] & (d["published_at"] >= cfg_av["cal_inicio"])
        & (d["published_at"] < cfg_av["teste_inicio"]) & d["y_rel"].notna()
    ]  # fmt: skip
    cal = d.loc[ids].copy()
    tr = d.index[d["linha"] & (d["published_at"] < pd.Timestamp("2025-09-01", tz="UTC"))]
    cols, _ = aplicar_pca(
        emb["titulo"], d.loc[tr, "post_id"], cal["post_id"], 2, "titulo", cfg.seed
    )
    cal[list(cols.columns)] = cols.to_numpy()
    q, _ = prever_quantis(bundle.quantis, cal[bundle.features], cal["ref"], cfg)
    esperado = calcular_q_chapeu(escores_cqr(q[:, 0], q[:, 2], cal["y"]), alfa)
    assert bundle.q_chapeu == pytest.approx(esperado)
    assert len(ids) > 10 and bundle.calibracao["n"] == len(ids)


def test_calibracao_nao_passa_pelo_teto_do_treino(base, cfg, cfg_av, tmp_path):
    """`teto_por_canal` é do treino: a calibração tem o mesmo tamanho com ou sem teto."""
    b, dur, emb = _partes(base)
    sem = treinar_bundle(cfg, b, dur, cfg_av, emb)
    (tmp_path / "t.yaml").write_text(CONFIG + "teto_por_canal: 150\n")
    com = treinar_bundle(ler_config(tmp_path / "t.yaml"), b, dur, cfg_av, emb)
    assert com.calibracao["n"] == sem.calibracao["n"] > 50


def test_calibracao_exclui_linha_sem_ancora(base, cfg, cfg_av, tmp_path):
    """Canal que só existe na calibração: tem y, mas ref NaN; não entra (alvo absoluto)."""
    novo = base[base["channel_id"] == "UC0"].copy()
    novo = novo[novo["published_at"] >= pd.Timestamp("2025-11-01", tz="UTC")]
    novo["channel_id"] = "UC9"
    novo["post_id"] = novo["post_id"].str.replace("c0", "c9", regex=False)
    novo["duplicate_group"] = novo["post_id"]
    base2 = pd.concat([base, novo], ignore_index=True)
    (tmp_path / "a.yaml").write_text(CONFIG + "alvo: absoluto\n")
    cfg_abs = ler_config(tmp_path / "a.yaml")
    b, dur, emb = _partes(base2)
    bundle = treinar_bundle(cfg_abs, b, dur, cfg_av, emb)
    d = preparar(b, dur, cfg_av, cfg_abs.lacuna_dias)
    nc = (
        d["linha"] & (d["published_at"] >= cfg_av["cal_inicio"])
        & (d["published_at"] < cfg_av["teste_inicio"])
    )  # fmt: skip
    assert (nc & d["y"].notna()).sum() > (nc & d["ref"].notna()).sum()  # o canal novo existe
    assert bundle.calibracao["n"] == (nc & d["ref"].notna()).sum()


def test_prever_intervalo_aceita_o_degrau_do_cold_start(base, cfg, cfg_av):
    bundle, _ = _bundle(base, cfg, cfg_av)
    x = pd.DataFrame([dict.fromkeys(bundle.features, 1.0)])
    n, lo, hi = prever_intervalo(bundle, x, 6.0, "inscritos")
    assert n == pytest.approx(prever_log(bundle, x, 6.0, "inscritos")) and lo <= n <= hi
    with pytest.raises(ValueError, match="degrau"):
        prever_intervalo(bundle, x, 6.0, "chute")


def test_prever_intervalo_contem_o_numero_do_modelo_pontual(base, cfg, cfg_av):
    bundle, (b, dur, emb) = _bundle(base, cfg, cfg_av)
    posts = preparar(b, dur, cfg_av)[["channel_id", "published_at", "y"]]
    posts = posts[posts["channel_id"] == "UC0"]
    novo = {
        "published_at": pd.Timestamp("2026-01-20T22:00Z"),
        "caption": "Título novo",
        "sector": "beleza",
        "duration_s": 30.0,
        "embeddings": {"titulo": np.zeros(5)},
    }
    x = features_de_um_short(bundle, posts, novo)
    ref, _ = referencia(posts, novo["published_at"])
    numero, baixo, alto = prever_intervalo(bundle, x, ref)
    assert numero == pytest.approx(prever_log(bundle, x, ref))  # o número é o do pontual (#40)
    assert baixo <= numero <= alto


def test_pacote_antigo_sem_os_campos_novos_carrega(base, cfg, cfg_av, tmp_path):
    bundle, _ = _bundle(base, cfg, cfg_av)
    antigo = Bundle(
        bundle.modelo, bundle.features, bundle.pcas, bundle.versao, bundle.lacuna_dias,
        bundle.ultimo_treino, bundle.arvores,
    )  # fmt: skip
    salvar(antigo, tmp_path / "a")
    lido = carregar(tmp_path / "a")
    assert lido.quantis == {} and lido.q_chapeu is None and lido.limiares_flag == {}
    with pytest.raises(ValueError, match="quantis"):
        prever_intervalo(lido, pd.DataFrame(), 5.0)


def test_model_version_traz_a_calibracao(base, cfg, cfg_av):
    bundle, _ = _bundle(base, cfg, cfg_av)
    bundle.run_id = "abcdef0123456789"
    assert model_version(bundle)["calibracao"] == bundle.calibracao


def test_bundle_sem_calibracao_falha_explicita(base, cfg, cfg_av):
    b, dur, emb = _partes(base)
    cfg_av = {**cfg_av, "cal_inicio": pd.Timestamp("2030-01-01", tz="UTC"),
              "teste_inicio": pd.Timestamp("2030-02-01", tz="UTC")}  # fmt: skip
    with pytest.raises(ValueError, match="calibração"):
        treinar_bundle(cfg, b, dur, cfg_av, emb)


def test_calibracao_sobreposta_ao_treino_falha(base, cfg, cfg_av, monkeypatch):
    """5.3, erros 2 e 3: se a calibração tocar o treino, o pacote não sai."""
    import previa.modelo as modelo

    b, dur, emb = _partes(base)
    monkeypatch.setattr(modelo, "fim_do_treino", lambda _: pd.Timestamp("2025-11-01", tz="UTC"))
    with pytest.raises(AssertionError, match="treino"):
        treinar_bundle(cfg, b, dur, cfg_av, emb)


def test_cli_grava_pacote_que_outro_processo_carrega():
    """O `python -m previa.modelo` não pode gravar `__main__.Bundle` no pickle (21/10)."""
    from pathlib import Path

    fonte = (Path(__file__).parents[1] / "src" / "previa" / "modelo.py").read_text(encoding="utf-8")
    bloco = fonte.split('if __name__ == "__main__":')[-1]
    assert "from previa import modelo as _modelo" in bloco
    assert "_modelo.main()" in bloco
