# Lote da auditoria (itens sem decisão) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** fechar os itens da auditoria #30 que não pedem decisão do Vinicius: F39/F40, F18, F26, F33, F13, F1 (+ skill score da F25), F22 + F35, F15 e o baseline mais forte da F25.

**Architecture:** mudanças pequenas e aditivas em módulos que já existem (`experimento`, `metricas`, `features`, `baselines`, `erros`, `esquema`), um módulo novo de caminhos (`caminhos.py`) e um de validação das sondas (`sondas.py`). Nenhuma métrica que já existe muda de valor; o que entra no runner é novo e só aparece nos runs novos (ou pelo `painel RUN_ID`, na abertura das hipóteses).

**Tech Stack:** Python 3.12 (uv), pandas 3, numpy 2.5, LightGBM 4.7, MLflow 3.16, pandera 0.33, scikit-learn 1.9, pytest, ruff.

**Spec:** `docs/AUDITORIA.md`, seções 2 (as faltas) e 3 (o que as peças precisam cumprir). Para cada item, a linha da tabela da seção 2 é o requisito.

**Ordem de execução (no checkout principal, branch `feat/lote-auditoria`):**
- Onda 1: Task 1, sozinha (mexe no ambiente do uv).
- Onda 2, em paralelo: Task 2 ∥ Task 3 ∥ Task 4 (arquivos disjuntos).
- Onda 3, em paralelo: Tasks 5 → 6 → 7 → 8, em sequência (dividem `experimento.py`, `metricas.py` e `features.py`) ∥ Task 9.

## Global Constraints

- **Runs lacrados.** As H2 a H9 e as ablações estão lacradas até a abertura das hipóteses. Nenhum subagente roda `previa.experimento`, `previa.ablacao`, `previa.erros`, `comparar` ou o `painel` em runs reais, nem lê métricas de runs do MLflow. `carregar_base(final=True)` nunca é chamado.
- **Dados reais só onde a task diz:** a Task 3 (AUC das sondas) e a Task 9 (tabela de baselines) rodam na base real e devolvem a saída ao controlador. Fora isso, testes com dados sintéticos (`previa.sintetico.gerar_base` ou DataFrames pequenos), sem rede e sem LLM (o CI roda sem dados).
- **Princípio 2 (sem vazamento):** histórico só com posts publicados ao menos 30 dias antes, inclusive (a regra do `historico_do_canal` e da `mediana_madura`, com `np.searchsorted`).
- **Princípio 7 (falha nunca silenciosa):** NaN onde não pode haver vira `ValueError` com mensagem; aviso é `warnings.warn`, não `print`.
- **Aditivo:** nenhuma métrica, coluna de feature ou valor de run existente muda. Colunas novas no `preparar` não podem começar com os prefixos curinga das configs (`sonda_`, `vlm_`, `sinal_`, `aud_`, `ruido_`).
- **pandas 3:** Copy-on-Write (use `.loc`); `bool` com NaN vira `object` no join (converta para `float`).
- **Estilo:** nomes e comentários em português, docstring citando a aula, como o código em volta. Linha de 100 colunas.
- **Verificação de cada task:** `uv run pytest` (todos passam; diga quantos), `uv run ruff check .` e `uv run ruff format src tests`.
- **Git no checkout compartilhado:** `git add` só dos arquivos da task, nunca `-A` nem `.`. Se aparecer `index.lock`, espere 5 s e tente de novo. Commit em inglês (Conventional Commits), citando a aula, terminando com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Sem push.
- **Documentos:** não edite `docs/` (DECISOES, ROTEIRO, AUDITORIA). Toda escolha de encanamento vai no relatório final da task, numa linha, para o controlador registrar.
- **Chaves:** nunca imprima o `.env`; erro da API passa por `previa.coletor.sem_chave`.

## Review Focus

1. **Um teste que escreve no `data/` ou no `artefatos/` reais** depois que os caminhos viram absolutos (Task 2). Esperado: o `tests/conftest.py` aponta `PREVIA_DADOS` e `GALAXIES_ARTEFATOS` para uma pasta temporária antes de qualquer import do `previa`. Teste: `test_testes_nunca_usam_os_dados_reais`.
2. **Runs antigos sem a tag `data_corte`** (todos os lacrados) no `comparar` da abertura (Task 7). Esperado: aviso explícito e segue; cortes diferentes levantam erro. Teste: `test_conferir_cortes_avisa_run_sem_tag_e_recusa_cortes_diferentes`.
3. **Canal com menos de 10 posts maduros** nas faixas (Task 5). Esperado: "sem histórico", fora do F1, nunca contado como "abaixo". Teste: `test_faixa_sem_historico_fica_fora_do_f1`.
4. **NaN em `y` ou `p` no painel** (Task 6). Esperado: `ValueError`, não métrica calculada com NaN. Teste: `test_painel_recusa_nan`.
5. **A tabela de treino real com NaN legítimos** (`hist_*` dos primeiros posts, `duracao_s` sem duração) no pandera (Task 8). Esperado: passa; só os invariantes quebrados falham. Teste: `test_rodar_valida_a_tabela_de_treino` (a base sintética inteira passa) e os testes de recusa.

---

### Task 1: dependência do explicador e aviso do shap (F39, F40)

**Files:**
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces: `langchain_google_genai` importável no ambiente; pytest sem os avisos de depreciação do shap.

- [ ] **Step 1:** rode `uv run pytest -q 2>&1 | tail -15` e copie o texto exato dos 3 avisos do shap.
- [ ] **Step 2:** `uv add langchain-google-genai==4.4.0` (a versão validada na 7.3 com `langchain==1.4.2` e `langchain-core==1.6.5`). Confira com `git diff uv.lock` que nenhum pacote fixado no `pyproject.toml` mudou de versão. Se mudou, desfaça (`git checkout pyproject.toml uv.lock && uv sync`) e pare com o relatório.
- [ ] **Step 3:** em `[tool.pytest.ini_options]`, `filterwarnings` com um `ignore:<início da mensagem>:DeprecationWarning` para cada aviso do Step 1 (comentário: "shap 0.52; aula 6.6").
- [ ] **Step 4:** `uv run python -c "import langchain_google_genai"` sem erro; `uv run pytest -q` passa sem os avisos do shap.
- [ ] **Step 5:** commit `build(deps): add langchain-google-genai 4.4.0 and silence shap deprecation warnings (aulas 5.7, 6.2, 6.6)`.

### Task 2: caminhos absolutos e variáveis de ambiente (F18)

**Files:**
- Create: `src/previa/caminhos.py`, `tests/conftest.py`, `tests/test_caminhos.py`
- Modify: `src/previa/coletor.py:34`, `dados.py:22`, `splits.py:18`, `baselines.py:34`, `amostra.py:24`, `conferencia.py:30`, `duplicatas.py:31`, `features.py:35`, `texto.py:118`, `modelo.py:37`; `tests/test_coletor.py` (os 2 testes com `chdir`), `tests/test_curadoria.py` (os 2 testes com `chdir`)

**Interfaces:**
- Produces: `previa.caminhos.RAIZ`, `DADOS`, `CONFIGS`, `ARTEFATOS` (todos `Path` absolutos). `previa.coletor.DADOS` continua existindo (é `caminhos.DADOS`), porque muitos módulos o importam.

```python
RAIZ = Path(__file__).resolve().parents[2]          # src/previa/caminhos.py -> raiz do repo
DADOS = Path(os.environ.get("PREVIA_DADOS", RAIZ / "data"))
CONFIGS = RAIZ / "configs"
ARTEFATOS = Path(os.environ.get("GALAXIES_ARTEFATOS", RAIZ / "artefatos"))  # o nome da 6.1 e 6.6
```

Lidos na importação, como na 6.1 (erro comum 3: "o caminho do modelo precisa ser definido antes da primeira chamada"). Os defaults de função (`arquivo=DADOS / "cota.json"`) ficam ligados na importação; por isso o `conftest` define as variáveis antes de importar qualquer coisa do `previa`.

- [ ] **Step 1: testes que falham.** Em `tests/test_caminhos.py`:
  - `test_caminhos_valem_de_outro_diretorio`: `subprocess.run([sys.executable, "-c", "from previa import caminhos, splits; print(caminhos.CONFIGS); splits.ler_avaliacao()"], cwd=tmp_path, env=<os.environ sem PREVIA_DADOS e GALAXIES_ARTEFATOS>)` sai com 0 e imprime o `configs` do repo.
  - `test_variavel_dos_artefatos_vale`: com `env GALAXIES_ARTEFATOS=<tmp>/x`, o mesmo subprocesso imprime `caminhos.ARTEFATOS == <tmp>/x` e `modelo.ARTEFATOS == <tmp>/x`.
  - `test_testes_nunca_usam_os_dados_reais`: `caminhos.DADOS != caminhos.RAIZ / "data"` e `caminhos.ARTEFATOS != caminhos.RAIZ / "artefatos"` dentro da sessão de testes.
- [ ] **Step 2:** rode e veja falhar (`ModuleNotFoundError: previa.caminhos`).
- [ ] **Step 3:** crie `caminhos.py` e troque os 10 caminhos relativos pelos de `caminhos`. Crie `tests/conftest.py`: no topo, antes de qualquer import do `previa`, `os.environ["PREVIA_DADOS"]` e `os.environ["GALAXIES_ARTEFATOS"]` apontando para pastas de um `tempfile.mkdtemp(prefix="previa-testes-")`; uma fixture `autouse` esvazia essas pastas antes de cada teste (recriando-as vazias). Nos 4 testes com `monkeypatch.chdir(tmp_path)`, troque `tmp_path / "data"` por `caminhos.DADOS` (sem `chdir`).
- [ ] **Step 4:** `uv run pytest -q`: todos passam. Compare o nº de passed e skipped com o de antes da task e explique qualquer diferença no relatório.
- [ ] **Step 5:** `ls -la data/ artefatos/ 2>/dev/null | head` e `git status`: nada novo nem modificado nas pastas reais por causa dos testes.
- [ ] **Step 6:** commit `fix(caminhos): absolute paths and GALAXIES_ARTEFATOS for the MCP server (aulas 6.1, 6.5, 6.6)`.

### Task 3: validar as sondas da capa contra o VLM (F26)

**Files:**
- Create: `src/previa/sondas.py`, `tests/test_sondas.py`
- Modify: `configs/multimodal.yaml` (bloco `siglip.validacao`)

**Interfaces:**
- Produces: `validar_sondas(sondas: pd.DataFrame, rotulos: pd.DataFrame, pares: dict[str, str], auc_min: float) -> pd.DataFrame`, uma linha por par, colunas `sonda, rotulo, n, positivos, auc, passa`. `sondas`: índice post_id, colunas `sonda_<nome>_thumb`; `rotulos`: índice post_id, colunas bool do VLM. CLI `uv run python -m previa.sondas`.

Config (4.4, exercício: "fique com as que passam de ~0,7"):

```yaml
  validacao:                        # F26: AUC de cada sonda contra um campo da rubrica do VLM, nas capas da amostra
    auc_min: 0.7
    pares: {rosto: tem_rosto, falando: tem_rosto, texto: texto_na_tela, produto_beleza: close_up_produto}
```

- [ ] **Step 1: testes que falham.** `test_sonda_que_separa_tem_auc_1_e_passa` (sonda = rótulo + ruído pequeno, AUC 1,0, `passa` True); `test_sonda_ao_acaso_nao_passa` (seed fixa, AUC < 0,7, `passa` False); `test_rotulo_de_uma_classe_so_vira_nan` (`auc` NaN, `passa` NaN, sem exceção); `test_alinha_pelos_post_ids_em_comum` (ids só de um lado ficam fora; `n` conta a interseção).
- [ ] **Step 2:** rode e veja falhar.
- [ ] **Step 3:** implemente com `sklearn.metrics.roc_auc_score`. A CLI lê `DADOS / "sondas_thumb_v1.parquet"` (índice `video_id` = post_id) e `DADOS / "vlm" / "rubrica_v1.jsonl"` (o mesmo `drop_duplicates("post_id", keep="last")` do `experimento.ler_extra`), imprime a tabela, lista as 6 sondas sem rótulo barato como "sem rótulo" e grava `DADOS / "sondas_auc_v1.csv"`. Docstring: o rótulo é do vídeo inteiro e a sonda é da capa, então o AUC é um piso (ruído no rótulo), e isso entra nas limitações.
- [ ] **Step 4:** `uv run pytest tests/test_sondas.py -v` passa; depois a suíte inteira.
- [ ] **Step 5:** rode `uv run python -m previa.sondas` na base real e cole a tabela inteira no relatório.
- [ ] **Step 6:** commit `feat(sondas): validate cover probes with AUC against the VLM rubric (aula 4.4)`.

### Task 4: corte por fonte do canal na análise de erros (F33)

**Files:**
- Modify: `src/previa/erros.py:31-38` e `tabela_do_run`; `tests/test_erros.py`

**Interfaces:**
- Produces: `origem_do_canal(fonte: pd.Series) -> pd.Series` ("busca" se o texto tem "busca", sem distinguir maiúsculas; "fora da busca" caso contrário; NaN vira "sem registro"). `CORTES["origem"] = (None, None)`.

- [ ] **Step 1: teste que falha.** `test_origem_do_canal_separa_a_busca`: `["busca:canal", "lista manual+busca", "colab", "lista manual", None]` → `["busca", "busca", "fora da busca", "fora da busca", "sem registro"]`; e `por_corte` numa tabela pequena com a coluna `origem` devolve os dois grupos com `n` e `canais` certos.
- [ ] **Step 2:** rode e veja falhar.
- [ ] **Step 3:** implemente; `tabela_do_run` mapeia `canais["fonte"]` e cria `t["origem"]`. A pergunta da 1.4: "o modelo erra mais nos canais que vieram da busca?".
- [ ] **Step 4:** testes passam (o arquivo e a suíte).
- [ ] **Step 5:** commit `feat(erros): cut errors by how the channel was found (aula 1.4)`.

### Task 5: faixas pelos quantis do histórico do canal (F13)

**Files:**
- Modify: `src/previa/features.py` (função nova), `src/previa/metricas.py:80-91`, `src/previa/experimento.py` (`preparar`); `tests/test_features.py`, `tests/test_metricas.py`

**Interfaces:**
- Produces:
  - `features.quantis_do_canal(df, lacuna_dias=30, min_posts=10) -> pd.DataFrame` com `faixa_p25`, `faixa_p75`, `faixa_p95` (quantis de `y` dos posts maduros com y, a mesma regra de corte do `historico_do_canal`; NaN com menos de `min_posts`). Não é feature do modelo; é a régua das faixas e, depois, da tool.
  - `metricas.faixa_do_canal(v, p25, p75, p95) -> np.ndarray` de str, a regra da 1.6: `v < p25` "abaixo"; `v <= p75` "na média"; `v <= p95` "acima"; senão "viral"; qualquer quantil NaN → "sem histórico".
  - `metricas.f1_faixas_canal(y, p, p25, p75, p95) -> float`: F1 macro nas 4 faixas, só nas linhas com histórico (NaN se nenhuma).
  - `preparar` passa a juntar `quantis_do_canal(d, 30, min_posts_alvo)` (sempre 30 dias, como a `ref`).
  - `faixa` e `f1_faixas` ficam (são o código da 2.4); o comentário da linha 80 passa a dizer que o projeto usa `faixa_do_canal` (#7, 1.6).

- [ ] **Step 1: testes que falham.**
  - `test_quantis_do_canal_caso_a_mao`: um canal com 10 posts maduros de y = 1..10 → p25 = 3,25, p75 = 7,75, p95 = 9,55 (interpolação linear do pandas); um post publicado 29 dias antes do alvo não entra, um de 30 dias entra.
  - `test_quantis_perturbar_o_futuro_nao_muda`: mudar o y de posts posteriores não muda os quantis de um post.
  - `test_faixa_do_canal_bordas`: com (3,25; 7,75; 9,55), os valores 3 → "abaixo", 3,25 → "na média", 7,75 → "na média", 8 → "acima", 9,55 → "acima", 9,6 → "viral".
  - `test_faixa_sem_historico_fica_fora_do_f1`: linhas com quantis NaN viram "sem histórico" e não mudam o F1 das outras.
- [ ] **Step 2:** rode e veja falhar.
- [ ] **Step 3:** implemente.
- [ ] **Step 4:** testes passam (os arquivos e a suíte; os testes de paridade continuam verdes).
- [ ] **Step 5:** commit `feat(faixas): bands from the channel's mature history quantiles, decision #7 (aula 1.6)`.

### Task 6: painel de métricas e skill score (F1, parte da F25)

**Files:**
- Modify: `src/previa/metricas.py` (função nova), `src/previa/experimento.py` (`avaliar`, `main`); `tests/test_metricas.py`, `tests/test_experimento.py`

**Interfaces:**
- Consumes: `f1_faixas_canal` e as colunas `faixa_p25/p75/p95` do `preparar` (Task 5).
- Produces:
  - `metricas.ic_por_canal(fn, t: pd.DataFrame, n_boot=2000, seed=0) -> np.ndarray` ([lo, hi]): reamostra canais (`t["channel_id"]`) com reposição, junta as linhas dos sorteados e aplica `fn(linhas) -> float`; percentis 2,5 e 97,5 ignorando NaN.
  - `experimento.painel(a: pd.DataFrame, n_boot=2000, seed=0) -> dict`: `a` tem `channel_id, y, p, p_base, faixa_p25, faixa_p75, faixa_p95, dobra`. Chaves: `mae_log`, `e_mae` (= e^MAE, o fator típico de erro), `rmse_log`, `mape`, `spearman_global`, `f1_faixas`, `skill_mae` (= 1 − MAE do modelo / MAE do baseline, 2.1, regra 3), cada uma com `_lo` e `_hi`; e, por dobra k, `dobra{k}_mae_log` e `dobra{k}_spearman_intra` (só o valor). NaN em `y` ou `p` levanta `ValueError`. A cobertura fica de fora até a M6 (sem intervalo ainda).
  - `avaliar` acrescenta `**painel(...)` ao dict, com o prefixo `painel_` em cada chave (nenhuma chave antiga muda).
  - CLI `uv run python -m previa.experimento painel RUN_ID`: lê o `previsoes.json` do run, refaz o `preparar` na base de desenvolvimento e imprime o painel por avaliação, sem treinar (para os runs lacrados, na abertura). A parte pura é `painel_das_previsoes(d, prev, n_boot, seed) -> pd.DataFrame`, testada com uma tabela pequena; a CLI nunca roda nesta task.

- [ ] **Step 1: testes que falham.**
  - `test_ic_por_canal_um_canal_so`: com um canal, o IC é o próprio valor nas duas pontas.
  - `test_painel_caso_a_mao`: 2 canais, 3 posts cada, p = y + 0,5 em todos → `mae_log` 0,5, `e_mae` = e^0,5, `rmse_log` 0,5; p_base = y + 1 → `skill_mae` 0,5.
  - `test_painel_recusa_nan`: um `p` NaN → `ValueError`.
  - `test_rodar_registra_o_painel`: no `test_rodar_ponta_a_ponta` sintético, `painel_mae_log`, `painel_skill_mae` e `painel_f1_faixas` existem nas duas avaliações e são finitos, e as chaves antigas (`manchete_modelo`, `ganho`, ...) têm os mesmos valores de antes.
  - `test_painel_das_previsoes_igual_ao_do_rodar`: o painel recalculado da `tabela_de_previsoes` bate com o do `rodar`.
- [ ] **Step 2:** rode e veja falhar.
- [ ] **Step 3:** implemente. Bootstrap com posições pré-calculadas por canal (sem `concat` de DataFrames a cada reamostra), para os ~6 mil posts do walk-forward caberem em segundos.
- [ ] **Step 4:** testes passam; a suíte inteira continua no mesmo tempo, ±20 s.
- [ ] **Step 5:** commit `feat(experimento): metrics panel with channel-bootstrap CIs and skill score (TAPI 6.1; aulas 2.1, 2.3, 2.4)`.

### Task 7: data de corte no run e importância por ganho (F22, F35)

**Files:**
- Modify: `src/previa/experimento.py` (`rodar`, `registrar`, `comparar`, `main`, `NAO_METRICA`); `tests/test_experimento.py`

**Interfaces:**
- Produces:
  - `registrar(cfg, caminho, resultado, base, tags=None, cfg_av=None)`: tags novas `data_corte` (`str(cfg_av["walk_forward"]["inicio"].date())`) e `gap_dias`; `mlflow.log_dict` do `configs/avaliacao.yaml` cru como `avaliacao.json`; `mlflow.log_dict` da importância como `importancia_gain.json`. Com `cfg_av=None`, usa `ler_avaliacao()`. A ablação continua chamando sem `cfg_av`.
  - `conferir_cortes(tags: dict[str, dict], corte_atual: str) -> None`: run sem `data_corte` → `warnings.warn` ("run X sem data_corte, anterior à F22: conferido só pelo hash da base"); runs com cortes diferentes entre si ou do atual → `ValueError`. O `comparar` chama antes do hash.
  - Em `rodar`, cada avaliação ganha `importancia` (`pd.Series` com o ganho somado nas dobras, `modelo.feature_importance("gain")` por `modelo.feature_name()`, normalizado para somar 1, em ordem decrescente) e `importancia_maior_fatia` (float). `"importancia"` entra em `NAO_METRICA`.
  - Alarme (2.2: "importância dominante"): `main` emite `warnings.warn` quando `importancia_maior_fatia > 0,5`, com o nome da feature e a fatia.

- [ ] **Step 1: testes que falham.**
  - `test_conferir_cortes_avisa_run_sem_tag_e_recusa_cortes_diferentes`: `{"a": {}, "b": {"data_corte": "2025-01-01"}}` com atual "2025-01-01" → `pytest.warns(UserWarning)`; `{"a": {"data_corte": "2025-01-01"}, "b": {"data_corte": "2024-07-01"}}` → `ValueError`.
  - `test_importancia_soma_1_e_pega_a_feature_vazada`: no `rodar` sintético, `importancia.sum() ≈ 1` e o índice está nas features; com uma feature `vazada` = `y_rel` + ruído pequeno na config, `importancia.index[0] == "vazada"` e `importancia_maior_fatia > 0,5`.
  - `test_registrar_grava_corte_e_avaliacao`: com `mlflow.set_tracking_uri` para `sqlite:///<tmp_path>/mlflow.db` (monkeypatch, nunca o `mlflow.db` real), o run tem as tags `data_corte` e `gap_dias` e os artefatos `avaliacao.json` e `importancia_gain.json`.
- [ ] **Step 2:** rode e veja falhar.
- [ ] **Step 3:** implemente.
- [ ] **Step 4:** testes passam (o arquivo, `tests/test_ablacao.py` e a suíte).
- [ ] **Step 5:** commit `feat(experimento): cutoff date tag, refuse mixed cutoffs and log gain importance (aulas 2.2, 2.5)`.

### Task 8: pandera na tabela de treino (F15)

**Files:**
- Modify: `src/previa/esquema.py` (modelo novo), `src/previa/experimento.py` (`rodar`), `src/previa/modelo.py` (`treinar_bundle`); `tests/test_esquema.py`, `tests/test_experimento.py`

**Interfaces:**
- Produces: `esquema.TabelaTreino(pa.DataFrameModel)` (não estrito: a tabela tem dezenas de colunas; confere só os invariantes) e `esquema.validar_tabela_de_treino(d: pd.DataFrame, features: list[str]) -> None`, que valida `TabelaTreino` em `d[d["linha"]]` (`lazy=True`) e confere que cada feature existe, é numérica e não tem ±inf (NaN pode: o LightGBM trata).
  - `TabelaTreino`: `post_id` str único; `channel_id` str; `published_at` datetime com fuso UTC; `y` float ≥ 0, não nulo; `setor` em {0.0, 1.0}; `hist_n` ≥ 0; `duracao_s` ≥ 0, nulo pode; `@pa.dataframe_check`: onde `hist_ultimo_ts` não é nulo, `hist_ultimo_ts < published_at` (o teste da M3, princípio 2).
  - Chamado em `rodar` logo depois do `expandir_features` e em `treinar_bundle` antes do treino (princípio 6).

- [ ] **Step 1: testes que falham.**
  - `test_rodar_valida_a_tabela_de_treino`: a base sintética inteira passa pelo `rodar` (com os NaN legítimos dos primeiros posts).
  - `test_tabela_de_treino_recusa`: a partir do `preparar` sintético, cada uma destas quebras levanta `pandera.errors.SchemaErrors`: um `hist_ultimo_ts` igual ao `published_at`; `setor` = 2.0; `post_id` repetido entre as linhas; e `validar_tabela_de_treino` levanta `ValueError` com `inf` numa feature e com uma feature que não existe.
- [ ] **Step 2:** rode e veja falhar.
- [ ] **Step 3:** implemente.
- [ ] **Step 4:** testes passam (os arquivos, `tests/test_modelo.py` e a suíte).
- [ ] **Step 5:** commit `feat(esquema): pandera contract on the training table (principle 6; aulas 1.8, 3.5)`.

### Task 9: baselines mais fortes (F25)

**Files:**
- Modify: `src/previa/baselines.py`, `configs/baselines.yaml`; `tests/test_baselines.py`

**Interfaces:**
- Produces:
  - `media_ultimos_k(df, k, dias=30, min_posts=3) -> pd.Series`: média de `y` dos últimos `k` posts maduros com y (a regra do `mediana_madura`: 30 dias inclusive); usa os que houver quando há entre `min_posts` e k; NaN com menos de `min_posts`.
  - `mediana_janela(df, janela_dias=90, dias=30, min_posts=3) -> pd.Series`: mediana de `y` dos posts maduros publicados nos `janela_dias` antes do limite de maturidade (de t − 120 a t − 30 dias, inclusive); com menos de 3 na janela, os 10 últimos maduros (o `historico_recente` da 6.1); NaN com menos de `min_posts` maduros.
  - `configs/baselines.yaml`: `ks: [3, 5, 10, 20]` e `janela_baseline_dias: 90`.
  - `leitura_janela` ganha as colunas `p_ultimos_{k}` e `p_mediana_90d` no painel (mesmas linhas para todos: 2.1, regra 3) e uma linha final dizendo qual k teve a maior `manchete_janela_peso_posts` (o k "escolhido na validação": o walk-forward de 2025, com o teste trancado).
- [ ] **Step 1: testes que falham.** `test_media_ultimos_k_caso_a_mao` (canal com y = 1..6 maduros: k = 3 → 5,0; k = 10 → 3,5; com 2 maduros → NaN); `test_ultimos_k_so_maduros` (post de 29 dias fora, de 30 dentro); `test_mediana_janela_cai_para_os_10_ultimos`; `test_baselines_fortes_perturbar_o_futuro_nao_muda`; `test_leitura_janela_mesmas_linhas` (todas as colunas do painel com o mesmo `linhas`).
- [ ] **Step 2:** rode e veja falhar.
- [ ] **Step 3:** implemente.
- [ ] **Step 4:** testes passam (o arquivo e a suíte).
- [ ] **Step 5:** rode `uv run python -m previa.baselines janela` na base real (só baselines: nenhum run de modelo) e cole a tabela no relatório.
- [ ] **Step 6:** commit `feat(baselines): last-k mature mean and 90-day median as stronger baselines (aula 2.1)`.

---

**Fora deste plano (o controlador faz):** os registros F4, F38 e F43 no `docs/DECISOES.md` e as escolhas de encanamento que as tasks relatarem.
