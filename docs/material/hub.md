# Hub do case Galaxies

Fonte: `~/autoestudos/autoestudos/docs/projetos/galaxies/`. São 10 páginas: visão geral, roadmap, monte sua base,
Stack 2026, leituras, templates, autoavaliação, troubleshooting, glossário e laboratório.

## Visão geral (`index.mdx`)

- **Datas:** kickoff em 23/09/2026; entrega em **23/10/2026**.
- **Pergunta norteadora:** é possível prever, antes da publicação, como um público real vai reagir a um criativo, e
  com que margem de erro?

**Decisões de escopo:**

| Tema | Decisão |
| :--- | :--- |
| Momento | antes de publicar |
| Plataforma | YouTube Shorts pela Data API v3 |
| Setores | pelo menos 2 de 3 |
| Multimodal | **obrigatório em versão leve:** thumbnail ou frames, transcrição e atributos por VLM |
| Acurácia | sem meta rígida; bater o baseline e ordenar bem; ~60% par-a-par prova o conceito |
| Interface | tool obrigatória (função + MCP); Streamlit opcional |
| Explicabilidade | obrigatória: SHAP + subagente em linguagem natural |
| Agnosticismo | opcional |

**Núcleo obrigatório:**

1. base documentada;
2. modelo Opção A: metadados + texto, gradient boosting e baseline (a média histórica do canal);
3. multimodal leve (Opção B reduzida), com o ganho medido sobre a Opção A;
4. experimento: validação temporal e por canal, MAE em log, MAPE, Spearman, par-a-par e análise de erros;
5. intervalo e sinal de baixa confiança;
6. **agente em LangGraph** com as 5 tools;
7. função Python, servidor MCP e demo executável;
8. README e registro de experimentos (MLflow).

**Extensões:** o módulo 7. **Fora do escopo:** front-end elaborado, integração real com a Galaxies, séries temporais do
mesmo post e personas.

**Contrato da tool.**

- Entrada: `creative`, `caption`, `planned_datetime`, `channel_id` ou `channel_stats`, e `sector`.
- Saída: `predicted_metric`, `interval_low`, `interval_high`, `confidence_flag`, `relative_to_channel`,
  `top_drivers`, `explanation` e `comparables`.
- Esquema mínimo da base: `post_id`, `channel_id`, `published_at`, `caption`, `media_ref`, `metric_target`,
  `channel_stats_at_collection` e `sector`.

**Entregáveis:**

1. agente funcionando;
2. modelo treinado;
3. experimento (hipótese; validação temporal; erro e ordenação; cobertura e ganho multimodal; análise de erros);
4. demo com canais **fora do treino** e métrica real conhecida;
5. README.

## Roadmap (`roadmap.mdx`): definições de pronto

- **Semana 1, base:**
  - problema no README;
  - 60+ canais;
  - coletor idempotente;
  - filtros de Short e de 30 dias;
  - alvo log1p ou relativo;
  - rascunho do data card.
- **Semana 2, avaliação e modelo tabular:**
  - split temporal e por canal;
  - baseline com MAE em log, Spearman e par-a-par;
  - features de canal sem vazamento, com teste;
  - features de texto com PCA só no treino;
  - LightGBM ou CatBoost batendo o baseline;
  - MLflow;
  - cache de thumbnails.
- **Semana 3, multimodal, incerteza e explicação:**
  - camada 1 offline;
  - camada 2 (VLM com JSON Pydantic);
  - ablação com bootstrap pareado;
  - intervalo de 80% (quantis + conformal) com cobertura no teste;
  - `confidence_flag`;
  - `top_drivers`;
  - análise de erros.
- **Semana 4, agente:**
  - `prever_criativo(req) -> PredictionResponse`;
  - grafo LangGraph com as 5 tools;
  - validador da explicação;
  - MCP testado no Inspector;
  - testes: contrato, `low ≤ pred ≤ high` e "número do agente = número do modelo";
  - demo fora do treino.
- **Reta final:** README, cards, relatório de experimento, demo ensaiada com gravação, autoavaliação.

**Se o tempo apertar, nesta ordem:**

1. base, split e baseline;
2. GBM batendo o baseline;
3. função e MCP mínimo;
4. multimodal camada 1 e ablação;
5. intervalo;
6. SHAP e explicador;
7. VLM, LangGraph completo e extensões.

"Não pule etapas para chegar ao agente." Um resultado negativo bem medido também é resultado: a literatura espera
de +0,02 a +0,04 de ganho do multimodal.

## Monte sua base (`monte-sua-base.mdx`)

- **Metas:** 5 a 15 mil Shorts de 60+ canais, com idade de 30+ dias e `collected_at`.
- **Passos:**
  1. chave restrita;
  2. planilha de curadoria;
  3. coletor pela playlist de uploads;
  4. filtros (≤ 180 s desde 15/10/2024; curtidas ocultas viram NaN; chave `video_id`);
  5. armazenamento com `collected_at`;
  6. atualizar as estatísticas, com recoleta antes do treino final (`videos.batchGetStats`, de balde próprio);
  7. data card.
- No Prévia, a base está pronta (ver [módulo 1](modulo-1.md)). Pendência: a recoleta antes do treino final, que o
  Claude confere na Fase 2.

## Stack 2026 (`stack-2026.mdx`), verificada em 27/09/2026

- **Dados e modelo:** pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, pyarrow 25.0.1, duckdb 1.5.5, scikit-learn 1.9.1,
  lightgbm 4.7.0, catboost 1.2.10, xgboost 3.4.1, optuna 5.0.0, shap 0.52.0, mapie 1.5.0, mlflow 3.16.1,
  pydantic 2.13.5 e pandera 0.33.1.
- **Coleta:** google-api-python-client 2.200.0, isodate 0.7.2, tenacity 9.1.4 e httpx 0.28.1.
- **Multimodal:** torch 2.14.0, transformers 5.17.0, sentence-transformers 6.1.0, opencv-python-headless 5.0.0.93,
  scenedetect 0.7.1, faster-whisper 1.2.1, librosa 1.0.0, `yt-dlp[default]==2026.8.19` e google-genai 2.25.0.
- **Agente:** langgraph 1.2.12, langchain 1.4.2, langchain-core 1.6.5, langchain-mcp-adapters 0.3.2, **fastmcp 3.4.7 +
  mcp 1.30.0** e faiss-cpu 1.15.1.
- **Pegadinha do MCP:** FastMCP 4.x e o SDK `mcp` 2.x quebram o adapter, que exige `mcp<2`.
- **O que mudou desde o briefing:**
  - o `decord` está abandonado;
  - o MediaPipe só tem a Tasks API;
  - o PySceneDetect 0.7 mudou a API de timestamps;
  - o yt-dlp exige Deno;
  - `search.list` tem 100 chamadas por dia;
  - as views mudaram de regra (prefira curtidas);
  - o `create_react_agent` deu lugar ao `create_agent`;
  - o SHAP exige Python ≥ 3.12;
  - o MAPIE 1.x tem API nova, e o helper de split é aleatório.

## Leituras comentadas (`leituras.mdx`)

Como ler um paper em 15 minutos: qual é o alvo, qual é o split, o que se sabia antes da publicação, qual é o
baseline, e se dá para reproduzir.

**Papers e datasets:**

| Referência | O que é | Ressalva |
| :--- | :--- | :--- |
| SMP Challenge | MAPE de 0,1754 em 2025 e 0,1529 em 2026 | o alvo é outro |
| MVP (vencedor de 2025) | X-CLIP + PCA, Word2Vec, histórico do usuário, CatBoost com Huber | — |
| HyperFusion | trilha de imagem | não é a de vídeo |
| SMTPD | long-form | — |
| MicroLens | — | — |
| Gupta et al. (2025) | o mais parecido com o case: rubrica VLM + clusters + XGBoost; Spearman 0,71, par-a-par 76,3% | split aleatório |
| Kayal et al. (2025) | LLM zero-shot com 82% | — |
| MMRA | retrieval de comparáveis | só vizinhos anteriores |
| Angelopoulos e Bates; Romano et al. | conformal e CQR | — |
| Lundberg e Lee | SHAP explica o modelo | não a causa |

**Números para comparar no README:** Spearman de 0,55 a 0,85 antes da publicação. O intra-canal costuma ser menor.

## Templates (`templates.mdx`)

- **README:** problema e recorte; base; arquitetura; instalação; modelos; metodologia; resultados (tabela com
  baseline, Opção A, +visual, +VLM e cobertura de 80%); limitações; evolução.
- **Data card**, **model card** e **relatório de experimento** (hipótese, setup, resultados com Δ e IC 95% por
  bootstrap, análise de erros, conclusão).
- **Roteiro de demo** de 5 a 7 min, com plano B: gravação e respostas em JSON.

## Autoavaliação (`autoavaliacao.mdx`)

**Camadas do multimodal:**

| Camada | O que inclui |
| :--- | :--- |
| 1 · mínima, obrigatória | frames, SigLIP ou CLIP, faster-whisper, OCR, rosto e objetos, OpenCV, PySceneDetect |
| 2 · recomendada | + VLM com rubrica em JSON e clusters |
| 3 · ambiciosa | + X-CLIP, FAISS e ensemble |

**Checklists:**

- **Agente:** contrato, MCP, LangGraph com tools, "o número vem do modelo" com teste, explicação validada.
- **Modelo:** data card, features por bloco, baseline, GBM batendo o baseline, artefatos versionados.
- **Experimento:** hipótese antes; temporal e por canal; MAE e MAPE; Spearman e par-a-par; cobertura por setor;
  ablação com IC; análise de erros.
- **Demo:** fora do treino, com plano B.
- **README:** as 9 seções.

**O que diferencia os melhores projetos:**

- avaliação honesta, com o otimismo do CV aleatório medido;
- resultados com IC;
- análise de erros com exemplos concretos;
- explicação que não inventa;
- README que roda do zero.

## Troubleshooting (`troubleshooting.mdx`): índice por sintoma

- **Coleta:**
  - `quotaExceeded`: esperar o reset, sem retry;
  - `rateLimitExceeded` e 5xx: backoff com jitter;
  - `likeCount` ausente vira NaN;
  - "Shorts" pela regra de ≤ 180 s.
- **Mídia:**
  - yt-dlp "not a bot": usar IP residencial;
  - erro de JS ou PO token: Deno;
  - Short baixado em 144×256: usar `-S "res:360"`;
  - transformers 5: usar `.pooler_output`;
  - Gemini 429 e `resp.parsed None`;
  - VLM com JSON inválido: Pydantic e temperatura 0;
  - MediaPipe: Tasks API, e aborta no Mac;
  - librosa não abre `.mp4`;
  - PySceneDetect 0.7;
  - Colab cai: cache por vídeo;
  - T4: `eager` + 4 bits.
- **Ambiente:**
  - `ImportError FastMCP`: fixar `mcp==1.30.0`;
  - adapter lento: sessão única;
  - Inspector sem `-m`: usar o caminho do arquivo e `PYTHONPATH`;
  - SHAP exige Python 3.12;
  - `libomp`: `brew install libomp`;
  - `MapieRegressor` mudou: usar `SplitConformalRegressor`;
  - `create_react_agent` deu lugar ao `create_agent`;
  - porta 5000: usar `--port 5001`;
  - `artifact_path` deu lugar a `name=`;
  - **segfault LightGBM + torch:** importar `lightgbm` primeiro, ou usar processos separados;
  - **segfault FAISS + LightGBM:** `OMP_NUM_THREADS=1`;
  - pandas 3: `bool` vira `object` no join; `ChainedAssignmentError` pede `.loc`.
- **Resultados estranhos:**
  - métricas boas demais (Spearman 0,95): checar vazamento (views, histórico, split, fit só no treino, duplicatas);
  - não bate o baseline: alvo relativo, histórico mais rico, early stopping, intra-canal, volume por canal;
  - cobertura de 65% para 80%: conformal na fatia recente, por setor, recalibrar;
  - salto nas views em 2025: mudança de regra.

## Laboratório e glossário

O laboratório reúne os interativos com dados simulados: QuotaCalculator, LongTailExplorer, LeakageDetective,
SplitExplorer, MetricsPlayground, TreeSplitter, BoostingPlayground, VideoToFeatures, EmbeddingMap,
VlmBudgetCalculator, IntervalExplorer, ShapleyGame, ShapWaterfall, AgentTrace, McpFlow e WhatIfSimulator. Servem para
revisar um conceito em 10 minutos quando não há tempo para o exercício. O glossário é um componente interativo do
site.
