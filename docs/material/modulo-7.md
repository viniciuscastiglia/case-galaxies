# Módulo 7: Ir além (5 aulas opcionais, 240 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/7-ir-alem/`.

As cinco aulas são marcadas como opcionais ("semana 4+") e são independentes entre si. No Prévia, a leitura fica
**para depois de 23/10**. Algumas ideias já servem como referência na entrega; estão sinalizadas em "No Prévia".

**Ambiente do módulo.** pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, pandera 0.33.1,
pydantic 2.13.5, pyyaml 6.0.3, mlflow 3.16.1, fastmcp 3.4.7, mcp 1.30.0, mapie 1.5.0, langchain 1.4.2,
langchain-core 1.6.5, transformers 5.17.0 e torch 2.14.0. Na 7.4, também `av==18.1.0`.

## 7.1 Pipeline agnóstico: treinar para a base de um cliente

55 min · AI Eng + MLOps · depois de 23/10

- **Ideia.** Trocar as decisões manuais por quatro peças:
  - um contrato (pandera para a base, Pydantic para a configuração em YAML, sempre com `safe_load`);
  - decisões automáticas documentadas;
  - artefatos versionados pelo hash da base e da configuração, com model card;
  - promoção humana: o MLflow registra só candidatos, e o alias `@champion` é posto por uma pessoa.
- **Constrói.**
  - `BaseMinima` (`strict=False`, `coerce=True`, validação com `lazy=True`).
  - `ConfigTreino`: `alvo` padrão `relativo_canal`, `idade_minima_dias=30`, `defasagem_historico_dias=7`,
    `frac_teste=0.2`, `quantis=(0.1, 0.9)`. O `pattern` no nome do cliente é segurança contra caminhos como `../`.
  - Degradação graciosa por blocos: um bloco sem colunas vai para `blocos_ignorados`. O `snapshot_canal` fica
    desligado, porque os inscritos da coleta vazam.
  - `par_a_par_intra_canal` com `np.subtract.outer` (empate vale 0,5).
  - LightGBM quantílico com hiperparâmetros fixos e conservadores.
  - Avisos no relatório: "GBM não venceu o baseline", "intervalo subcobre".
  - A tool MCP `treinar_modelo`, com `job_id` + `status_treino`, fila de 1 worker, pasta de uploads com caminho
    resolvido e limite de tamanho.
- **No projeto Galaxies.** Uma base sem colunas de vídeo treina mesmo assim, com canal, tempo e texto: 57,8% dos pares
  contra 63,5% com vídeo. A resposta do agente precisa avisar que o conteúdo visual não foi avaliado.
- **Erros comuns.**
  1. Validar só os nomes das colunas.
  2. Usar `channel_stats_at_collection` como feature.
  3. Tratar coluna ausente como erro fatal.
  4. Deixar a tool promover o próprio modelo.
  5. Treinar de forma síncrona dentro da chamada MCP.
  6. Aceitar qualquer caminho de arquivo.
- **Exercício.** Checagem de DataFrame no pandera: cada canal precisa de 10 ou mais posts com curtidas públicas.
- **No Prévia.** Referência para a `par_a_par_intra_canal` (M2) e para os avisos do model card. A defasagem de 7 dias
  conflita com os 30 dias da 2.2 (ver [conflitos](conflitos.md)).

## 7.2 What-if: otimizando legenda, horário e gancho

45 min · ML + AI Eng · depois de 23/10

- **Ideia.** O what-if re-pontua variações com o modelo, e a resposta é sobre o modelo, não sobre o mundo. Antes de
  virar sugestão, cada variação passa por três filtros: ser controlável, estar no suporte dos dados e ter efeito
  estável.
- **As três armadilhas.**
  - Correlação herdada: sem o gancho no modelo, as 19h valem +23% contra +13% reais.
  - Extrapolação: a árvore devolve a folha da borda.
  - Efeito de ruído.
- **Constrói.**
  - `P(V>O) = Φ((q50V − q50O)/√(σV² + σO²))`, com σ = (q90 − q10)/(2 × 1,2816).
  - `suporte`: o valor dentro da faixa e pelo menos 20 posts parecidos no setor, olhando a combinação de features.
  - `efeito_bootstrap`: reamostrar canais inteiros.
  - O LLM propõe legendas com saída estruturada e o modelo julga.
- **No projeto Galaxies.** "Publicar às 19h com gancho forte" melhora A, mas ele continua atrás de B. Mudar a legenda
  não teve efeito detectável.
- **Erros comuns.**
  1. Apresentar o Δ como causal.
  2. Mexer em features que a marca não controla.
  3. Ignorar o suporte.
  4. Checar as colunas isoladamente.
  5. Usar P(V>O) como teste de significância.
  6. Deixar o LLM julgar as próprias propostas.
- **Exercício.** `what_if_seguro`, que devolve a tabela mais o dicionário `descartadas`.
- **No Prévia.** A fórmula de P(V>O) é a mesma do `prob_supera` da Tool 5 (6.3).

## 7.3 LLM zero-shot vs. modelo treinado

50 min · avaliação + AI Eng · depois de 23/10

- **Ideia.** Um duelo justo exige:
  - os mesmos pares intra-canal, do mesmo período temporal;
  - a mesma pergunta (P(A>B));
  - a mesma informação de entrada;
  - a mesma régua, incluindo o desvio entre execuções, o custo e a latência;
  - teste posterior ao corte de conhecimento do LLM.
- **Constrói.**
  - Do lado do GBM, P(A>B) = Φ(Δŷ/(σ√2)), com σ medido na validação.
  - Do lado do LLM: saída estruturada; as duas ordens, `p = (p_AB + 1 − p_BA)/2`; k execuções.
  - Borda para o Spearman.
  - Filtro de contaminação: sem busca na web, sem URL nem ID do vídeo.
  - Stacking por regressão logística, ajustado na validação.
  - Bootstrap por canal (2.000 reamostras).
  - Troca para o LLM real com `init_chat_model("google_genai:gemini-3.5-flash-lite")` e
    `with_structured_output(include_raw=True)`.
- **No projeto Galaxies.** A opinião do LLM entra como feature de um combinador validado, ou como "segunda opinião"
  rotulada. Nunca entra no lugar do P(B>A) do modelo.
- **Erros comuns.**
  1. Comparar com o número de um paper.
  2. Rodar o LLM uma vez só.
  3. Usar Shorts antigos e famosos.
  4. Perguntar numa ordem só.
  5. Dar informações diferentes a cada lado sem declarar.
  6. Ajustar o prompt olhando o teste.
  7. Declarar vencedor sem IC por canal.
  8. Ler a probabilidade do LLM como calibrada.
- **Liga com.** EDD 4: viés de posição e a técnica de inverter a ordem.

## 7.4 Embeddings de vídeo e áudio

45 min · multimodal · depois de 23/10

- **Ideia.** A média dos frames ignora ordem, ritmo e gancho. As features temporais baratas (ritmo, maior corte,
  embedding do gancho) vêm primeiro. A camada 3 só entra depois da camada 2 estável, e com hipótese.
- **Modelos.**
  - X-CLIP (MIT, 8 frames): quase não codifica a ordem (1 − cos ≈ 4×10⁻⁶).
  - VideoMAE: não comercial.
  - Qwen3-VL-Embedding-2B: exige GPU.
  - CLAP (`laion/clap-htsat-unfused`, 48 kHz, janelas de 10 s): corte as janelas você mesmo para evitar o
    `rand_trunc`. Sondas de texto viram colunas interpretáveis.
- **Constrói.** PyAV; a função `quadrado` contra o center crop (que corta ~22% em cima e embaixo, onde fica o texto);
  cache versionado; cerca de 2,8 s por Short em CPU.
- **Lição de PCA.** Ele guarda o que mais varia, não o que importa: com 8 componentes o modelo piorou −0,131, e com 16
  ganhou +0,029, com IC cruzando o zero.
- **Erros comuns.**
  1. Confiar no nome do modelo.
  2. Deixar o center crop cortar o vídeo vertical.
  3. Aceitar o `rand_trunc`.
  4. Jogar 512 colunas cruas no modelo.
  5. Ajustar o PCA na base inteira ou não versioná-lo.
  6. Não versionar a extração.
  7. Ignorar a licença.
  8. Começar por aqui.

## 7.5 Conformal adaptativo e drift

45 min · ML · depois de 23/10

- **Ideia.** O split conformal depende de permutabilidade. Com drift, a cobertura cai em silêncio: de 80% para 54% na
  simulação.
- **ACI.** `α_{t+1} = α_t + γ(α − err_t)`. A garantia é sobre a média de longo prazo, e o α_t ≤ 0 dá intervalo
  infinito, que vira baixa confiança. Com o atraso de 30 dias da maturação, use γ pequeno (0,002 a 0,005); um γ
  grande oscila.
- **MAPIE.** `TimeSeriesRegressor(method="aci", cv="prefit")` com `adapt_conformal_inference` e
  `allow_infinite_bounds=True`; EnbPI com `BlockBootstrap`.
- **Receita para o projeto.** Retreino periódico com recalibração numa janela recente e madura (posts de 60 a 30 dias
  atrás). O ACI fica como monitor semanal: a tool só lê o estado dele. Avalie sempre com o atraso.
- **No projeto Galaxies.** Uma mudança de nível afeta pouco a ordem entre A e B; uma mudança de relação exige
  retreinar.
- **Erros comuns.**
  1. Calibrar uma vez e nunca mais olhar a cobertura.
  2. Avaliar o ACI com rótulo imediato.
  3. Usar γ grande.
  4. Ler a garantia como condicional.
  5. Usar o ACI para esconder um modelo velho.
  6. Misturar posts imaturos na calibração.
- **No Prévia.** A "janela recente e madura" serve de referência para a data de calibração (M6).
