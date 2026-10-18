# Módulo 3: Gradient boosting (6 aulas, 300 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/3-gradient-boosting/`.

As aulas sobem uma escada:

- **3.1:** a árvore e a instabilidade dela;
- **3.2:** os ensembles, e o gargalo passa a ser informação;
- **3.3:** o boosting por dentro (resíduo = gradiente; Huber e quantil; early stopping temporal);
- **3.4:** a receita de treino;
- **3.5 e 3.6:** a informação que faltava.

As aulas 3.1 a 3.4 usam um gerador com colunas de conteúdo que a base real não tem; 3.5 e 3.6 se aplicam quase
inteiras à base real.

**Ambiente do módulo.** pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, catboost
1.2.10, xgboost 3.4.1, optuna 5.0.0, sentence-transformers 6.1.0 e torch 2.14.0.

**Mínimo para o Prévia.** A 3.4, a 3.5, a 3.6 e a regra de early stopping temporal da 3.3. A 3.1 e a 3.2 são teoria.
O próprio material trata como dispensáveis o GBDoZero, a comparação das três bibliotecas, o Optuna e o sin/cos.

## 3.1 Árvores de decisão por dentro

45 min · ML · leitura rápida (corte de 15/10, #24) · prática: depois de 23/10

- **Conceitos.**
  - Split guloso por ganho de SSE; a folha é a média do alvo.
  - Controles de tamanho: `max_depth` (3 a 8), `min_samples_leaf`, `ccp_alpha`.
  - Pontos fortes: degraus, interações, invariância a transformações monotônicas e NaN nativo.
  - Pontos fracos: não extrapola e é instável (50 bootstraps: a raiz alterna; o desvio da previsão é ~36%).
- **Constrói.** `gerar_shorts(canais_por_setor=20, posts_por_canal=45, seed=7)`: 2.607 Shorts, efeitos conhecidos
  (gancho 0,09 por ponto, texto 0,15, horário 18–22h 0,12...). `sinal_verdadeiro` nunca é feature.
- **No projeto Galaxies.** A diferença verdadeira B−A é 1,09 em log. As árvores acertam a direção, mas variam de 0 a
  1,33. A árvore serve para entender a base (`export_text`), não para comparar criativos.
- **Erros comuns.**
  1. Deixar a árvore crescer sem limite.
  2. Escolher hiperparâmetros olhando o teste.
  3. Confiar na importância de uma árvore só.
  4. Esperar extrapolação.
  5. Padronizar features para árvore.
  6. Usar colunas da simulação como feature.
- **Exercício.** Ganho da raiz à mão. Fica para depois de 23/10.

## 3.2 Bagging vs. boosting

40 min · ML · leitura rápida (corte de 15/10, #24) · prática: depois de 23/10

- **Conceitos.**
  - Viés² e variância medidos: árvore rasa 0,695/0,017; funda 0,365/0,426; RF 0,329/0,036; GB 0,310/0,028.
  - Var = ρσ² + (1−ρ)σ²/B, então a correlação entre as árvores é o piso.
  - Reduzir `max_features` piora quando uma feature domina.
  - Boosting: árvores rasas em sequência, com early stopping.
  - OOB é validação aleatória: promete 0,500 e o teste temporal entrega 0,593.
  - Quando falta informação, trocar de algoritmo rende centésimos; as features da 3.5 rendem décimos.
- **Constrói.** Walk-forward com 3 janelas de 4 meses. GB 0,505 × RF 0,513 × árvore 0,574.
- **No projeto Galaxies.** Os ensembles dizem B > A de forma consistente, mas subestimam a diferença: o que falta é o
  nível atual do canal.
- **Erros comuns.**
  1. Confiar em OOB ou em qualquer validação aleatória.
  2. Declarar vencedor por 0,01 numa janela só.
  3. Aumentar `n_estimators` sem early stopping.
  4. Reduzir `max_features` por hábito.
  5. Esperar que o algoritmo compense a falta de informação.
- **Notas.** O RF com padrões é um bom segundo baseline: se o boosting não vence o RF, desconfie dele ou da
  validação.
- **Exercício.** Bootstrap pareado GB × RF, com IC que contém zero. Fica para depois; a ferramenta reaparece na M9.

## 3.3 Gradient boosting passo a passo

50 min · ML · leitura rápida (corte de 15/10, #24) · prática: depois de 23/10 (a regra entra na M4)

- **Conceitos.**
  - Os 5 passos de Friedman; o resíduo é o gradiente negativo.
  - Learning rate × número de árvores: fixe o lr e deixe o early stopping escolher.
  - Perdas: absoluta (mediana), Huber (corta em ±δ) e quantil (pinball; a folha é o quantil α dos resíduos).
  - O `validation_fraction` do sklearn embaralha: para tarde e erra mais.
- **Constrói.**
  - `GBDoZero`, idêntico ao sklearn.
  - Early stopping temporal (validação = 2º semestre de 2025): com lr 0,1, 185 árvores e MAE 0,502.
  - Regra: lr de 0,05 a 0,1 ao iterar e de 0,02 a 0,05 no final, depois refit em treino+validação.
  - `gb_generico`. Com 3% de virais, a Huber tem o menor MAE.
- **No projeto Galaxies.** "A escolha da perda decide o que o modelo diz sobre A e B." Com Huber, um viral no
  histórico pesa menos; os quantis 0,1 e 0,9 dão uma faixa.
- **Erros comuns.**
  1. Usar `validation_fraction` ou `n_iter_no_change` do sklearn num problema temporal.
  2. Otimizar lr e número de árvores como independentes.
  3. Usar árvores fundas no boosting (comece com profundidade 3–6 ou `num_leaves` 15–63).
  4. Esquecer que a perda define o que é previsto.
  5. Escolher a melhor iteração olhando o teste.
- **Notas.** Paciência de 50 a 200 rodadas.
- **Exercício.** Early stopping no GBDoZero. Fica para depois; a regra é usada na M4.

## 3.4 LightGBM, CatBoost e XGBoost na prática

60 min · ML · leitura profunda · prática: **Vinicius escreve M4**

- **Conceitos.**
  - LightGBM cresce folha a folha e é controlado por `num_leaves`; `bagging_fraction` exige `bagging_freq>0`.
  - XGBoost põe `early_stopping_rounds` no construtor.
  - CatBoost usa árvores simétricas, ordered target statistics e `has_time=True`.
  - Fluxo: early stopping na validação temporal, depois refit em treino+validação. Pular o refit custa 0,05 a 0,08
    de MAE.
  - Categóricas: `setor` sim; `canal_id` não. Em canais novos, o MAE vai de 0,704 a 0,932.
  - Huber no CatBoost: centralizar o alvo pela mediana.
  - Restrição monotônica só onde a direção é defensável (inscritos, número de posts).
- **Constrói.** Split com treino até 01/07/2025, validação até 01/01/2026 e teste em 2026.

  | Biblioteca | MAE |
  | :--- | :--- |
  | LightGBM | 0,590 |
  | CatBoost | 0,554 |
  | XGBoost | 0,587 |

  Pontos de partida da tabela: 3.000 a 5.000 árvores com ES; 7 a 63 folhas; mínimo de 20 a 100 por folha; L2 de 1 a
  10. O Optuna em walk-forward ganha só 0,009 no teste.
- **No projeto Galaxies.** B−A fica em 0,75 a 0,81, contra 1,09 verdadeiro: "o que falta é informação". Não usar
  `canal_id`.
- **Erros comuns.**
  1. Esquecer o refit.
  2. Limitar só `max_depth` no LightGBM.
  3. Passar `early_stopping_rounds` no `fit` do XGBoost 2 ou 3.
  4. Usar `canal_id` como categórica.
  5. Esquecer `has_time=True` ou desordenar as linhas.
  6. Usar Huber no CatBoost sem centralizar o alvo.
  7. Acreditar no MAE do Optuna.
- **Recomendação da aula.** CatBoost como principal e LightGBM para iterar: "escolha uma, fixe a validação e invista
  nas features". 3.5, 3.6, 4.8, 5.x e 6.x usam LightGBM (ver [conflitos](conflitos.md)).
- **No Prévia.** M4: o Vinicius escreve o treino (a chamada do LightGBM com early stopping na validação temporal, o
  refit em treino+validação e a comparação com o baseline); o Claude faz o runner em volta (config, dados, features,
  MLflow). A escolha entre
  LightGBM e CatBoost é decisão do Vinicius na Fase 2.

## 3.5 Features do canal e do tempo

55 min · ML · leitura profunda · prática: **Vinicius escreve M3** (o maior ganho do módulo)

- **Conceitos.**
  - Feature de canal só usa posts anteriores: ordenar por `published_at`, depois `groupby` + `shift(1)`.
  - `historico_do_canal(df, lacuna_dias)` com `np.searchsorted` trata empates e a lacuna de maturação.
  - Dois testes: `hist_ultimo_ts < published_at`, e a perturbação do futuro (embaralhar `y` depois de uma data não
    muda as features de antes).
  - Os inscritos são um snapshot e "sabem" do futuro.
  - Hora local crua basta para árvores.
- **Constrói.**
  - Colunas: `hist_n`, `hist_mediana`, `hist_media`, `hist_desvio`, `hist_med5`, `hist_med20`, `tendencia`
    (med5−med20), `dias_desde_ultimo`, `posts_30d`, `feriado`, `idade_canal_dias`, `hora_local` e
    `dia_semana_local`.
  - `feriados_br` e `merge_asof(direction="backward", allow_exact_matches=False, by=...)`.
  - Ablação: com o histórico, o MAE vai de 0,595 a 0,508, mas o **par-a-par intra-canal piora**, de 0,607 para 0,558.
    O ruído dos posts recentes muda as features de um post para o outro do mesmo canal. O alvo relativo recupera
    parte (0,571). O teto, de quem conhece o sinal verdadeiro, é 65,6%. Isso pesa no critério de sucesso (#5 do
    ROTEIRO).
- **No projeto Galaxies.** A e B compartilham o histórico; só o conteúdo e o horário mudam. Em produção, a tool de
  histórico calcula exatamente estas features, com a mesma função e a mesma lacuna.
- **Erros comuns.**
  1. `transform("mean")` sem `shift`.
  2. Não ordenar por `published_at`.
  3. Calcular as features separadamente no treino e no teste.
  4. Usar curtidas finais de posts recentes sem lacuna.
  5. Tirar a hora sem converter de UTC.
  6. Tratar o snapshot de inscritos como se fosse da época do post.
  7. Fazer `merge` comum em tabela com data.
- **Exercício.** "Quanto custa a lacuna?" (0, 7 e 30 dias): MAE 0,508, 0,508 e 0,514. O passo 4 mede a maturação na
  base real.
- **No Prévia.** M3: o corte "anteriores e maduros" e as agregações; os dois testes vêm prontos do Claude. A lacuna
  fica em 30 dias até a medição da maturação (19–22/10). É a "mediana madura" pendente desde 08/10.

## 3.6 Features de texto

50 min · ML + AI Eng · leitura rápida (corte de 15/10, #24) · prática: **o Vinicius decide** as features; o Claude implementa

- **Conceitos.**
  - Três caminhos: 9 features manuais (forma); TF-IDF+SVD (vocabulário do treino); embeddings+PCA (significado,
    inclusive paráfrases).
  - Tudo se ajusta só no treino, dentro de `Pipeline`/`ColumnTransformer`.
  - Cache de embeddings por `video_id` + hash do texto.
  - O efeito do texto é pequeno: o MAE quase não muda, e o what-if mostra se o efeito foi aprendido.
  - No SHAP, as dimensões de embedding só valem em grupo.
- **Constrói.**
  - `features_de_texto`, com 9 colunas `txt_*`: caracteres, palavras, hashtags, emojis, pergunta, CTA, números,
    maiúsculas e inglês.
  - `embeddings_com_cache` com `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões, trunca em 128 tokens).
  - TF-IDF (1,2) com `min_df=3` + SVD(16); PCA(16).
  - Na paráfrase, os embeddings acertam a direção e o TF-IDF fica perto de zero.
- **No projeto Galaxies.** O título soma pouco à vantagem de B. Para o texto do criativo do cliente, "prefira
  embeddings", com o efeito relatado em grupo.
- **Erros comuns.**
  1. Ajustar TF-IDF, SVD ou PCA na base inteira.
  2. Descartar o texto porque o MAE não mudou.
  3. Recalcular embeddings a cada treino.
  4. Usar a descrição padrão do canal, que vira identificador de canal.
  5. Interpretar uma dimensão isolada.
  6. Usar texto posterior (comentários, título editado).
  7. Ignorar o limite de 128 tokens.
- **Notas.**
  - Remova o bloco padrão da descrição.
  - No macOS, `pred_contrib` com torch e LightGBM no mesmo processo dá segfault; use o wrapper com `n_jobs=1`.
  - Comece com 16 a 32 componentes.
- **Liga com.** RAG 4 (embeddings), 4.4 (mesma regra do PCA) e o SHAP agrupado da 5.6.
- **No Prévia.** O Claude implementa o bloco `txt_` (Fase 2). O Vinicius decide a lista manual: CTAs e termos de
  beleza e automotivo, e se os embeddings entram. O exercício (número de componentes) fica para depois.
