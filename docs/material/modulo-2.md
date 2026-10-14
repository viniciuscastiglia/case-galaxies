# Módulo 2: Avaliação honesta (6 aulas, 305 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/2-avaliacao-honesta/`.

O encadeamento do módulo:

- **2.1:** a régua (baselines);
- **2.2:** features sem vazamento;
- **2.3:** o protocolo de validação;
- **2.4:** o painel de métricas;
- **2.5:** experimentos rastreáveis;
- **2.6:** a interpretação, com análise de erros e ablação.

O módulo inteiro usa `gerar_base` (sintética). No Prévia, trocar pelo loader da base real
(`previa.dados.carregar_base`), atentando para os nomes de coluna. Ambiente do módulo: lightgbm 4.7.0, mlflow
3.16.1, pyyaml 6.0.3 e os pins do módulo 1.

## 2.1 Baselines

45 min · ML · leitura feita · prática: feita na sintética; na base real, Claude (Fase 1)

- **Conceitos.**
  - O baseline calibra o "bom", detecta vazamento e vira o fallback em produção.
  - Candidatos: mediana ou média do canal, média dos últimos k, regressão em log(inscritos), mediana global.
  - Cold start em cascata: com 3+ posts anteriores, usa o canal; senão a regressão (ajustada no treino); senão a
    mediana global.
  - Com o alvo relativo, o baseline é prever 0, o que dá 50% par-a-par.
  - 5 regras de reporte: comparar com o mais forte; nas mesmas linhas; em absoluto e em relativo (skill score); com
    várias métricas; com IC.
- **Constrói.** `gerar_base(n_canais=60, seed=42)`, features com `shift(1).expanding()`, `spearman`, `spearman_intra`,
  corte em 01/03/2026 e a tabela abaixo.

  | Modelo | MAE em log |
  | :--- | :--- |
  | Mediana do canal | 0,792 |
  | Últimos 5 | 0,728 |
  | Modelo | 0,722 |

  No intra-canal, o Spearman vai de −0,13 a +0,16.
- **No projeto Galaxies.** Dizer 2.300 curtidas para A e para B empata os dois. Há duas barras a passar: errar menos
  que "2.300 para qualquer Short" e acertar mais de 50% dos pares.
- **Erros comuns.**
  1. Calcular a média com todos os posts.
  2. Comparar com o baseline mais fraco.
  3. Avaliar em linhas diferentes.
  4. Escolher o k olhando o teste.
  5. Usar baseline em escala linear.
  6. Reportar só MAE ou só Spearman global.
- **Notas.**
  - Inscritos de hoje são vazamento brando.
  - O k é hiperparâmetro e se escolhe na validação.
  - Reporte quantas linhas caem em cada degrau do fallback.
  - Os baselines de histórico ficam negativos no intra-canal por regressão à média.
- **No Prévia.** Exercício feito (`estudos/aula_2_1_baselines.py`, k=10). A tabela na base real é do Claude, na
  Fase 1, com o split definitivo e a mediana madura.

## 2.2 Vazamento de dados

50 min · ML · leitura feita · prática: feita

- **Conceitos.**
  - 5 famílias: alvo, temporal, grupo, pré-processamento e duplicatas.
  - 4 detectores: métrica boa demais, importância dominante, remover e ver, e auditoria com teste de perturbação.
  - O split temporal não corrige uma feature já calculada com o futuro.
- **Constrói.**
  - 5 cenários. O honesto dá Spearman 0,845, MAE 0,788 e intra 0,226; com comentários, o Spearman sobe a 0,983.
  - `mediana_madura(grupo, dias=30)` com `np.searchsorted`. O preço da honestidade é MAE +0,019.
- **No projeto Galaxies.** A auditoria de A/B. A mediana de 90 dias inclui posts imaturos, então treino e produção
  calculam cada feature do mesmo jeito.
- **Erros comuns.**
  1. "É só uma proporção".
  2. Target encoding na base inteira.
  3. Esquecer o `shift(1)`.
  4. "O PCA não vê o alvo".
  5. Confiar no split temporal para tudo.
  6. Deduplicar depois do split.
  7. Tratar métrica alta como vitória: Spearman acima de 0,9 exige explicação.
- **Notas.**
  - Scaler, PCA, TF-IDF e encoders se ajustam só no treino, dentro de `Pipeline`.
  - Inscritos: coletar periodicamente, ou documentar e testar a sensibilidade.
  - O prompt do VLM nunca recebe views.
  - Duplicatas: detectar por título e duração, hash perceptual ou embeddings, e manter as cópias do mesmo lado.
- **No Prévia.** Feito (`docs/AUDITORIA_FEATURES.md`, `estudos/aula_2_2_vazamento.py`). **Pendente:** a regra das
  duplicatas (decisão do Vinicius, Fase 1). Liga com EDD 2 (contaminação de golden datasets).

## 2.3 Validação temporal e por canal

50 min · ML · leitura profunda · prática: **Vinicius escreve M1** (Fase 1)

- **Conceitos.**
  - KFold aleatório é otimista.
  - Gap de 30 dias, pela maturidade do alvo e pelo contágio na fronteira (purging/embargo).
  - Walk-forward expansivo ou deslizante: reportar a média e cada fold.
  - `GroupKFold` para canal nunca visto.
  - "Canal novo no futuro" combina grupo e tempo.
  - Calibração por data: treino → gap → calibração → teste.
- **Constrói.** `splits.py` com `split_temporal(df, data_corte, gap_dias)`,
  `walk_forward(n_folds=4, gap_dias=30, inicio_teste)`, `split_por_canal(frac=0.25)` e `split_treino_cal_teste`.

  | Esquema | MAE | Spearman | Intra |
  | :--- | :--- | :--- | :--- |
  | KFold | 0,680 | 0,860 | 0,268 |
  | Walk-forward | 0,760 | 0,845 | 0,167 |
  | Canal novo | 0,774 | 0,830 | 0,219 |
  | Canal novo no futuro | 0,945 | 0,785 | 0,117 |

- **No projeto Galaxies.** Se a Lumina estiver na base, walk-forward com gap. Se não estiver (o provável), "canal
  novo no futuro". A confiança comunicada vem do cenário mais duro.
- **Erros comuns.**
  1. Reportar KFold aleatório.
  2. Usar `TimeSeriesSplit` num painel de canais.
  3. Esquecer o gap.
  4. Ajustar hiperparâmetros olhando o teste.
  5. Sortear a calibração.
  6. Usar um único fold curto.
  7. Fazer split por canal quando a pergunta é temporal, ou o contrário.
- **Notas.**
  - KFold aleatório só serve para depurar e nunca vai ao README.
  - O teste deve ser recente, de 3 a 6 meses, e anterior ao último mês.
  - Queda nos folds recentes é drift e vai ao README.
  - O `train_conformalize_test_split` do MAPIE sorteia: separe por data à mão.
- **No Prévia.** M1: o Vinicius escreve `split_treino_cal_teste` (cortes por data com gap) e `canal_novo_no_futuro`;
  o Claude porta `split_temporal`, `walk_forward` e `split_por_canal`, com os testes. Entram também as decisões da
  Fase 1: datas, gap, calibração e canais fora do treino. A trava atual (`configs/avaliacao.yaml`, 01/04/2026) é provisória. O exercício
  (`walk_forward_deslizante`) fica para depois de 23/10.

## 2.4 Métricas de erro e de ordenação

55 min · ML · leitura profunda · prática: **Vinicius escreve M2**; o Claude porta `metricas.py`

- **Conceitos.**
  - MAE em log é um fator: 0,766 vira ×2,15.
  - RMSE destaca os virais.
  - MAPE explode perto de zero e não se compara com o do SMP.
  - Spearman global mede sobretudo o tamanho do canal. O intra-canal é ponderado por n_c, com 3+ posts.
  - Par-a-par intra-canal é a métrica da decisão: PA = (τ+1)/2, e 60% equivale a τ = 0,2.
  - F1 nas faixas e cobertura.
  - IC por bootstrap de canais: 2.000 reamostras, percentis 2,5 e 97,5.
- **Constrói.**
  - `metricas.py`: `mae_log`, `rmse_log`, `mape`, `spearman_global`, `_pares`, `por_canal`, `spearman_intra_canal`,
    `pairwise_accuracy`, `faixa` e `f1_faixas`, `cobertura`, mais 6 testes.
  - `ic_bootstrap_canal`.
  - Faixas fixas: <−0,5; −0,5 a 0,5; 0,5 a 1,5; ≥1,5.
  - Convenções: o MAPE ignora y=0; Spearman de série constante vale 0; empate vale 0,5.
- **No projeto Galaxies.** O par-a-par sustenta a recomendação: 57% [55; 60]. O MAE dá a escala do erro, mas não é
  intervalo calibrado. Extensão: par-a-par por faixa de |ŷA−ŷB|, para separar "confiança alta" de "praticamente
  empate".
- **Erros comuns.**
  1. Spearman global como manchete.
  2. MAPE em curtidas cruas.
  3. Comparar o seu MAPE com o do SMP.
  4. Bootstrap por post.
  5. Intra-canal com canais de 1 ou 2 posts.
  6. Esquecer a convenção de empates.
- **No Prévia.** M2 é o exercício `pairwise_janela` (14 dias), na base real e escrito pelo Vinicius; o Claude porta o
  resto do `metricas.py`. Na sintética, o baseline dá 0,347,
  abaixo da moeda, por regressão à média. Há também a decisão sobre os 3 canais que dominam os pares: 61% dos pares,
  79% na janela de 14 dias.

## 2.5 Experimentos rastreáveis com MLflow

50 min · AI Eng (MLOps) · leitura profunda · prática: Claude monta o esqueleto; **o Vinicius escreve
`HIPOTESES.md`**

- **Conceitos.**
  - Experimento = hipótese + configuração + commit + versão dos dados + resultado.
  - Nenhum número vai ao README sem `run_id`.
  - `HIPOTESES.md` tem data, hipótese, mecanismo, métrica e limiar, controle, tratamento, resultado e decisão.
  - Seeds: o LightGBM é determinístico; com threads, `deterministic=True` mais `force_row_wise`.
  - O hash da base é a "seed" mais esquecida.
- **Constrói.**
  - YAML + dataclass `Config`, `commit_atual` e `hash_base` (sha256 de `hash_pandas_object`, 12 caracteres).
  - `treinar.py`: tags `git_commit`, `versao_base`, `hash_base`, `data_corte` e `config`, mais `log_params`,
    `log_metrics` e `mlflow.lightgbm.log_model(name=...)`.
  - `comparar.py` (`search_runs`).
  - `test_splits.py` com teste de perturbação do alvo.
- **No projeto Galaxies.** A resposta do agente carrega o endereço do modelo: URI, run, commit, corte e base. A tool
  recebe a URI por configuração.
- **Erros comuns.**
  1. Registrar métricas sem a configuração.
  2. Recoletar a base sem versionar.
  3. Comparar runs com cortes diferentes.
  4. Escolher o melhor de 40 configurações pelo teste.
  5. Usar `artifact_path=`.
  6. Deixar a porta 5000 ocupada no macOS (use `--port 5001`).
  7. Ter segfault ao carregar o modelo (resolve com `import lightgbm` no topo).
  8. Commitar `mlflow.db` e `mlruns/`.
- **Notas.** Teste só o que produz os números do README: métricas, splits e features de histórico. Commit antes de
  rodar o que vai ao relatório.
- **No Prévia.** O Claude monta o esqueleto na Fase 1; a hipótese, a métrica e o limiar são escritos pelo Vinicius
  antes de rodar. O exercício (H2, horário) fica para depois.

## 2.6 Análise de erros e ablação

55 min · ML · leitura profunda · prática: **Vinicius escreve M9** (Fase 4)

- **Conceitos.**
  - Resíduo: e = y − ŷ.
  - MAE e viés por corte; o erro-padrão é ≈ s/√n.
  - Cortes geram hipóteses: com mais de 15 grupos, algum parece estranho por acaso.
  - Os 20 piores erros, assistidos e classificados: viral genuíno, canal novo, problema de dado, alvo imaturo ou
    impulsionamento.
  - Ablação por blocos: cumulativa, deixa-um-de-fora e controle negativo de ruído.
  - Bootstrap pareado por canal.
- **Constrói.** `BLOCOS` + `ruido_1`/`ruido_2`, `bootstrap_pareado(n_boot=2000)` e a tabela cumulativa de par-a-par
  (canal 0,472 → +VLM 0,550; ruído 0,479). Os ICs no Spearman intra:

  | Bloco | Ganho | IC |
  | :--- | :--- | :--- |
  | Ruído | +0,000 | [−0,057; +0,058] |
  | Visual | +0,052 | [+0,001; +0,108] |
  | VLM | +0,078 | [+0,028; +0,132] |

- **No projeto Galaxies.** A ablação dá lastro às razões do agente: o gancho tem evidência; texto e rosto, evidência
  moderada; o horário não se distinguiu de zero, então não vira argumento principal. O SHAP diz o que pesou nesta
  previsão; a ablação diz se aquilo ajuda em geral. Só vale a explicação em que as duas concordam.
- **Erros comuns.**
  1. Fazer só a ablação cumulativa.
  2. Não ter controle negativo.
  3. Comparar ICs separados.
  4. Reamostrar posts em vez de canais.
  5. Declarar vitória com +0,01.
  6. Caçar cortes.
  7. Ler os piores erros só em números.
- **Notas.** "Não detectado" não é "não existe". Ganhos de +0,02 a +0,04 exigem teste pareado, vários folds e várias
  seeds.
- **No Prévia.** M9: o bootstrap pareado por canal e os 20 piores erros assistidos, sobre os blocos reais. O exercício
  das 5 seeds fica para depois de 23/10.
