# Módulo 5: Incerteza e explicabilidade (7 aulas, 310 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/5-incerteza-e-explicabilidade/`.

O módulo segue duas linhas que se encontram na 5.7:

- **Incerteza (5.1 a 5.4):** mediana, intervalo de 80% e flag; quantis; calibração com CQR; flag de baixa confiança.
- **Explicação (5.5 a 5.7):** Shapley; TreeSHAP e `top_drivers`; narrativa ancorada com validador.

**Entrega ao contrato:** `interval_low` e `interval_high` calibrados, `confidence_flag`, `top_drivers`, `explanation`
e P(B>A). Entrega ao README: as tabelas de cobertura, largura e MAE.

**Ambiente do módulo.** numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, catboost 1.2.10,
shap 0.52.0 (exige Python ≥ 3.12), mapie 1.5.0, pydantic 2.13.5, langchain 1.4.2, langchain-core 1.6.5 e matplotlib
3.11.2. Na 5.7: `langchain-google-genai` 4.4.0 e `GOOGLE_API_KEY`.

**Mínimo** (inferido dos sinais do material): 5.1 → 5.2 → 5.3 (CQR) → 5.4 (regras aplicáveis e validação) → 5.6
(`top_drivers`), mais o validador e o template da 5.7. A 5.5 é só leitura. O Mondrian, o ACI, o LLM explicador e o
juiz são extras.

## 5.1 Por que um número só não basta

40 min · ML · leitura rápida (corte de 15/10, #24) · prática: entra na M6

- **Conceitos.**
  - A previsão pontual é a mediana (q0,5); o intervalo de 80% é [q0,1; q0,9]. Os 80% são escolha de produto; o
    inegociável é declarar o nível e medir a cobertura.
  - IC não é IP: o intervalo de predição tem a largura mínima do ruído.
  - Incerteza aleatória vem da regressão quantílica; a largura total, do conformal; a epistêmica, do flag.
  - Aplique `np.expm1` aos três quantis. `predicted_metric` é a mediana.
- **Constrói.** `para_log` e `prob_b_maior` (Monte Carlo e analítico). Com os quantis de A e B do caso,
  P(B>A) = 0,938 com ρ = 0, a escolha conservadora.
- **No projeto Galaxies.** A: 1.480 (780–2.800), 0,64×. B: 4.380 (2.300–8.300), 1,90×. P(B>A) ≈ 94%.
- **Erros comuns.**
  1. Chamar de intervalo o erro-padrão da média.
  2. Converter a média do log com `exp`.
  3. Declarar 80% sem medir.
  4. Decidir pela sobreposição dos intervalos.
  5. Comunicar "± 3.000".
  6. Achar que o intervalo resolve a incerteza epistêmica.
- **Notas.**
  - Meça a cobertura por setor e por tamanho de canal.
  - Critério de Gneiting: nitidez sujeita à calibração.
  - Erro-padrão da cobertura com 200 Shorts: ≈ 2,8 pp.
- **Exercício.** O item 3 (P prevista × acerto real dos pares) entra na análise da M6.

## 5.2 Regressão quantílica e pinball loss

45 min · ML · leitura profunda · prática: **Vinicius escreve M6** (parte 1)

- **Conceitos.**
  - Pinball: com α = 0,9, errar por baixo custa 9×. O minimizador é o quantil.
  - Ruído heterocedástico gera largura adaptativa.
  - Corrija o *quantile crossing* ordenando.
  - Quatro checagens: fração abaixo de cada quantil, cobertura, largura e pinball.
- **Constrói.**
  - LightGBM com um modelo por quantil (`objective="quantile"`, 400 árvores, lr 0,03, 15 folhas,
    `min_child_samples=40`), depois `np.sort`. Cobertura de 76,4%.
  - CatBoost `MultiQuantile`. Canal com menos de 10 posts: cobertura de 64% e largura 1,56.
- **No projeto Galaxies.** `interval_low = expm1(q0,1)`, sempre depois de ordenar. Mostre no README um criativo de
  canal com 5 posts.
- **Erros comuns.**
  1. Treinar em curtidas cruas.
  2. Usar o mesmo `min_child_samples` para 0,5 e 0,9.
  3. Fazer early stopping pelo MSE (use `metric="quantile"` com o mesmo alpha).
  4. Publicar sem ordenar.
  5. Olhar só a cobertura total.
  6. Aceitar 76% sem declarar.
- **Notas.**
  - Mais de 1 a 2% de cruzamentos indica subregularização.
  - Cobertura sozinha é trapaceável: sempre com a largura.
  - Cobertura de 70 a 78% fora da amostra é comum; acima de 85% é suspeito.

## 5.3 Predição conformal

50 min · ML · leitura profunda · prática: **Vinicius escreve M6** (parte 2, o CQR à mão)

- **Conceitos.**
  - Escore de não conformidade; k = ⌈(n+1)(1−α)⌉; q̂ é o k-ésimo menor escore.
  - A garantia é marginal e só vale com permutabilidade entre a calibração e o futuro.
  - O tempo quebra a garantia: calibre na fatia mais recente antes do teste, nunca no treino.
  - CQR: escore = max(q_lo − y, y − q_hi); a banda é [q_lo − q̂; q_hi + q̂], e q̂ pode ser negativo.
  - Mondrian: um q̂ por grupo.
- **Constrói.**
  - `split_conformal` e `cqr_na_mao`.
  - MAPIE 1.5: `SplitConformalRegressor(confidence_level=0.8, prefit=True)` → `conformalize` → `predict_interval`;
    `ConformalizedQuantileRegressor([inferior, superior, mediana])`.
  - Cobertura: quantis crus 64,9%; split 77,0%; CQR 76,2%. Calibrar no treino dá 61,7%.
- **No projeto Galaxies.** A tool usa os 3 quantis mais o q̂ do CQR, calculado na fatia mais recente de posts maduros.
  O q̂ e a data da calibração são versionados com o modelo. No README, a tabela cru × split × CQR, total e por setor.
- **Erros comuns.**
  1. Usar `train_conformalize_test_split`, que é aleatório.
  2. Calibrar no treino.
  3. Calibrar com posts imaturos.
  4. Ler a garantia como condicional.
  5. Errar a ordem [inferior, superior, mediana].
  6. Trocar `confidence_level` por `alpha`.
  7. Calibrar uma vez e esquecer.
- **Notas.**
  - A calibração mais recente termina 30 dias antes de hoje.
  - Intervalo infinito é motivo automático de baixa confiança.
  - O exercício sugere calibrar com as 6 a 8 semanas antes do teste.

## 5.4 Sinal de baixa confiança

40 min · ML + contrato · leitura rápida (corte de 15/10, #24) · prática: o Vinicius decide as regras e os limiares; o Claude implementa

- **Conceitos.**
  - O intervalo calibrado na média não cobre a incerteza epistêmica.
  - Regras explícitas, com códigos estáveis.
  - Fora da distribuição por kNN.
  - A taxa de sinalização é decisão de produto.
- **Constrói.** As 6 regras:
  - `intervalo_largo`: largura acima do p80 da calibração;
  - `canal_novo`: menos de 10 posts;
  - `setor_desconhecido`;
  - `fora_da_distribuicao`: kNN com K = 10 e p99, num PCA de 8 a 32 dimensões;
  - `extracao_falhou`;
  - `data_fora_do_periodo`.

  `confidence_flag(...)` devolve `(bool, [{codigo, texto}])`. Os `LIMIARES` são calculados fora do teste e salvos em
  JSON versionado. Resultado da aula: 36% sinalizados, com MAE de 0,622 contra 0,371.
- **No projeto Galaxies.** A e B saem sem flag. O flag acenderia numa submarca com 4 posts ou com falha na extração.
- **Erros comuns.**
  1. Flag sem avaliação.
  2. Limiar calculado na hora.
  3. kNN no próprio treino.
  4. Distância em 768 dimensões sem padronizar.
  5. Sinalizar tudo.
  6. Esconder a previsão.
  7. Motivos só em texto livre.
- **Notas.**
  - Uma taxa de 15 a 35% é um bom ponto de partida.
  - Regra que não separa o erro sai ou vira aviso.
  - Mínimo útil: as regras 1, 2 e 6, mais a 4 e a 5 se houver embeddings ou extração.
  - Use o mesmo espaço de vizinhos dos comparáveis (6.4).
  - No Prévia, com só 2 setores no `Literal`, a regra 3 vira validação do contrato.

## 5.5 Shapley values: a intuição

40 min · ML · leitura rápida (corte de 15/10, #24) · prática: depois de 23/10 (o exercício leva 30 min)

- **Conceitos.**
  - Os jogadores são as features; v(∅) é o valor base; φ é a contribuição marginal média.
  - Axiomas: eficiência (a conta fecha), simetria, jogador nulo e aditividade.
  - "Remover" uma feature é trocá-la por um background; o modo intervencional é fiel ao modelo.
- **Constrói.** O exemplo G/T/H à mão, com `shapley_exato` igual ao `TreeExplainer`. O custo cresce como 2^p.
- **No projeto Galaxies.** O "+0,28" do gancho de B significa ×1,33 contra o Short típico. Não quer dizer "sem o
  gancho, B teria 1/1,33". A e B, com o mesmo background, se subtraem feature a feature.
- **Erros comuns.**
  1. Explicar sem dizer a referência.
  2. Ler Shapley como "o que acontece se eu remover a feature".
  3. Esquecer as interações.
  4. Tratar o crédito dividido entre features correlacionadas como fraqueza.
  5. Comparar backgrounds diferentes.
  6. Usar mais de 100 linhas de background (o SHAP 0.52 subamostra).

## 5.6 SHAP na prática

50 min · ML · leitura profunda · prática: **Vinicius escreve M8**

- **Conceitos.**
  - TreeSHAP é exato. Tem dois modos: `tree_path_dependent` e `interventional` (`data=fundo`).
  - Os SHAP somam em log e multiplicam em curtidas (e^φ); porcentagens não somam.
  - Com o background do próprio canal, as features do canal viram 0 e sobram as diferenças de conteúdo.
  - SHAP explica o modelo, não a causa.
- **Constrói.**
  - `assert` de aditividade.
  - `agrupar_blocos` e `top_drivers(shap_row, feature_names, k=5, min_abs=0.01)`, que devolve `feature`, `valor`,
    `contribuicao_log`, `multiplicador` e `direcao`.
  - Alternativa nativa: LightGBM `predict(pred_contrib=True)`.
- **No projeto Galaxies.** Os `top_drivers` de B: gancho ×1,33, energia ×1,11, rosto ×1,10, texto ×1,08 e hora ×1,06;
  o total dá ×1,90. O JSON vai ao explicador sem nenhum número a mais.
- **Erros comuns.**
  1. Publicar SHAP em log.
  2. Somar porcentagens.
  3. Usar o background da base inteira.
  4. Comparar backgrounds diferentes.
  5. Listar componentes do PCA (agrupe em blocos).
  6. Ler SHAP como causa.
  7. Não testar a aditividade.
- **Notas.**
  - Registre o background no payload.
  - O background respeita o tempo.
  - Com menos de 10 posts, a explicação é fraca.
- **No Prévia.** M8: conversão em multiplicador, agrupamento e o `assert`. O Claude calcula o SHAP. Decisão sobre o
  background: [conflitos](conflitos.md).

## 5.7 Da explicação à narrativa

45 min · AI Eng · leitura profunda · prática: **Vinicius escreve A4**

- **Conceitos.**
  - A pior alucinação é "uma explicação plausível para um número verdadeiro".
  - O payload é fechado: nada do vetor SHAP inteiro, de valores em log, da transcrição ou do público.
  - O validador determinístico roda em toda chamada; o juiz roda offline.
  - O explicador fala do modelo; a persona da Galaxies fala do espectador.
- **Constrói.**
  - `montar_payload`.
  - Pydantic `Fator` e `Explicacao` (resumo, 2 a 5 fatores, ressalvas).
  - 7 regras de prompt: só fatos do JSON; duas conversões permitidas; nomes exatos; linguagem de marketing; sempre
    o intervalo; "Baixa confiança:" na primeira ressalva; associação, não causa. Temperatura 0.
  - O validador: `_para_float` (pt-BR), `numeros_permitidos` e `validar`.
  - `explicar(tentativas=2)`, com uma correção e falha fechada no template.
  - O juiz (`fidelidade`) usa um modelo diferente do explicador, num golden set de 20 a 40 payloads, calibrado com
    cerca de 20 rótulos humanos.
- **No projeto Galaxies.** O campo `explanation` nasce do mesmo payload: "nenhum número da explicação nasce fora da
  tool". No README entram a taxa de aprovação na primeira tentativa, a fidelidade média e um exemplo reprovado.
- **Erros comuns.**
  1. Dar contexto demais.
  2. Deixar o LLM calcular.
  3. Confiar na temperatura 0.
  4. Afrouxar o validador (ponha o fato no payload).
  5. Usar o mesmo modelo como explicador e juiz.
  6. Misturar explicação e persona.
  7. Esconder o intervalo e o flag.
- **Lacuna.** Prompt injection não é tratada; só há payload fechado, validador e saída estruturada. No Prévia
  (princípio 8), a A4 inclui o caso "legenda com instruções".
- **Liga com.** EDD 4 (juiz e G-Eval), EDD 5 (fidelidade como na tríade do RAG) e Conceitos de LLM (alucinação).
- **No Prévia.** A4. O LLM é o Gemini; no free tier, só com Shorts publicados. Nos testes, um LLM falso.
