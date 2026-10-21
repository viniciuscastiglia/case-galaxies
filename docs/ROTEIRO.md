# Roteiro da reta final (14 a 21/10/2026)

O plano vivo do projeto. O Claude lê este arquivo no início de cada sessão e o atualiza no fim (`/fechar-sessao`).
As decisões tomadas vão para `docs/DECISOES.md`; o histórico, para `docs/DIARIO.md`. O mapa das aulas está em
`docs/material/`.

## Agora

- **Fase (21/10, 18h):** dia da entrega (até 23h59; apresentação em 23/10). O lote da auditoria está no `main` e no
  GitHub (F1, F13, F15, F18, F22, F25, F26, F33, F35, F39, F40 e os registros F4, F38 e F43; 363 testes, ruff ok;
  o CI rodou pela primeira vez). Decididas hoje: #38 (corte 4, B), #32 (Huber testada: a H11 rodou e está lacrada,
  `bfba1020`), #33 (B: teste no modelo de janeiro, uso numa v2 retreinada), #34 (B: número com `canal_novo`), #35
  (A: transcrição e idioma fora, como limitação). As H2 a H9, a H11 e as ablações seguem **lacradas**.
- **Próximo passo (sessão nova):**
  1. as decisões #36 (F32, a leitura da ablação) e #37 (F23, o carregamento do modelo, aprovar ou vetar);
  2. a H10, dele; depois, a F2 (rosto e texto nas capas), do Claude, em segundo plano;
  3. a A1, com o caso à mão em `.superpowers/sdd/2026-10-20-a1/caso_a_mao.md` (com F19, F10, F34 e a cadeia da
     #34);
  4. o bloco 2 de decisões, com a F11;
  5. abrir as H2 a H9 e a H11 (com a F32) e escolher os blocos (F17). Na H7: `produto_beleza` tem AUC 0,814 só em
     beleza, e `roda` tem 0,670 e não passa;
  6. a M6 e o flag (#14, com a regra de data da #33);
  7. pelo corte #38: A2, A4, A7 e a pré-triagem da M9 pelo Claude; o teste final (F16) e depois a v2 da #33; README
     (as 3 seções dele) e autoavaliação.
- **Bloqueios:** a A1 trava o `prever_criativo`, o grafo (A6) e o MCP (A2); a abertura trava o modelo final, que
  trava a M6, o teste final e a v2. Se às 22h a v2 não estiver pronta, a entrega sai com o modelo de janeiro (#33).
  O Vinicius só leu até o módulo 2: cada peça abre com o conceito mínimo no chat (#29). Jobs longos como tarefas da
  sessão.

## Como funciona

- **Vinicius escreve** (M# de ML, A# de AI Engineering): o raciocínio do coração da peça. As peças são M1–M4, M6, M8, M9
  (os 20 piores erros), A1, A2, A4, A7 e A8; não existem M5, M7 nem A3 (a A3 entrou na A2). A5, A6 e o bootstrap da M9
  passaram para o Claude no corte de 15/10 (#24); o prompt da A4 e o teste de injeção da A7, no de 18/10 (#25); a M8, no
  de 20/10 (#28). Desde 19/10 (#26), sem esqueleto para preencher: o Claude traz um caso minúsculo com números, o
  Vinicius resolve à mão e escreve os passos em português, o Claude escreve o código a partir deles (o caso vira teste,
  ao lado dos testes-contrato da aula) e o Vinicius roda no mesmo caso e confere. Depois, o teach-back (`/teach-back`),
  antes de alguém construir em cima da peça.
- **o Vinicius decide:** em duas mensagens (desde 19/10). Primeiro, o Claude traz só o problema, com um exemplo
  concreto, e o Vinicius raciocina sobre ele; depois, 2 ou 3 opções, a recomendação com a aula e o custo de errar.
  Uma decisão por vez, em texto. A resposta vai para `docs/DECISOES.md`.
- **Claude:** encanamento, testes, integração e documentação de apoio. O Vinicius lê um resumo de 3 linhas.
- **Leitura (desde 18/10, #25; desde 20/10, #28, só nas aulas das peças do Vinicius):** dirigida. Antes de cada peça do
  Vinicius, só as seções que ela usa: o código, "No projeto Galaxies" e "Erros comuns" (~20 min por aula). Nas listas
  abaixo, "a fundo" agora quer dizer dirigida, e "rápida" vira um resumo de 3 linhas do Claude. A leitura inteira vai
  para a trilha depois de 23/10.
- **Ritmo:** cerca de 5,2 h por dia do Vinicius (#24), sem folga. No fim de cada dia, `/fechar-sessao` compara o
  feito com este roteiro. Se o dia estourar em mais de meio dia, entra o próximo corte.

## Fases

### Fase 0 · 14/10 de manhã · reestruturação

- [x] Material lido pelo Claude e mapeado (`docs/material/`).
- [x] CLAUDE.md, ROTEIRO, DECISOES, DIARIO, skills, settings e memória criados (13/10), com revisão independente.
- [x] Vinicius aprova o commit e o merge da branch `docs/reta-final` (14/10).
- [ ] Vinicius lê este roteiro (10 min).

### Fase 1 · 14 a 16/10 · avaliação honesta na base real

- **Leitura:**
  - [x] 2.3
  - [x] 2.4 (18/10)
  - [x] 2.5 (19/10, dirigida)
  - [ ] 2.6
  - [ ] RAG 1–8 (rápidas)
- **Vinicius decide** (tabela "Decisões pendentes", #1 a #8):
  - [x] duplicatas
  - [x] início do treino
  - [x] datas de teste, gap e calibração
  - [x] canais fora do treino
  - [x] critério de sucesso (#5) (19/10)
  - [x] peso dos canais dominantes (#6) (19/10)
  - [x] faixas (18/10)
  - [x] teto por canal (#8) (18/10)
  - [x] limiares de histórico (#20) (18/10)
- **Vinicius escreve:**
  - [x] M1 (17/10, teach-back em `estudos/TEACHBACKS.md`)
  - [x] M2 (19/10; código do Claude a partir do caso à mão, #26)
  - [ ] `HIPOTESES.md` (19/10: H1, H2 e H4; faltam H3, H5 a H9)
- **Claude:**
  - [x] `splits.py` portado da 2.3, com testes, e `folds_de_desenvolvimento` (17/10)
  - [x] `metricas.py` portado da 2.4, com testes (18/10), e a manchete da #6 (19/10)
  - [x] o esqueleto e os testes-contrato da M1
  - [x] o esqueleto e os testes-contrato da M2 (18/10)
  - [x] a contagem de Shorts por mês e por canal, para a decisão das datas
  - [x] a trava do teste ajustada às datas decididas (continua em 01/04/2026)
  - [x] a regra das duplicatas (#1), corrigida na revisão de 17/10 e estendida entre canais
  - [x] a base congelada (versão e hash) (19/10: v1, hash `09f6035c11c9`)
  - [x] o esqueleto de experimentos (YAML + dataclass, MLflow com commit, hash e corte, teste de perturbação) (19/10)
  - [x] `auditar_datas` (1.7) (19/10)
  - [ ] a tabela de baselines na base real, com a mediana madura e o fallback por degrau: a tabela e a manchete
    existem (19/10); os baselines mais fortes da 2.1 entraram em 21/10 (k = 10, manchete 0,501, empate com a
    mediana madura, 0,497); o fallback por degrau é a cadeia da #34, na A1
  - [x] a lista dos 50 Shorts para conferir (19/10)
- **Pronta quando:** existe a tabela de baselines na base real, com o split definitivo e o run id, e as hipóteses
  estão escritas antes de rodar.

### Fase 2 · 17 e 18/10 · GBM Opção A

- **Leitura:**
  - a fundo: [ ] 3.4, [ ] 3.5, [ ] 6.1
  - rápida: [ ] 3.1, [ ] 3.2, [ ] 3.3, [ ] 3.6
- **Vinicius decide:**
  - [x] features de texto (#9) (18/10)
  - [x] LightGBM × CatBoost (#10) (18/10)
  - [x] lacuna (#11) (18/10)
  - [ ] conferência dos 50 Shorts (#19)
  - [x] idade do rótulo (#21) (18/10)
- **Vinicius escreve:**
  - [x] M3 (19/10; código do Claude, #26; a parte do Vinicius é a H2 e a leitura)
  - [ ] M4
- **Claude:**
  - [x] o esqueleto e os testes da M3 (`hist_ultimo_ts < published_at` e perturbação do futuro) (19/10)
  - [x] o bloco `txt_` (manuais, cache de embeddings, PCA por fold no runner) (19/10)
  - [x] (19/10, com o treino da M4, #26) o runner em volta da M4: configuração, carga dos dados, montagem das features e registro no MLflow. O treino
    em si (early stopping, refit e comparação com o baseline) é a M4
  - [ ] a recoleta das estatísticas antes do treino final
  - [x] o início do download das thumbnails, em background (19/10; ~8 h, termina à noite)
- **Marco 1:** GBM × baseline em walk-forward e em "canal novo no futuro", com IC pareado por canal.

### Fase 3 · 18 e 19/10 · tool mínima

- **Leitura:**
  - a fundo: [ ] 6.5, [ ] EDD 1–8 (rápidas: 1, 7 e 8)
  - rápida: [ ] 6.2, [ ] 6.3, [ ] Function calling (revisão)
- **Vinicius decide:** [x] setores no contrato (#22) (18/10)
- **Vinicius escreve:**
  - [ ] A1
  - [ ] A2
- **Claude:**
  - [ ] `prever_criativo` como esqueleto andante (carrega o modelo uma vez, devolve `model_version`)
  - [x] as versões fixadas (`fastmcp==3.4.7`, `mcp==1.30.0`, langgraph, langchain) (19/10)
  - [x] a guarda do OpenMP (20/10: `previa.openmp.um_runtime`; no Mac M4, torch + LightGBM dão segfault em qualquer ordem de import sem `OMP_NUM_THREADS=1`)
  - [ ] os stubs das tools
- **Marco 2:** `prever_criativo` responde pelo MCP no Inspector e dentro do Claude Code (`claude mcp add`), com o
  número do GBM.

### Fase 4 · 19 e 20/10 · multimodal leve e incerteza

- **Leitura:**
  - a fundo: [ ] 4.8, [ ] 5.2, [ ] 5.3
  - rápida: [ ] 4.1, [ ] 4.2, [ ] 4.3, [ ] 4.4, [ ] 4.5, [ ] 4.6, [ ] 4.7, [ ] 5.1, [ ] 5.4
- **Vinicius decide:**
  - [x] recorte multimodal (#12) (18/10)
  - [x] rubrica do VLM (#13) (18/10)
  - [ ] limiares do flag (#14)
- **Vinicius escreve:**
  - [ ] M6
  - [ ] M9: assistir e classificar os 20 piores erros (a lista está pronta e lacrada: `data/erros/piores_2d1bb463.csv`, 20/10)
- **Claude:**
  - [x] thumbnails da base toda (SigLIP 2, sondas, OpenCV, PCA no treino) (20/10: 30.981 capas com embedding, 10 sondas e cor e brilho da faixa 9:16)
  - [x] amostra com frames, cortes e transcrição (apagando a mídia depois de extrair) (21/10, 0h15: 1.399 de 1.400; H9 e ablação da amostra rodadas e lacradas)
  - [x] rubrica do VLM na amostra (20/10: 1.400 Shorts, 0 erros; reteste em 50 com kappa de 0,72 a 1,00, os 6 campos acima de 0,6)
  - [x] bootstrap pareado por canal da M9 (corte #24; o Vinicius faz teach-back): o da ablação, nas duas réguas (20/10)
  - [x] fusão com prefixos (19/10, no runner)
  - [x] o runner da ablação (20/10: `previa.ablacao`, escada + todos menos um + ruído; base toda e amostra rodadas e lacradas)
  - [ ] o `confidence_flag` com as regras decididas
  - [x] a medição da maturação (19/10: crescimento mediano de 0% depois de 21 dias; `previa.maturacao`)
- **Pronta quando:** existe a tabela de ablação com IC e a cobertura do intervalo de 80% está medida, total e por
  setor.

### Fase 5 · 19 a 21/10 · explicação e agente

- **Leitura:**
  - a fundo: [ ] 5.6, [ ] 5.7, [ ] 6.6
  - rápida: [ ] 5.5, [ ] 6.4
- **Vinicius decide:**
  - [ ] LLM do explicador (#15)
  - [ ] background do SHAP (#16)
  - [ ] explicação pelo MCP (#17)
  - [ ] criativos da demo (#18)
  - [ ] payload do explicador (#23)
- **Vinicius escreve:**
  - [ ] A4: o validador
  - [ ] A7: a invariante
- **Claude:**
  - [ ] A5 e A6 (corte #24; o Vinicius faz teach-back de cada uma): A5 pronta em 19/10 (`previa.comparaveis`)
  - [ ] o prompt da A4 e o teste de injeção da A7 (corte #25; o Vinicius faz teach-back)
  - [ ] M8, o `top_drivers` (corte #28; o Vinicius faz teach-back)
  - [x] SHAP (`pred_contrib`, e o intervencional para o fundo da #16) (19/10)
  - [ ] payload e template da explicação
  - [ ] cliente do LLM e LLM falso
  - [ ] corpos das tools 1 a 5
  - [x] índice de comparáveis (19/10)
  - [ ] `Passos` do grafo
  - [x] golden set congelado (canais fora do treino) (19/10: 30 Shorts de 6 canais, `data/golden/`)
  - [ ] testes de contrato e de regressão
  - [x] CI no GitHub Actions (pytest e ruff, sem dados e sem LLM) (19/10; o primeiro run, em 21/10, passou em
    ~2 min 20 s no Ubuntu: run `39647277817`, commit `1b25813`)
- **Pronta quando:** o agente passa nos testes (invariante, injeção, golden set) e a explicação é validada.

### Fase 6 · 21/10 entrega (até 23h59) · 22 e 23/10 apresentação

- **Leitura:** [ ] 6.7 (rápida)
- **Vinicius escreve:**
  - [ ] A8: o README até 21/10; o ensaio cronometrado e o ensaio de perguntas em 22 e 23/10
  - [ ] autoavaliação do hub (até 21/10)
- **Claude:**
  - [x] rascunho do README a partir do `DECISOES.md` (corte #25; o Vinicius reescreve) (20/10: marcas `V:` e `PREENCHER:`)
  - [ ] tabelas do README
  - [ ] data card e model card finais
  - [ ] respostas da demo salvas em JSON (plano B)
  - [ ] verificação final

## Dia a dia (horas do Vinicius)

Replanejado em 18/10 pelo corte #25 e, para 20 e 21/10, em 20/10 pelo #28, peça por peça (as horas da tabela "Peças",
sem o que passou para o Claude). De 14 a 17/10 saíram a 2.2, a 2.3, a M1 e as decisões #1–#4 e #24; em 18/10, a 2.4 e a
#25. A leitura é dirigida (~20 min por aula, só as seções da peça); as rápidas são os resumos do Claude, lidos junto com
a peça.

| Dia | Leitura dirigida | Prática e decisões | ~h |
| :--- | :--- | :--- | :--- |
| Dom 18/10 | 2.4 (feita), 2.5 | decisão #25 (feita); #5 e `HIPOTESES.md`; M2; bloco 1 de decisões (#6–#13, #20–#22); conferir os 50 Shorts (#19); criar a `GOOGLE_API_KEY` no AI Studio e pôr no `.env` (5 min) | 4,3 |
| Seg 19/10 | 3.5, 3.4, 6.1 | M3; M4; leitura do marco 1; A1 | 5,7 |
| Ter 20/10 | 6.1, 5.2, 5.3 (3.4 e 3.5 viram resumo) | #28 (feita); H3 e H5 a H9, curtas; abrir a ablação e o marco 1; A1; bloco 2 de decisões (aprovar ou vetar); M6 | 5,7 |
| Qua 21/10 | o conceito mínimo de cada aula dentro da peça (#29) | o que sobrou de 20/10 (A1, bloco 2 de decisões, abrir H2 a H9 e o marco 1, M6) e as decisões da auditoria (#30: a #31 feita; #32 a #37) e a H10; a B da #28; M9 (20 piores erros); A2; A4 (validador); A7 (invariante); os rótulos do juiz da explicação (F8) e a conferência dos 50 Shorts (#19, F14); teach-back de A5, A6, M8, do bootstrap da M9, do prompt da A4 e da injeção da A7; README (reescrever o rascunho); autoavaliação; **entrega até 23h59** | ~12,5 |
| Qui 22/10 | de manhã, a leitura dirigida de 3.4, 3.5, 4.8, 5.2, 5.3, 5.6, 6.1, 6.2, 6.5 e 6.6, com as perguntas prováveis da banca (#29) | à tarde, ensaio cronometrado da demo e ensaio de perguntas | ~3,5 + ensaios |
| Sex 23/10 | — | **apresentação** | — |

Em 21/10, às 13h30, o resto do dia somava ~12,9 h dele para ~8,5 h úteis; o corte #38 (B) leva a ~8,3 h.

Com a #28 (A), 20 e 21/10 somam ~13,2 h para ~10,4 h: ~2,8 h de estouro. A B (a A2 para o Claude, ~1,8 h com
a 6.5) volta no fechamento de 20/10, com o dia medido, antes de a A2 começar.

## Peças que o Vinicius escreve

Os nomes de arquivo são provisórios; o plano de cada bloco os fixa.

| # | Peça | Aulas | O que é dele | Teste-contrato (escrito antes pelo Claude) | ~h |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M1 | splits por data (treino, gap, calibração, teste) e "canal novo no futuro" (`splits.py`) | 2.3 | as funções `split_treino_cal_teste` (cortes por data com gap, lidos do config) e `canal_novo_no_futuro` (grupo + tempo); o Claude porta `split_temporal`, `walk_forward` e `split_por_canal` | os blocos não se sobrepõem e respeitam o gap; nenhum canal "novo" aparece no treino; a trava do teste é respeitada | 1 |
| M2 | par-a-par intra-canal na janela de 14 dias, na base real (`metricas.py`) | 2.4 | a `pairwise_janela` e a leitura do resultado do baseline; o Claude porta o resto do `metricas.py` da 2.4 | casos pequenos com resposta conhecida: 2 canais, empate vale 0,5, pares fora da janela ignorados | 1 |
| — | `HIPOTESES.md` | 2.5 | hipótese, mecanismo, métrica, limiar e controle, antes de rodar | — | 0,5 |
| M3 | histórico do canal com posts anteriores e maduros (`features.py`) | 3.5, 2.2 | o corte no tempo e as agregações | `hist_ultimo_ts < published_at`; perturbar o futuro não muda as features; post com menos de 30 dias não entra | 2 |
| M4 | GBM Opção A com early stopping temporal e refit | 3.3, 3.4 | o treino: a chamada do LightGBM com early stopping na validação temporal, o refit em treino+validação com o melhor número de árvores e a comparação com o baseline; o Claude faz o runner em volta (config, dados, features, MLflow) | o early stopping usa só a validação; nenhuma linha do teste; a seed fixa reproduz a métrica | 1,5 |
| M6 | intervalo de 80%: 3 quantis + CQR calibrado por data | 5.2, 5.3 | os quantis, o CQR à mão e a cobertura por setor | q10 ≤ q50 ≤ q90 depois de ordenar; q̂ calculado só na calibração; em dados permutáveis, cobertura de 80% ± 3 pp | 2 |
| M9 | ablação: bootstrap pareado por canal e os 20 piores erros | 2.6, 4.8 | assistir e classificar os 20 piores; o bootstrap passou para o Claude (corte #24) | reamostra canais inteiros; modelos idênticos dão IC com zero | 1,5 |
| M8 | `top_drivers` a partir do SHAP | 5.6 | multiplicador, direção e agrupamento por bloco | aditividade (soma dos SHAP + base = previsão); multiplicador = e^φ | 1 |
| A1 | contrato Pydantic de `prever_criativo` (`schemas.py`) | 6.1 | Request e Response, validadores, `extra="forbid"` | recusa data sem fuso, setor fora, campo extra e data a mais de 90 dias; aceita o pedido válido; low ≤ pred ≤ high | 1 |
| A2 | servidor MCP + Inspector + `claude mcp add` (`servidor_mcp.py`) | 6.5 | o servidor, as descrições das tools e os erros legíveis | o cliente em memória lista as tools; o MCP devolve o mesmo que a função; pedido sem fuso vira erro legível | 1,5 |
| A4 | explicador com validador de fidelidade | 5.7, EDD 3–5 | o `validar`; o prompt (legenda como dado) passou para o Claude no corte #25, com teach-back | aprova a explicação fiel; reprova número inventado e falta de intervalo; a legenda com instruções não muda o número | 2 |
| A5 | comparáveis sem vazamento (`comparaveis.py`) | 6.4, RAG 4–6 | corte #24: o Claude escreve a busca kNN com os filtros aplicados antes; o Vinicius faz teach-back | nunca devolve post publicado depois da data de corte, nem o próprio post, nem outro setor | 0,75 |
| A6 | grafo LangGraph do fluxo da previsão (`agente.py`) | 6.2 | corte #24: o Claude escreve o `StateGraph` (estado, nós e aresta condicional); o Vinicius faz teach-back | com `Passos` falsos, texto pula a extração e vídeo passa por ela; o número final é o da função | 1,5 |
| A7 | testes do agente | 6.6, EDD 2 e 6 | a invariante "número do agente = número do modelo"; o teste de injeção na legenda passou para o Claude no corte #25, com teach-back | (os testes são a peça) | 0,5 |
| A8 | README e demo | 6.7 | problema, metodologia, resultados e limitações, reescritos a partir do rascunho do Claude (#25); ensaio cronometrado; ensaio de perguntas | — | 1 até 21/10; o ensaio em 22 e 23/10 |

## Decisões pendentes

O Claude traz cada uma na hora certa, com contexto e opções. A recomendação abaixo é a de partida.

| # | Decisão | Aula | Fase | Recomendação do Claude | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Regra das duplicatas (`duplicate_group`: 668 grupos e 1.125 cópias extras em 14/10; só 262 com a mesma thumbnail) | 2.2, 2.3 | 1 | título + duração + thumbnail (dHash ≤ 10); a primeira publicada nas linhas de treino e teste, todas no histórico; 10 pares da Principia conferidos à mão | **decidida 14/10 (A); 17/10: também entre canais** |
| 2 | Início do treino (336 rotulados antes de jul/2021 em 15/10: vídeos verticais curtos, ainda não Shorts) | 2.3 | 1 | a partir de 01/07/2021 | **decidida 15/10 (01/07/2021)** |
| 3 | Datas de teste, gap e calibração | 2.3, 5.3, 7.5 | 1 | treino até 01/01/2026, gap de 30 dias, calibração fev–mar/2026, teste 01/04 a 14/09/2026 | **decidida 15/10 (A)** |
| 4 | Canais fora do treino, para o golden set e a demo | 2.3, 6.6, 6.7 | 1 | reservas revisadas, fora da base: 2 marcas + 1 criador por setor, sorteio dentro do estrato, 5 Shorts maduros do teste por canal | **decidida 15/10 (B)** |
| 5 | Critério de sucesso (também vai para o README) | 2.4, 2.5 | 1 | par-a-par intra-canal do GBM acima do baseline, com IC pareado por canal sem o zero; os ~60% como referência; incluir "canal novo no futuro" e a cobertura do intervalo | **decidida 19/10 (B): ganho ≥ +3 pontos com IC sem o zero, e IC sem o zero no canal novo** |
| 6 | Peso dos 3 canais que dominam os pares: Mobiauto, Mundo do Automóvel para PCD e Principia. Num teste desde 01/04 eles somam 61% dos pares (79% na janela de 14 dias), e os 3 não têm linha no treino; isso também pesa na #4 | 2.4 | 1 | manchete com média ponderada e, ao lado, a média com peso igual por canal; teto de pares por canal | **decidida 19/10 (B): peso pelo nº de Shorts do canal; Salon Line = 53% dos pares de 2025** |
| 7 | Convenção de faixas | 1.6, 2.4 | 1 | quantis do histórico do canal (1.6) | **decidida 18/10 (A)** |
| 8 | Teto por canal nas linhas de treino. Já decidido em 08/10: o teto é parâmetro do experimento e o histórico usa todos os posts. Resta decidir o valor (400?) e qual versão é a manchete | 1.4 | 1 | rodar com e sem teto; manchete sem teto, com o resultado com teto ao lado | **decidida 18/10 (A)** |
| 9 | Features de texto: a lista manual (CTA, pergunta, hashtags, números, termos do setor) e se os embeddings entram | 3.6 | 2 | manuais + embeddings multilíngues com PCA de 16, decididos pela ablação | **decidida 18/10 (A)** |
| 10 | LightGBM × CatBoost | 3.4 | 2 | LightGBM (ver conflitos #3); o CatBoost como experimento opcional | **decidida 18/10 (A)** |
| 11 | Lacuna do histórico | 3.5, 1.7 | 2 e 4 | 30 dias até a medição da maturação (19 e 20/10); rodar o experimento 0/7/30 da 3.5 | **decidida 18/10 (A)** |
| 12 | Recorte multimodal | 4.1, 4.2, 4.8 | 4 | thumbnails da base toda (SigLIP 2, sondas, OpenCV) + amostra estratificada de ~1.000 a 1.500 Shorts com frames, cortes e transcrição, por regra escrita antes | **decidida 18/10 (A)** |
| 13 | Rubrica do VLM | 4.7 | 4 | 5 a 8 campos com âncoras (gancho, rosto, produto em close, antes/depois, texto na tela, energia da música), via Gemini por URL na amostra; vem no bloco 1 (#24) | **decidida 18/10 (A)** |
| 14 | Regras e limiares do `confidence_flag`, inclusive a regra "fora da distribuição" (conflitos #12) | 5.4, 6.4 | 4 | regras 1, 2 e 6 (+4 com embeddings); "fora da distribuição" por cosseno ao vizinho mais próximo, no espaço dos comparáveis; limiares calculados na calibração; taxa entre 15 e 35% | aberta |
| 15 | LLM do explicador | 5.7 | 5 | Gemini (`gemini-3.5-flash-lite`) via `langchain-google-genai`; free tier só com Shorts publicados (princípio 5); a chave `GOOGLE_API_KEY` é criada pelo Vinicius e vai para o `.env`; testes com LLM falso | aberta |
| 16 | Background do SHAP | 5.6, 6.3 | 5 | o da 6.3, mais o `assert` de aditividade (ver conflitos #7) | aberta |
| 17 | Explicação pelo MCP | 6.5, 5.7 | 5 | template no servidor (sem LLM); explicador LLM só no grafo (ver conflitos #8) | aberta |
| 18 | Criativos da demo | 6.6, 6.7 | 5 | regra escrita antes, por exemplo o Short maduro mais recente de cada canal fora do treino, ao menos um por setor | aberta |
| 19 | Conferência de 50 classificações de Short (concordância vai para o data card) | 1.5 | 2 (até 16/10) | o Claude monta a lista; o Vinicius confere | aberta |
| 20 | Limiares de histórico: 30 Shorts para o canal entrar (1.4), 10 posts anteriores para o alvo relativo (1.6), 3 para o cold start (2.1) | 1.4, 1.6, 2.1 | 1 | manter os três, um em cada papel; `canal_novo` (5.4) com menos de 10 | **decidida 18/10 (A)** |
| 21 | Idade do rótulo (viés 3 do data card): mediana de 324 dias, 396 no treino e 52 no teste | 1.7, 3.5 | 2 | experimento na M3/M4: `target_age_days` como variável de controle, fixada em 30 na previsão, × treinar só com rótulos de idade parecida. Decidir pela validação temporal | **decidida 18/10 (A)** |
| 22 | Setores no contrato: `Literal` com os 2 setores, e setor fora vira erro de validação (conflitos #14) | 6.1 | 3 | aceitar a proposta, entra na A1 | **decidida 18/10 (A)** |
| 23 | Payload do explicador: o da 6.3, mais a checagem de ressalvas da 5.7 (conflitos #13) | 5.7, 6.3 | 5 | aceitar a proposta, entra na A4 | aberta |
| 24 | Corte pela entrega em 21/10: absorver ~10 h (o que estava em 22–23/10, menos o ensaio, e o atraso de 14/10) | ROTEIRO ("Se atrasar") | 1 | aplicar já os 6 itens de "Se atrasar" (~7,5 h; nada sai do produto); a M8 continua dele, e o resto (~2,5 h) sai de ~25 min a mais por dia | **decidida 15/10 (A)** |
| 25 | Corte 2: absorver ~11 h de atraso (18/10) | ROTEIRO ("Se atrasar"), 5.6 | 1 | leitura dirigida; prompt da A4 e injeção da A7 para o Claude; decisões em dois blocos (menos a #5); rascunho do README do Claude. A M8 fica dele, como fusível | **decidida 18/10 (A)** |
| 26 | Formato das peças "Vinicius escreve" (19/10): o esqueleto com `TODO(human)` não ensinava (a M2 travou sem contexto) | — | 1 | o raciocínio é dele (caso à mão + passos em português), o código é do Claude; treinar a sintaxe de pandas e numpy vai para depois de 23/10 | **decidida 19/10** |
| 27 | Régua dos blocos nas H2 a H6 (19/10): o histórico pode somar quase nada na ordem e muito no número | 2.4, 2.6 | 1 | duas réguas: fica se melhorar a manchete ou o `mae_log` (IC pareado sem o zero), sem piorar a outra; +3 pontos só na H1 | **decidida 19/10 (A)** |
| 28 | Corte 3 (19/10, noite): ~3,7 h de 19/10 ficaram para trás; 20 e 21/10 somam ~16 h de trabalho do Vinicius para ~10,4 h | ROTEIRO ("Se atrasar") | 1 | A: leitura dirigida só nas aulas das peças dele, M8 para o Claude, bloco 2 em aprovar ou vetar, H3 e H5 a H9 curtas; a B (a A2 para o Claude) volta no fechamento de 20/10, antes de a A2 começar | **decidida 20/10 (A; a B no fechamento)** |
| 29 | Leitura dos módulos 3 a 6 (21/10): ele não leu nada depois do módulo 2 | ROTEIRO, hub | 6 | A: o conceito mínimo dentro de cada peça hoje; a leitura dirigida das aulas da apresentação em 22/10 | **decidida 21/10 (A; se sobrar tempo, adianta parte de 22/10)** |
| 30 | Auditoria de conformidade e completude contra o TAPI, o hub e as aulas (21/10) | protocolo do material | 6 | fazer já, antes das peças do Claude; ele confere por amostragem | **decidida 21/10; feita** (`docs/AUDITORIA.md`: F1 a F43) |
| 31 | Camada 1 do multimodal (F2): o TAPI §6.3 a chama de obrigatória; hoje há SigLIP só na capa, e rosto, texto e produto só pela rubrica, na amostra | 4.6, 4.7, 4.1 | 6 | A: rosto (YuNet) e texto (PaddleOCR) nas 30.981 capas, sem objetos; concordância com o VLM; H10 escrita antes de rodar | **decidida 21/10 (A)** |
| 32 | Perda do GBM (F24): L2 em todas as configs, sem decisão registrada; Huber é o padrão das aulas | 1.2, 3.3, 3.4 | 6 | rodar Huber (δ = 1) como variante do completo, com hipótese escrita antes, e decidir pela régua #27 antes de abrir o teste | **decidida 21/10 (A); H11 rodada e lacrada (`bfba1020`)** |
| 33 | Base do treino final e regra de data do flag (F31 e F21): a `data_fora_do_periodo` dispararia em toda previsão de hoje | 5.4, 6.3, 7.5, hub (`monte-sua-base`) | 6 | teste final na v1 congelada; modelo de uso retreinado na v2 (recoleta), até set/2026, com recalibração numa janela recente e madura | **decidida 21/10 (B): v2 com treino até ~jun/2026 e CQR em ~jul–set; plano B às 22h** |
| 34 | Canal com menos de 3 posts maduros (F20): hoje a previsão recusa | 2.1, 5.4 | 6 | a cadeia da 2.1 (mediana; regressão em log dos inscritos; mediana do setor), sempre com o flag `canal_novo`; decide-se no caso à mão da A1 | **decidida 21/10 (B): número com `canal_novo`; a âncora sai do caso à mão da A1** |
| 35 | Transcrição no bloco de texto (F27): guardada, mas descartada em silêncio | 4.5 | 6 | limitação escrita e o comentário corrigido: do Whisper, o TAPI pede idioma, palavras e densidade de fala, que já entram (`aud_*`), e o efeito na amostra seria inconclusivo por tamanho (#101) | **decidida 21/10 (A); corrigido: o idioma também ficava fora (`FORA_DO_MODELO`)** |
| 36 | Leitura da ablação na abertura (F32): as três leituras da 2.6 lado a lado | 2.6 | 6 | mostrar escada, todos menos um e ruído juntos; a régua #27 continua sendo a regra | **decidida 21/10 (A): a #27 decide o modelo; só o bloco sólido nas três leituras vira argumento** |
| 37 | Carregamento do modelo de uso (F23): a 2.5 e a 7.1 pedem o MLflow (`models:/...`); a 6.1 e a 6.6 usam pasta mais variável de ambiente | 2.5, 6.1, 6.6, 7.1 | 6 | seguir a 6.x e registrar o bundle como artefato do run de treino final, com o run id no `model_version`; vai para `conflitos.md` | **decidida 21/10 (A); conflitos #19** |
| 38 | Corte 4 (21/10, 13h30): ~12,9 h de trabalho dele para ~8,5 h úteis até 23h59 | "Se atrasar" | 6 | B: A2 para o Claude (ele escreve descrições e erros); teach-backs para 22/10; autoavaliação e F14 do rascunho do Claude; M9 com pré-triagem do Claude; A4 e A7 do Claude com o caso à mão dele; README: ele reescreve 3 seções | **decidida 21/10 (B)** |

**Pendências do Claude:**

- resumos de 3 linhas da 3.4, 3.5, 4.8, 5.6, 6.7 e das EDD (corte #28), cada um entregue com a peça que o usa;
- recoleta das estatísticas antes do treino final (Fase 2);
- apagar os dados brutos em 30/10, pelo passo a passo do data card v1 (inclui as pastas do módulo 4 e do golden set); o MLflow é decisão do Vinicius;
- inscritos da API versionados em `data/canais.csv` até 30/10;
- adiados da revisão final do lote de 21/10 (nenhum bloqueia; o detalhe estava no ledger do plano
  `2026-10-21-lote-auditoria`):
  - antes da A1: `faixa_do_canal` com escalar devolve um array 0-d (a tool precisa de `.item()`);
  - antes do teste final: `k_escolhido` depende da ordem de `ks` e pula NaN em silêncio; `painel_do_run` e
    `previa.erros` não chamam `conferir_cortes`; o `avaliacao.json` do run é lido do arquivo, e não do `cfg_av`;
  - polimento: o F1 do painel por `bincount` (~6 s a menos por avaliação); o `auditoria.py:43` com caminho
    relativo; o check do pandera poderia exigir a lacuna (`<= published_at − lacuna`); a `linha` vazia passa no
    contrato; o `LIMITE_IMPORTANCIA` no código, e não na config.

**README (texto do Vinicius, na A8):** o porquê do alvo e "posts anteriores **e maduros**"; o critério de sucesso.

## Se atrasar

**Corte = o que o Vinicius deixa de estudar a fundo ou de escrever; nada sai do produto** (decisão do Vinicius,
15/10). O Claude implementa a parte cortada e o Vinicius faz teach-back de 3 linhas, para poder explicá-la em 23/10.

Corte nesta ordem:

1. leitura profunda só nas aulas que sustentam uma peça do Vinicius; viram rápidas 3.1, 3.2, 3.3, 3.6, 4.1, 4.4,
   5.1, 5.4, 5.5, 6.2, 6.3, 6.4, RAG 1–8 e EDD 1, 7 e 8 (6.2, 6.4 e RAG 4–6 porque A5 e A6 passaram para o Claude;
   as decisões dessas aulas seguem com ele, com a recomendação do Claude);
2. M9: o Claude escreve o bootstrap pareado; o Vinicius continua assistindo e classificando os 20 piores erros. A
   ablação roda com todos os blocos;
3. rubrica do VLM: continua no produto (o TAPI a põe no multimodal obrigatório); o Claude implementa com a
   recomendação da #13;
4. A6: o Claude escreve o grafo; o Vinicius faz teach-back;
5. A5: o Claude escreve; o Vinicius faz teach-back;
6. decisões em que a recomendação é "aceitar a proposta" (#10, #13, #16, #17, #22, #23) vêm num bloco, com
   aprovação ou veto em uma linha cada;
7. leitura dirigida em todas as aulas: só as seções que a peça usa (o código, "No projeto Galaxies" e "Erros
   comuns"); as rápidas viram resumo de 3 linhas do Claude; a leitura inteira vai para depois de 23/10;
8. o prompt da A4 e o teste de injeção da A7: o Claude escreve; o Vinicius faz teach-back;
9. as decisões restantes em dois blocos de aprovar ou vetar, menos a #5 (critério de sucesso), que vem inteira;
10. o README: o Claude rascunha a partir do `DECISOES.md`; o Vinicius reescreve;
11. M8: o Claude escreve; o Vinicius faz teach-back;
12. leitura dirigida só nas aulas das peças que continuam do Vinicius (6.1, 6.5, 5.2, 5.3, 2.6, 5.7 e 6.6); 3.4,
    3.5, 4.8, 5.6, 6.7 e as EDD viram resumo de 3 linhas do Claude, lido com a peça; o bloco 2 de decisões (#14 a
    #18 e #23) em aprovar ou vetar, uma linha cada; H3 e H5 a H9 curtas (hipótese e mecanismo; a métrica é a régua
    da #27);
13. A2: o Claude escreve o servidor e o Vinicius faz teach-back, ou o Vinicius escreve só as descrições das tools e
    as mensagens de erro (~20 min). Decide-se no fechamento de 20/10, antes de a A2 começar.
14. (#38) a A2 com o Vinicius só nas descrições e nos erros; os teach-backs em 22/10; autoavaliação e F14 do
    rascunho do Claude; M9 com pré-triagem do Claude e confirmação dele (~1 min por vídeo); A4 e A7 escritas pelo
    Claude a partir do caso à mão dele; no README, ele reescreve problema, resultados e limitações.

Os itens 1 a 6 valem desde 15/10 (#24); os itens 7 a 10, desde 18/10 (#25); os itens 11 e 12, desde 20/10
(#28); os itens 13 e 14, desde 21/10 (#38).

**Sempre raciocinadas pelo Vinicius (caso à mão e passos, #26):** M1 a M4, M6, A1, A2, o validador da A4, a invariante da A7 e o texto da A8.

## Depois de 23/10 (trilha de consolidação)

- **Leitura:** o módulo 7 inteiro e a parte rápida do módulo 4.
- **Exercícios das aulas, na base sintética:**
  - 3.1 (ganho da raiz), 3.2 (RF × GB com bootstrap), 3.3 (GB do zero com early stopping), 3.4 (Optuna com
    CatBoost), 3.6 (número de componentes);
  - 4.x (calibrar detectores e sondas);
  - 5.5 (Shapley à mão, 30 min);
  - extensões da 6.2, 6.3 e 6.5;
  - os de 2.3 a 2.6.
- **Extensões para o portfólio:** 7.3 (duelo LLM zero-shot × modelo) e 7.5 (monitor de cobertura com ACI).
