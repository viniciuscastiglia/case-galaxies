# Roteiro da reta final (14 a 23/10/2026)

O plano vivo do projeto. O Claude lê este arquivo no início de cada sessão e o atualiza no fim (`/fechar-sessao`).
As decisões tomadas vão para `docs/DECISOES.md`; o histórico, para `docs/DIARIO.md`. O mapa das aulas está em
`docs/material/`.

## Agora

- **Fase:** 1, avaliação honesta na base real (14 e 15/10).
- **Próximo passo:** `/aula 2.2` e `/aula 2.3`. A primeira decisão é a regra das duplicatas (#1). Depois vêm as
  outras decisões da Fase 1 e a M1, cada bloco numa branch `feat/<bloco>`.
- **Bloqueios:** nenhum. A reestruturação de 13/10 foi commitada e integrada à `main` em 14/10.

## Como funciona

- **Vinicius escreve** (M# de ML, A# de AI Engineering): o coração da peça real, marcado com `TODO(human)`. As peças
  são M1–M4, M6, M8, M9, A1, A2 e A4–A8; não existem M5, M7 nem A3 (a A3 entrou na A2). Antes, o Claude deixa pronto
  o pandas de apoio, o esqueleto e o **teste-contrato** que a aula pede. O Vinicius implementa até o teste passar e
  depois faz o teach-back (`/teach-back`), antes de alguém construir em cima da peça.
- **o Vinicius decide:** o Claude traz contexto, 2 ou 3 opções, a recomendação com a aula e o custo de errar. Uma
  decisão por vez, em texto. A resposta vai para `docs/DECISOES.md`.
- **Claude:** encanamento, testes, integração e documentação de apoio. O Vinicius lê um resumo de 3 linhas.
- **Leitura:** todas as aulas, em duas velocidades. Profunda no núcleo de ML e AI Engineering; rápida nas de
  encanamento ("No projeto Galaxies", "Erros comuns" e "Resumo"). O tempo oficial das aulas inclui exercícios, então
  só a leitura leva cerca de 75% dele.
- **Ritmo:** cerca de 4,8 h por dia do Vinicius, quase sem folga. No fim de cada fase, `/fechar-sessao` compara o
  feito com este roteiro. Se uma fase estourar em mais de meio dia, entra o próximo corte.

## Fases

### Fase 0 · 14/10 de manhã · reestruturação

- [x] Material lido pelo Claude e mapeado (`docs/material/`).
- [x] CLAUDE.md, ROTEIRO, DECISOES, DIARIO, skills, settings e memória criados (13/10), com revisão independente.
- [x] Vinicius aprova o commit e o merge da branch `docs/reta-final` (14/10).
- [ ] Vinicius lê este roteiro (10 min).

### Fase 1 · 14 e 15/10 · avaliação honesta na base real

- **Leitura:**
  - [ ] 2.3
  - [ ] 2.4
  - [ ] 2.5
  - [ ] 2.6
  - [ ] EDD 1–8
  - [ ] RAG 1–8
- **Vinicius decide** (tabela "Decisões pendentes", #1 a #8):
  - [ ] duplicatas
  - [ ] início do treino
  - [ ] datas de teste, gap e calibração
  - [ ] canais fora do treino
  - [ ] critério de sucesso
  - [ ] peso dos canais dominantes
  - [ ] faixas
  - [ ] teto por canal (#8)
  - [ ] limiares de histórico (#20)
- **Vinicius escreve:**
  - [ ] M1
  - [ ] M2
  - [ ] `HIPOTESES.md`
- **Claude:**
  - [ ] `splits.py` e `metricas.py` portados da 2.3 e da 2.4, com testes. Ficam fora, como `TODO(human)`, as funções
    da M1 (`split_treino_cal_teste` e `canal_novo_no_futuro`) e a `pairwise_janela` da M2
  - [ ] o esqueleto e os testes-contrato da M1 e da M2
  - [ ] a contagem de Shorts por mês e por canal, para a decisão das datas
  - [ ] a trava do teste ajustada às datas decididas
  - [ ] a base congelada (versão e hash)
  - [ ] o esqueleto de experimentos (YAML + dataclass, MLflow com commit, hash e corte, teste de perturbação)
  - [ ] `auditar_datas` (1.7)
  - [ ] a tabela de baselines na base real, com a mediana madura e o fallback por degrau
  - [ ] a lista dos 50 Shorts para conferir
- **Pronta quando:** existe a tabela de baselines na base real, com o split definitivo e o run id, e as hipóteses
  estão escritas antes de rodar.

### Fase 2 · 16 e 17/10 · GBM Opção A

- **Leitura:**
  - [ ] 3.1
  - [ ] 3.2
  - [ ] 3.3
  - [ ] 3.4
  - [ ] 3.5
  - [ ] 3.6
  - [ ] 6.1
- **Vinicius decide:**
  - [ ] features de texto (#9)
  - [ ] LightGBM × CatBoost (#10)
  - [ ] lacuna (#11)
  - [ ] conferência dos 50 Shorts (#19)
  - [ ] idade do rótulo (#21)
- **Vinicius escreve:**
  - [ ] M3
  - [ ] M4
- **Claude:**
  - [ ] o esqueleto e os testes da M3 (`hist_ultimo_ts < published_at` e perturbação do futuro)
  - [ ] o bloco `txt_` (manuais, cache de embeddings, PCA no Pipeline)
  - [ ] o runner em volta da M4: configuração, carga dos dados, montagem das features e registro no MLflow. O treino
    em si (early stopping, refit e comparação com o baseline) é a M4
  - [ ] a recoleta das estatísticas antes do treino final
  - [ ] o início do download das thumbnails, em background
- **Marco 1:** GBM × baseline em walk-forward e em "canal novo no futuro", com IC pareado por canal.

### Fase 3 · 18/10 · tool mínima

- **Leitura:**
  - [ ] 6.5
  - [ ] 6.2
  - [ ] 6.3
  - [ ] Function calling (revisão)
- **Vinicius decide:** [ ] setores no contrato (#22)
- **Vinicius escreve:**
  - [ ] A1
  - [ ] A2
- **Claude:**
  - [ ] `prever_criativo` como esqueleto andante (carrega o modelo uma vez, devolve `model_version`)
  - [ ] as versões fixadas (`fastmcp==3.4.7`, `mcp==1.30.0`, langgraph, langchain)
  - [ ] a guarda do OpenMP
  - [ ] os stubs das tools
- **Marco 2:** `prever_criativo` responde pelo MCP no Inspector e dentro do Claude Code (`claude mcp add`), com o
  número do GBM.

### Fase 4 · 19 e 20/10 · multimodal leve e incerteza

- **Leitura:**
  - a fundo: [ ] 4.1, [ ] 4.4, [ ] 4.8
  - rápida: [ ] 4.2, [ ] 4.3, [ ] 4.5, [ ] 4.6, [ ] 4.7
  - [ ] 5.1, [ ] 5.2, [ ] 5.3, [ ] 5.4
- **Vinicius decide:**
  - [ ] recorte multimodal (#12)
  - [ ] rubrica do VLM (#13)
  - [ ] limiares do flag (#14)
- **Vinicius escreve:**
  - [ ] M6
  - [ ] M9
- **Claude:**
  - [ ] thumbnails da base toda (SigLIP 2, sondas, OpenCV, PCA no treino)
  - [ ] amostra com frames, cortes e transcrição (apagando a mídia depois de extrair)
  - [ ] rubrica do VLM, se estiver no ritmo
  - [ ] fusão com prefixos
  - [ ] o runner da ablação
  - [ ] o `confidence_flag` com as regras decididas
  - [ ] a medição da maturação
- **Pronta quando:** existe a tabela de ablação com IC e a cobertura do intervalo de 80% está medida, total e por
  setor.

### Fase 5 · 21 e 22/10 · explicação e agente

- **Leitura:**
  - [ ] 5.5
  - [ ] 5.6
  - [ ] 5.7
  - [ ] 6.4
  - [ ] 6.6
- **Vinicius decide:**
  - [ ] LLM do explicador (#15)
  - [ ] background do SHAP (#16)
  - [ ] explicação pelo MCP (#17)
  - [ ] criativos da demo (#18)
  - [ ] payload do explicador (#23)
- **Vinicius escreve:**
  - [ ] M8
  - [ ] A4
  - [ ] A5
  - [ ] A6
  - [ ] A7
- **Claude:**
  - [ ] SHAP (`pred_contrib`)
  - [ ] payload e template da explicação
  - [ ] cliente do LLM e LLM falso
  - [ ] corpos das tools 1 a 5
  - [ ] índice de comparáveis
  - [ ] `Passos` do grafo
  - [ ] golden set congelado (canais fora do treino)
  - [ ] testes de contrato e de regressão
  - [ ] CI no GitHub Actions (pytest e ruff, sem dados e sem LLM)
- **Pronta quando:** o agente passa nos testes (invariante, injeção, golden set) e a explicação é validada.

### Fase 6 · 23/10 · entrega

- **Leitura:** [ ] 6.7
- **Vinicius escreve:**
  - [ ] A8
  - [ ] autoavaliação do hub
- **Claude:**
  - [ ] tabelas do README
  - [ ] data card e model card finais
  - [ ] respostas da demo salvas em JSON (plano B)
  - [ ] verificação final

## Dia a dia (horas do Vinicius)

| Dia | Leitura | Prática e decisões | ~h |
| :--- | :--- | :--- | :--- |
| Qua 14/10 | 2.2 (revisão do erro comum 6), 2.3, 2.4, EDD 1–8 | decisões #1–#4 e #20; M1 | 4,1 |
| Qui 15/10 | 2.5, 2.6, RAG 1–8 | decisões #5–#8 (depois da 2.5); M2; `HIPOTESES.md`; leitura da tabela de baselines | 4,6 |
| Sex 16/10 | 3.1, 3.2, 3.3, 3.5 | M3; conferir os 50 Shorts | 4,8 |
| Sáb 17/10 | 3.4, 3.6, 6.1 | decisões #9–#11 e #21; M4; leitura do marco 1 | 4,9 |
| Dom 18/10 | 6.5, 6.2, 6.3 | decisão #22; A1; A2 | 4,9 |
| Seg 19/10 | 4.1, 4.4, 4.8 + rápidas 4.2, 4.3, 4.5–4.7; 5.1, 5.2 | decisões #12 e #13 | 4,4 |
| Ter 20/10 | 5.3, 5.4 | decisão #14; M6; M9; criar a `GOOGLE_API_KEY` no AI Studio e pôr no `.env` (5 min) | 5,0 |
| Qua 21/10 | 5.5, 5.6, 5.7 | decisões #15–#18 e #23 (manhã); M8; A4 | 5,1 |
| Qui 22/10 | 6.4, 6.6 | A5; A6; A7 | 4,8 |
| Sex 23/10 | 6.7 | A8 (README, demo, ensaio de perguntas); autoavaliação; entrega | 4,2 |

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
| M9 | ablação: bootstrap pareado por canal e os 20 piores erros | 2.6, 4.8 | o bootstrap, mais assistir e classificar os 20 piores | reamostra canais inteiros; modelos idênticos dão IC com zero | 1,5 |
| M8 | `top_drivers` a partir do SHAP | 5.6 | multiplicador, direção e agrupamento por bloco | aditividade (soma dos SHAP + base = previsão); multiplicador = e^φ | 1 |
| A1 | contrato Pydantic de `prever_criativo` (`schemas.py`) | 6.1 | Request e Response, validadores, `extra="forbid"` | recusa data sem fuso, setor fora, campo extra e data a mais de 90 dias; aceita o pedido válido; low ≤ pred ≤ high | 1 |
| A2 | servidor MCP + Inspector + `claude mcp add` (`servidor_mcp.py`) | 6.5 | o servidor, as descrições das tools e os erros legíveis | o cliente em memória lista as tools; o MCP devolve o mesmo que a função; pedido sem fuso vira erro legível | 1,5 |
| A4 | explicador com validador de fidelidade | 5.7, EDD 3–5 | o prompt (legenda como dado) e o `validar` | aprova a explicação fiel; reprova número inventado e falta de intervalo; a legenda com instruções não muda o número | 2 |
| A5 | comparáveis sem vazamento (`comparaveis.py`) | 6.4, RAG 4–6 | a busca kNN com os filtros aplicados antes | nunca devolve post publicado depois da data de corte, nem o próprio post, nem outro setor | 0,75 |
| A6 | grafo LangGraph do fluxo da previsão (`agente.py`) | 6.2 | o `StateGraph`: estado, nós e aresta condicional | com `Passos` falsos, texto pula a extração e vídeo passa por ela; o número final é o da função | 1,5 |
| A7 | testes do agente | 6.6, EDD 2 e 6 | a invariante "número do agente = número do modelo" e a injeção na legenda | (os testes são a peça) | 1 |
| A8 | README e demo | 6.7 | problema, metodologia, resultados e limitações; ensaio cronometrado; ensaio de perguntas | — | 3 |

## Decisões pendentes

O Claude traz cada uma na hora certa, com contexto e opções. A recomendação abaixo é a de partida.

| # | Decisão | Aula | Fase | Recomendação do Claude | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Regra das duplicatas (`duplicate_group`: 644 grupos e 1.089 cópias extras; Principia e Mundo do Automóvel para PCD têm títulos de código como "CH 01") | 2.2, 2.3 | 1 | critério escrito antes de olhar curtidas: uma cópia por grupo (a primeira publicada) nas linhas de treino e teste, e todas no histórico do canal; conferir à mão 10 grupos da Principia antes de aplicar | aberta |
| 2 | Início do treino (236 rotulados antes de set/2020 e 273 antes de jul/2021: vídeos verticais curtos, ainda não Shorts) | 2.3 | 1 | a partir de 01/07/2021 | aberta |
| 3 | Datas de teste, gap e calibração (a trava provisória está em 01/04/2026) | 2.3, 5.3, 7.5 | 1 | teste de 3 a 6 meses, começando em 01/04/2026 ou depois e terminando no último Short maduro; gap de 30 dias; calibração nas 6 a 8 semanas antes do teste, só com posts maduros. O Claude traz a contagem por mês e por canal | aberta |
| 4 | Canais fora do treino, para o golden set e a demo | 2.3, 6.6, 6.7 | 1 | 2 a 3 canais por setor, sorteados por regra escrita antes e estratificados por tamanho | aberta |
| 5 | Critério de sucesso (também vai para o README) | 2.4, 2.5 | 1 | par-a-par intra-canal do GBM acima do baseline, com IC pareado por canal sem o zero; os ~60% como referência; incluir "canal novo no futuro" e a cobertura do intervalo | aberta |
| 6 | Peso dos 3 canais que dominam os pares: Mobiauto, Mundo do Automóvel para PCD e Principia. Num teste desde 01/04 eles somam 61% dos pares (79% na janela de 14 dias), e os 3 não têm linha no treino; isso também pesa na #4 | 2.4 | 1 | manchete com média ponderada e, ao lado, a média com peso igual por canal; teto de pares por canal | aberta |
| 7 | Convenção de faixas | 1.6, 2.4 | 1 | quantis do histórico do canal (1.6) | aberta |
| 8 | Teto por canal nas linhas de treino. Já decidido em 08/10: o teto é parâmetro do experimento e o histórico usa todos os posts. Resta decidir o valor (400?) e qual versão é a manchete | 1.4 | 1 | rodar com e sem teto; manchete sem teto, com o resultado com teto ao lado | aberta |
| 9 | Features de texto: a lista manual (CTA, pergunta, hashtags, números, termos do setor) e se os embeddings entram | 3.6 | 2 | manuais + embeddings multilíngues com PCA de 16, decididos pela ablação | aberta |
| 10 | LightGBM × CatBoost | 3.4 | 2 | LightGBM (ver conflitos #3); o CatBoost como experimento opcional | aberta |
| 11 | Lacuna do histórico | 3.5, 1.7 | 2 e 4 | 30 dias até a medição da maturação (19–22/10); rodar o experimento 0/7/30 da 3.5 | aberta |
| 12 | Recorte multimodal | 4.1, 4.2, 4.8 | 4 | thumbnails da base toda (SigLIP 2, sondas, OpenCV) + amostra estratificada de ~1.000 a 1.500 Shorts com frames, cortes e transcrição, por regra escrita antes | aberta |
| 13 | Rubrica do VLM | 4.7 | 4 | 5 a 8 campos com âncoras (gancho, rosto, produto em close, antes/depois, texto na tela, energia da música), via Gemini por URL na amostra, só se estiver no ritmo | aberta |
| 14 | Regras e limiares do `confidence_flag`, inclusive a regra "fora da distribuição" (conflitos #12) | 5.4, 6.4 | 4 | regras 1, 2 e 6 (+4 com embeddings); "fora da distribuição" por cosseno ao vizinho mais próximo, no espaço dos comparáveis; limiares calculados na calibração; taxa entre 15 e 35% | aberta |
| 15 | LLM do explicador | 5.7 | 5 | Gemini (`gemini-3.5-flash-lite`) via `langchain-google-genai`; free tier só com Shorts publicados (princípio 5); a chave `GOOGLE_API_KEY` é criada pelo Vinicius e vai para o `.env`; testes com LLM falso | aberta |
| 16 | Background do SHAP | 5.6, 6.3 | 5 | o da 6.3, mais o `assert` de aditividade (ver conflitos #7) | aberta |
| 17 | Explicação pelo MCP | 6.5, 5.7 | 5 | template no servidor (sem LLM); explicador LLM só no grafo (ver conflitos #8) | aberta |
| 18 | Criativos da demo | 6.6, 6.7 | 5 | regra escrita antes, por exemplo o Short maduro mais recente de cada canal fora do treino, ao menos um por setor | aberta |
| 19 | Conferência de 50 classificações de Short (concordância vai para o data card) | 1.5 | 2 (até 16/10) | o Claude monta a lista; o Vinicius confere | aberta |
| 20 | Limiares de histórico: 30 Shorts para o canal entrar (1.4), 10 posts anteriores para o alvo relativo (1.6), 3 para o cold start (2.1) | 1.4, 1.6, 2.1 | 1 | manter os três, um em cada papel; `canal_novo` (5.4) com menos de 10 | aberta |
| 21 | Idade do rótulo (viés 3 do data card): mediana de 324 dias, 396 no treino e 52 no teste | 1.7, 3.5 | 2 | experimento na M3/M4: `target_age_days` como variável de controle, fixada em 30 na previsão, × treinar só com rótulos de idade parecida. Decidir pela validação temporal | aberta |
| 22 | Setores no contrato: `Literal` com os 2 setores, e setor fora vira erro de validação (conflitos #14) | 6.1 | 3 | aceitar a proposta, entra na A1 | aberta |
| 23 | Payload do explicador: o da 6.3, mais a checagem de ressalvas da 5.7 (conflitos #13) | 5.7, 6.3 | 5 | aceitar a proposta, entra na A4 | aberta |

**Pendências do Claude:**

- `auditar_datas` (Fase 1);
- congelar a base de treino (Fase 1);
- recoleta das estatísticas antes do treino final (Fase 2);
- medir a maturação (19–22/10, com cerca de 2 semanas de snapshots);
- apagar os dados brutos em 30/10;
- inscritos da API versionados em `data/canais.csv` até 30/10.

**README (texto do Vinicius, na A8):** o porquê do alvo e "posts anteriores **e maduros**"; o critério de sucesso.

## Se atrasar

Corte nesta ordem:

1. leitura rápida também em 3.1 a 3.3 e 6.7;
2. M9 só com os blocos meta, canal e texto;
3. rubrica do VLM fora, como limitação documentada;
4. A6 vira teach-back sobre o grafo escrito pelo Claude;
5. A5 vira teach-back.

**Nunca cortar:** M1 a M4, M6, A1, A2, o validador da A4, a invariante da A7 e a A8.

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
