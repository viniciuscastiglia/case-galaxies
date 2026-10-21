# Auditoria de conformidade (#30), segunda passada

Feita em 21/10/2026 (terminada às 09h40 e revisada às 09h55), a pedido do Vinicius. Substitui a primeira passada, da manhã cedo, que leu
só o "No projeto Galaxies" e os "Erros comuns" de cada aula. Esta leu tudo inteiro e cruzou com o código, nas duas
direções: o que as aulas pedem está implementado, e o que está implementado segue as aulas.

**Fontes lidas inteiras:**
- o TAPI (`escopo_projeto.md`);
- as 10 páginas do hub (`index`, `roadmap`, `monte-sua-base`, `stack-2026`, `leituras`, `templates`, `autoavaliacao`,
  `troubleshooting`, `laboratorio` e `glossario`);
- as 47 aulas da trilha: 42 nos módulos 1 a 6 e 5 no módulo 7, o "ir além", que é opcional;
- RAG 1 a 8 e EDD 1 a 8.

**Repositório:** `src/previa/` inteiro, `configs/`, `tests/`, o CI, o README, o data card, o `DECISOES.md`, o
`ROTEIRO.md` e as regras e os textos do `HIPOTESES.md`.

**Verificação (21/10, 09h30):**
- `uv run pytest`: 273 passed em 51 s, com 3 avisos de depreciação do shap;
- `uv run ruff check .`: All checks passed;
- `uv run ruff format --check src tests`: 58 arquivos ok;
- todo nome de teste citado aqui existe (conferido com grep em `tests/`).

## O que esta passada não cobriu

1. **Resultados.** Nenhum número de experimento foi conferido. Os runs das H2 a H9 e da ablação estão lacrados até a
   abertura das hipóteses, e o teste nunca foi aberto. A auditoria confere o método, não os números.
2. **As peças de hoje ainda não existem** (A1, M6, flag, M8, tools, A6, A2, A4, A7 e A8). Para elas, a seção 3 diz o
   que precisam cumprir, não se cumprem.
3. **Instalação e CI não testados.** O README não foi seguido numa máquina limpa (6.7, EC6), e o CI não foi visto
   rodando no GitHub (o primeiro push é do Vinicius).
4. **Material interativo das aulas.** Exercícios, quizzes e simuladores não foram executados. Só entram aqui quando
   o "No projeto Galaxies", o hub ou o TAPI os pedem como entregável.
5. **Referências externas.** Papers e documentação das bibliotecas não foram lidos; as versões são as da Stack 2026
   do hub.
6. **`estudos/`** (os exercícios do Vinicius) ficou de fora, porque não é produto.
7. **Decisões conferidas contra o código.** Conferi só as que mudam código: #1, #2, #3, #4, #7, #9, #11, #13, #20,
   #87 e os princípios 2, 5 e 6. As de processo (cortes, formato das peças) não têm o que conferir no código.
8. **A compactação da conversa no meio da leitura.** As notas das aulas 6.1 a 6.5 vieram do resumo automático da
   conversa. Às 09h55, conferi contra o texto das aulas o "No projeto Galaxies", os "Erros comuns", o "Resumo" e os
   números citados. Os achados F1 a F42 estavam inteiros, mas a seção 3 tinha perdido 10 itens de revisão, já
   repostos, e saiu a F43.

**Como ler as situações:**
- **ok:** feito, com evidência;
- **desvio:** feito de outro jeito, com o porquê no `DECISOES.md`;
- **parcial:** uma parte está feita;
- **hoje:** está no plano de 21/10;
- **falta:** não existe e não estava no plano.

Nas tabelas, **NP** = "No projeto Galaxies" e **EC** = erro comum. Para conferir por amostragem, sorteie linhas e peça
a evidência ao vivo.

## 1. Resumo

- **O que já existe segue as aulas**, com quatro exceções que mudam um número ou uma leitura:
  - as faixas ainda usam os cortes fixos (F13);
  - a data de corte não está no MLflow (F22);
  - as sondas da capa nunca foram validadas (F26);
  - a transcrição nunca entra no modelo, embora o código diga que entra (F27).
- **O caminho crítico de hoje tem quatro buracos que não estavam no plano:**
  - não há código para o teste final (F16);
  - o bundle recusa os blocos extras (F17);
  - os caminhos relativos quebram o servidor MCP (F18);
  - falta o `features_used` no contrato. Sem ele, a invariante da A7 não consegue recalcular o número por fora (F19).
- **Decisões novas para o Vinicius**, uma por vez, com o problema primeiro (seção 9):
  - a perda do modelo (F24);
  - o canal com menos de 3 posts (F20);
  - a regra de data do flag e a base do treino final, que andam juntas (F21 e F31);
  - a camada 1 do multimodal (F2);
  - a transcrição (F27).
- **Entregáveis do hub que estavam fora do plano:**
  - o relatório de experimento (F28);
  - a comparação com a literatura (F29);
  - as taxas de trajetória e de fidelidade do agente (F30).

## 2. Faltas e pendências, por prioridade

### 2.1 Caminho crítico de hoje (sem isto, a entrega não fecha)

| # | Exigência | Fonte | Situação e evidência | Custo | Quem |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F16 | Avaliação final: a H1 no teste (01/04 a 14/09/2026) nos dois cenários, temporal e canal novo no futuro. Ganho com IC pareado por canal; `carregar_base(final=True)` chamado uma vez; run com tag própria | 2.3; 2.5 (EC4); `HIPOTESES.md` (H1); #5 | **falta.** `carregar_base(final=True)` só aparece nos testes; o runner avalia só 2025 (walk-forward e canal novo de desenvolvimento) | ~1 h, depois da abertura das hipóteses | Claude |
| F17 | A ablação decide quais blocos a tool calcula para A e B | 4.8 NP; 6.3 | **falta.** O bundle recusa blocos extras (`modelo.py:56`, `test_bundle_recusa_extras`). Se a capa (H7) ficar, a tool calcula SigLIP, sondas e aparência da capa do criativo, no processo da tool e com a guarda do OpenMP. Se o VLM ou o vídeo (H8, H9) ficarem, entram a Tool 1 com MP4 local (F11) e a rubrica nos 30 Shorts do golden | ~1 h para a capa; mais se H8 ou H9 ficarem | Claude, depois da escolha dos blocos |
| F18 | Caminhos absolutos e variável de ambiente para artefatos, dados e configs. O servidor MCP é lançado de outro diretório | 6.5 (EC2); 6.6 (`GALAXIES_ARTEFATOS` no `conftest`) | **falta.** Hoje: `Path("artefatos")` (`modelo.py:37`), `Path("data")` (`coletor.py:34`) e `Path("configs")` (`dados.py:22`, `splits.py:18`, `baselines.py:34`, `amostra.py:24`) | ~30 min | Claude |
| F19 | `features_used` na resposta, que é a "inspeção" do TAPI (quais atributos entraram) | TAPI §7; 1.1 NP; 6.1; 6.6 (contrato e invariante) | **hoje** (A1), mas não estava na lista do contrato | dentro da A1 | Vinicius (contrato) e Claude (código) |

### 2.2 Alta: muda o número, a leitura ou a defesa

| # | Exigência | Fonte | Situação e evidência | Custo | Quem |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F20 | Canal com menos de 3 posts maduros: recusar, ou seguir uma cadeia de fallback (mediana; regressão em log dos inscritos, ajustada no treino; mediana do setor). Na cadeia, o número aparece sempre, com o flag `canal_novo`, e o README diz quantos posts caem em cada degrau | 2.1; 5.4 ("o número aparece mesmo com o flag"); #20 | **hoje, sem decisão.** O `prever_log` recusa (`modelo.py:145`). O ROTEIRO (Fase 1) já pedia "o fallback por degrau" | ~30 min | Vinicius decide (A1) |
| F21 | A regra `data_fora_do_periodo` dispararia em toda previsão de hoje e nos Shorts do golden publicados depois de ~30/06. Ela marca mais de 180 dias entre a data planejada e o último post da tabela de treino do bundle (a `ultima_data` da 6.1); o nosso bundle treina até ~01/01/2026, ~290 dias antes de 21/10 | 5.4; 6.3 (limiares no bundle); 7.5 (retreino com recalibração) | **hoje** (#14) | dentro do flag | Vinicius decide, junto com a F31 |
| F22 | Data de corte e esquema de avaliação no run (tag `data_corte`, `avaliacao.yaml` como artefato), e o `comparar` recusando runs com cortes diferentes | 2.5 (EC3); hub, roadmap | **falta.** O `registrar` grava commit, hash da base e config (`experimento.py:458-478`), mas não o corte; o `comparar` confere só o hash (`:491`) | ~15 min | Claude |
| F23 | Rastreio do modelo de uso: o bundle como artefato do run de treino final, e o run id dentro do `model_version` | 2.5 NP; 7.1 (registro no MLflow) × 6.1, 6.5 e 6.6 (pasta de artefatos com variável de ambiente) | **falta.** Hoje: joblib em `artefatos/` (`modelo.py:82-83`). As aulas conflitam: a recomendação é seguir a 6.x (pasta mais variável de ambiente) e registrar o bundle no run | ~20 min, mais um conflito novo em `conflitos.md` | Claude |
| F24 | A perda do GBM. Todas as configs usam `objective: regression` (L2), sem decisão registrada. A 3.3 diz que "a escolha da perda decide o que o modelo diz sobre A e B"; Huber (δ ≈ 1 em log) é o padrão das aulas e do vencedor do SMP 2025 | 1.2 (EC3); 3.3 NP; 3.4 | **falta** (desvio sem registro) | uma hipótese curta, 1 config e 1 run de desenvolvimento (~20 min), antes de abrir o teste; ou registrar a L2 | Vinicius decide |
| F25 | Baseline mais forte (média dos últimos k posts maduros, com k escolhido na validação; mediana de 90 dias) e skill score (1 − MAE do modelo / MAE do baseline) | 2.1 (regras 1 e 3; exercício) | **parcial.** Existem a mediana madura e a mediana da 2.1 (`baselines.py`) | ~30 min | Claude |
| F26 | Sondas da capa validadas (AUC acima de ~0,7 contra rótulos) antes de virar feature | 4.4 (EC5) | **falta.** `configs/multimodal.yaml:20` diz "a validar"; o `test_sondas_relativas_tiram_a_media_da_imagem` confere só a conta. Rótulos baratos: os campos `tem_rosto`, `texto_na_tela` e `close_up_produto` do VLM, nas capas da amostra | ~15 min, antes de ler a H7 | Claude |
| F27 | A transcrição entra no bloco de texto (embedding mais PCA no fold) | 4.5 ("a transcrição entra no bloco de texto, junto com título e legenda") | **falta, e em silêncio.** O `midia.py:354` guarda `txt_transcricao` com o comentário "vai para o texto", mas o `ler_extra` descarta as colunas de texto (`experimento.py:145`) | ~45 min e 1 run na amostra; ou limitação no README | Vinicius decide (recorte multimodal) |
| F2 | Camada 1 do multimodal: SigLIP dos frames (o gancho, com vetor próprio) e detectores (rosto, OCR, objetos), mais a concordância do VLM com os detectores, que é a evidência barata de que a rubrica mede o que diz | TAPI §6.3; autoavaliação; 4.4; 4.6; 4.7 | **parcial.** Cumpre a leitura "thumbnail ou frames, mais Whisper e VLM" (4.1; TAPI §3 e §4.1), mas não a lista literal da camada 1. Os MP4 foram apagados. Saídas: rosto com o YuNet do OpenCV e texto com o PaddleOCR (roda no macOS arm64) nas 30.981 capas, sem baixar nada; ou baixar de novo o gancho dos 1.399 Shorts (~4 a 6 h de yt-dlp) | 1 a 2 h (só nas capas) | **decidida em 21/10 (#31, A)**; o Claude implementa depois da H10 |

### 2.3 Entregáveis do hub e do TAPI

| # | Exigência | Fonte | Situação | Custo | Quem |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F28 | Relatório de experimento (template do hub): hipótese, setup, resultados com Δ e IC, análise de erros com links, conclusão | hub (`templates`; `roadmap`, reta final) | **falta.** O `HIPOTESES.md` tem as hipóteses, não o relatório | ~45 min, depois do teste | Claude rascunha, Vinicius conclui |
| F29 | O README compara os números com a literatura: Spearman de 0,55 a 0,85; par-a-par de 76% nos 11 mil Shorts; +0,02 a +0,04 do visual | hub (`leituras`); 6.7 | **falta** (o README não cita a literatura) | ~20 min | Claude rascunha, Vinicius aprova |
| F30 | Taxas de trajetória correta e de fidelidade do agente com LLM real, em 20 a 30 perguntas com armadilhas (canal fora da base, data sem fuso, pergunta sobre views), e o tracing do MLflow (`mlflow.langchain.autolog()`) | 6.6 (camada 5; EC6); 6.7 (README, seção 7); EDD 2 e 6 | **falta.** A A7 cobre a invariante e a injeção | ~1 h e ~60 pedidos ao Gemini (só criativos publicados) | Claude |
| F3 | Model card e `model://card`: versão, métricas do teste temporal, cobertura, limites e as licenças dos checkpoints (SigLIP 2, Whisper, MiniLM, Gemini) | `roadmap`; 6.5; 6.7 (checklist); 7.4 | **hoje** (ROTEIRO, "data card e model card finais") | ~45 min | Claude |

### 2.4 Decisões de leitura, sem código

| # | Questão | Fonte | Situação | Quem |
| :--- | :--- | :--- | :--- | :--- |
| F31 | Base do treino final. Recomendação: o teste final roda na v1 congelada (a mesma base das H1 a H9). O modelo de uso é retreinado na v2, que é a recoleta, com dados até set/2026 e recalibração numa janela recente e madura. Isso fecha a F21 e a "recoleta das estatísticas antes do treino final" do hub | hub (`monte-sua-base`); 7.5 (receita 1); #87; ROTEIRO (Fase 2) | **hoje** (ROTEIRO, "a recoleta das estatísticas antes do treino final") | Vinicius decide |
| F32 | Na abertura das hipóteses, as três leituras da 2.6 lado a lado: escada, todos menos um e controle de ruído. Pela 2.6, a contribuição é sólida quando aparece na escada e na remoção, com IC sem o zero, acima do ruído; a régua #27 olha só controle × tratamento | 2.6 | **hoje.** A ablação já tem as três leituras (`previa.ablacao`) | Vinicius (custo zero) |

### 2.5 Da primeira passada (continuam valendo)

| # | Exigência | Fonte | Situação | Custo | Quem |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F1 | Painel de métricas: MAPE, Spearman global, F1 nas faixas, RMSE e e^MAE, a cobertura e o valor de cada fold, cada um com IC. As funções existem (`test_mae_e_mape`, `test_spearman_bate_com_scipy_e_constante_vale_zero`, `test_f1_faixas_e_cobertura`) | TAPI §6.1; 2.4; 2.3 (EC6) | **falta** no runner | ~30 min, sem rodar de novo | Claude |
| F4 | Baseline "regressão linear sobre inscritos": os inscritos são do dia da coleta | TAPI §6.1; 1.7 (EC3) | desvio sem registro | 5 min (registrar) | Claude |
| F5 | Repetir as comparações decisivas em outras seeds | 2.6 (EC5); 3.2 (EC2) | **falta** (seed 0 em tudo) | ~15 min na base toda | Claude |
| F6 | Limiar do detector de cortes calibrado à mão | 4.3 (EC4) | **falta**: vai para as limitações | — | Claude |
| F7 | A explicação só usa como argumento os blocos que a ablação aprovou | 2.6 NP; 4.8 NP | **hoje** (A4, M8) | — | Claude e Vinicius |
| F8 | Juiz da explicação: de outra família que o redator, temperatura 0, calibrado com ~20 afirmações rotuladas. No README: aprovação na 1ª tentativa e depois da correção, fidelidade do juiz e um exemplo reprovado | 5.7; EDD 4 | **falta** | ~1 h, mais 15 min dele rotulando | Claude e Vinicius |
| F9 | Proveniência na resposta: "modelo treinado em N Shorts de M canais, entre X e Y; curtidas aos 30 dias; estimativa do modelo, não métrica do YouTube"; mais a versão (F23) | 1.7 NP; 1.8 NP; 2.5 NP | **hoje** (A1, A4) | — | Claude |
| F10 | Canal com curtidas ocultas: a tool diz isso em vez de inventar (`test_curtidas_ocultas_viram_nulo` na coleta) | 1.5 NP | **hoje** (A1) | — | Vinicius |
| F11 | Criativo não publicado: o mesmo caminho da base (360p, os mesmos frames), uma regra escrita para a capa, e o VLM nunca no free tier | 4.1 (EC3); 4.2 NP; 4.7 NP; 1.8 (EC4); princípio 5 | **hoje** (Tool 1) | — | Claude; a regra é do Vinicius |
| F12 | Sanidade: um canal fictício por faixa de tamanho, com o intervalo mais largo onde a base é rala | 1.4 NP | **falta** | ~20 min, depois da M6 | Claude |
| F13 | Faixas pelos quantis do histórico do canal (#7). Hoje, cortes fixos em `metricas.py:81` | #7; 1.1; 1.6 | **falta** | ~20 min | Claude |
| F14 | Conferência dos 50 Shorts (#19). A lista existe | 1.5 | **falta** | ~20 min | Vinicius |
| F15 | pandera na tabela de treino. Hoje roda só na base mínima (`esquema.py:120`, `coletor.py:484`) | princípio 6 | **falta** | ~20 min | Claude |

### 2.6 Pequenos (até ~10 min cada, todos do Claude)

| # | Exigência | Fonte | Situação |
| :--- | :--- | :--- | :--- |
| F33 | Corte por `fonte` do canal na análise de erros ("o modelo erra mais nos canais que vieram da busca?") | 1.4 | falta (`erros.py:31`) |
| F34 | Recusar ou sinalizar criativo horizontal ou com mais de 180 s ("o modelo não deveria pontuá-lo") | 1.5 NP | hoje (A1, Tool 1) |
| F35 | Importância (gain) registrada nos runs, para o alarme "uma feature domina" | 2.2 | falta |
| F36 | Random forest com defaults como segundo baseline ("se o boosting não vence, desconfie") | 3.2 | falta (~20 min) |
| F37 | Faixa de valores do treino por feature guardada no bundle, insumo do `fora_da_distribuicao` | 3.1 (EC4); 5.4 | hoje (flag) |
| F38 | Registrar no `DECISOES.md` que não há clusterização dos descritores (a camada 2 do TAPI §6.2; a rubrica não tem descritores livres) | TAPI §6.2; 4.8 | desvio sem registro |
| F39 | `langchain-google-genai==4.4.0` no `pyproject.toml` (o explicador e o juiz da 5.7; o `create_agent` da 6.2). O `mapie` não é preciso, porque o CQR é feito à mão | 5.1; 5.7; 6.2; hub (`stack-2026`) | falta, antes da A4 e da A6 |
| F40 | Silenciar o aviso de depreciação do shap no `pyproject.toml` | 6.6 | falta |
| F41 | Cobertura do intervalo por mês de publicação no teste, nas limitações | 7.5 (exercício); 5.2 | depois da M6 |
| F42 | Limitações a escrever: o VLM recebe a URL pública (risco pequeno de "lembrança" do pré-treino, 7.3); não há varredura semanal completa do coletor (1.5); o recall dos comparáveis não foi medido (RAG 6) | 7.3; 1.5; RAG 6 | falta (README e data card) |
| F43 | `rel_vizinhos` nunca foi medido: fica fora do modelo, e os comparáveis entram só como evidência, que é o caminho da 6.4 quando não há ganho medido. Registrar no `DECISOES.md` | 6.4 (Resumo) | desvio sem registro |

## 3. O que as peças de hoje precisam cumprir

Este é o "Review Focus" de cada peça. Cada item vira um teste ou uma saída antes do "pronto".

**A1, o contrato (6.1; 6.6, camada 1).**
- `PredictionRequest`:
  - `creative`, `caption`, `planned_datetime` (`AwareDatetime`), `channel_id` ou `channel_stats`, `sector` como
    `Literal["beleza", "automotivo"]` (#22);
  - `extra="forbid"`, métrica `likes` e `interval_level` 0,8.
- `PredictionResponse`:
  - `predicted_metric`, `interval_low` e `interval_high`, `relative_to_channel`;
  - `confidence_flag` com `motivos`, `top_drivers` (até 5), `comparables`, `explanation` (resumo, fatores, ressalvas);
  - `features_used` (F19) e `model_version` (F9, F23).
- Validadores:
  - `low ≤ pred ≤ high`;
  - data planejada até 90 dias no futuro (exercício da 6.1). Data passada não é recusada, porque o golden set usa.
- Casos decididos dentro da A1: menos de 3 posts (F20), curtidas ocultas (F10), horizontal ou longo (F34), tipo do
  criativo (vídeo, imagem ou texto).
- Testes:
  - recusas: sem fuso, setor fora, sem canal, campo extra;
  - hora local (22h UTC = 19h em São Paulo);
  - ida e volta pelo JSON;
  - intervalo incoerente recusado.

**`prever_criativo` e o bundle (6.1; 6.3).**
- É a API principal, uma função pura. O grafo e o MCP são adaptadores finos: só convertem formatos e erros e nunca
  calculam uma feature (6.1, EC3).
- Começa como esqueleto andante: sem os atributos do vídeo, responde com o flag `extracao_falhou`, e as peças entram
  sem mudar a assinatura (6.1, EC1 e NP).
- O modelo é carregado uma vez (`lru_cache`), nunca na importação, e há um `aquecer()`.
- Caminhos vêm da variável de ambiente (F18).
- Blocos extras conforme a ablação (F17).
- No bundle ficam os limiares do flag: largura p80 na calibração, `sim_min` p1, última data, 180 e 365 dias, e as
  faixas por feature (F37).

**M6, o intervalo (5.1 a 5.3).**
- Quantis:
  - três quantis em log (0,1, 0,5 e 0,9);
  - early stopping pela métrica `quantile` do mesmo `alpha`;
  - `min_child_samples` maior nos extremos;
  - `np.sort` e o número de cruzamentos registrado no run.
- Saída: o `predicted_metric` é `expm1` do q50, nunca `exp` da média; o intervalo é dito como "entre X e Y".
- CQR à mão, com `k = ⌈(n+1)(1−α)⌉` na calibração (fev–mar/2026, só posts maduros). O q̂ e a data ficam versionados
  com o modelo.
- Avaliação no teste:
  - fração abaixo de cada quantil, pinball por `alpha`, cobertura e largura;
  - cru × CQR, total, por setor e por faixa de posts anteriores (menos de 10, de 10 a 50, mais de 50);
  - erro-padrão da cobertura √(0,8 · 0,2 / n);
  - cobertura por mês (F41).

**Flag (5.4; #14).**
- Formato: `{baixa_confianca, motivos: [{codigo, texto}]}` com os 6 códigos estáveis da aula.
- Limiares calculados na calibração, num JSON versionado. O kNN fica no espaço padronizado dos comparáveis, com o p99
  na calibração, nunca no treino.
- A regra de data é decidida na F21.
- Avaliação: taxa de sinalizados entre 15% e 35%; MAE dos sinalizados × não sinalizados, também por motivo.
- O número aparece mesmo com o flag ligado. Os códigos vão para o explicador e para a ordenação.

**M8, o `top_drivers` (5.5; 5.6; #16).**
- Formato: `{feature, valor, contribuicao_log, multiplicador, direcao}`, com multiplicador e^φ "em relação ao Short
  típico do canal".
- Soma em log, convertida no fim; `assert` de aditividade dentro da tool.
- Agrupamentos: os PCs por bloco e as colunas redundantes que ficarem juntas (por exemplo, `vlm_tem_rosto` com a sonda
  de rosto).
- Fora do `top_drivers` ficam as features de contexto: `hist_*`, tendência, cadência, idade do canal, setor e
  `rel_vizinhos` (6.3, Tool 4).
- "Associação, não causa."

**A4, o explicador (5.7; EDD 3 a 5).**
- O payload é fechado: sem transcrição, sem o JSON do VLM, sem o SHAP inteiro. O código faz as contas; o LLM só narra.
- O validador determinístico checa:
  - todo número citado está nas saídas das tools (regex no formato brasileiro: "4.237" e "0,884");
  - os fatores estão contidos no `top_drivers`;
  - não há termos proibidos ("garante", "certeza", "causa") nem frase de persona.
- Uma nova tentativa; se falhar de novo, fecha com o template. O intervalo e o flag aparecem sempre.
- Intervalos que se sobrepõem são ditos: a ordem é provável, não garantida (6.2 NP).
- Comparáveis e atributos não são evidências independentes: a explicação não soma "o gancho é forte" e "os
  parecidos foram bem" como duas provas (6.4 NP).
- F7 e F8 entram aqui.

**As cinco tools (6.3; 6.4; RAG 4 a 6).**
- Cada tool responde às sete perguntas da 6.3: nome, docstring, entrada, saída, erros, idempotência e cache, custo e
  limites.
- O LLM enxerga só duas tools de alto nível, no nível do pedido (`prever_criativo` e `comparar_criativos`). Nunca um
  `prever(features)`, porque ele preencheria um gancho que ninguém extraiu (6.2, EC2; 6.3, EC1).
- As features saem das mesmas funções do treino, inclusive as dos blocos extras (6.3, EC4; F17).
- Tool 1: cache pelo SHA-256 do conteúdo, timeout de 120 s, erro que diz o que reenviar, criativo não publicado
  (F11).
- Tool 2, contexto:
  - canal desconhecido devolve "reenvie com `channel_stats`";
  - o índice dos comparáveis é construído com título e capa concatenados e o mesmo PCA do treino, com o encoder
    versionado junto com o índice (6.4, EC5);
  - o filtro de setor e de data (até a data planejada − 30 dias) vem antes da similaridade;
  - cada comparável cita `post_id` e data; é evidência, não causa.
- Tool 3: a única fonte do número.
- Tool 4: SHAP do q50 e template; o LLM é opcional e fecha com o template se falhar.
- Tool 5:
  - P(B > A) = Φ((q50B − q50A) / √(σA² + σB²)), com σ = (q90 − q10) / (2 · 1,2816);
  - a independência é dita como conservadora, e o número é "uma escala para ordenar";
  - o complemento é dito: 0,88 quer dizer que, em ~12% dos cenários, A rende mais (6.3, EC6);
  - o what-if é feito só dentro do suporte, como "o que o modelo acha" (7.2), ou declarado como não feito;
  - `ToolCallLimitMiddleware(run_limit=6)`.

**A6, o grafo (6.2).**
- Grafo fixo: extrair → contexto → prever → explicar → responder.
- Estado em JSON, com o reducer `trilha`; `InMemorySaver` com `thread_id`; `stream_mode="updates"` e
  `get_stream_writer()` para a inspeção nó a nó.
- Texto sem vídeo pula a extração e sai com `extracao_falhou`.
- `create_agent` (não `create_react_agent`), com `args_schema` igual ao contrato e `handle_tool_error=True`.
- Testes com passos falsos, `FakeComTools` e o subagente de explicação num `RunnableLambda`, incluindo grafo ==
  função pura.

**A2, o servidor MCP (6.5).**
- Servidor:
  - FastMCP com `instructions`; tools com `annotations` e timeout; `ToolError`;
  - resources `model://card`, `schema://prediction-request` e `schema://prediction-response`;
  - prompt `decidir_entre_criativos`;
  - stdio com `show_banner=False`, logging no stderr e nenhum `print()`; HTTP (Streamable, em `/mcp`) só em
    `127.0.0.1:8000`;
  - nenhum LLM no servidor, porque a persona tem o dela. Pelo MCP, a explicação sai do template (6.5, EC5 e NP;
    #17);
  - `fastmcp==3.4.7` com `mcp==1.30.0`.
- Testes com o `fastmcp.Client` em memória (`@pytest.mark.anyio`).
- Inspector e `claude mcp add` com caminhos absolutos e `-e` para a variável dos artefatos.
- Dogfooding: `MultiServerMCPClient` com `client.session(...)` e `load_mcp_tools`, uma sessão para todas as chamadas,
  não uma por chamada (6.5, EC4).
- O teste da banca: o número dito no Claude Code é exatamente o `predicted_metric`.
- Pergunta para a Galaxies: qual era do protocolo o cliente da persona fala.

**A7, os testes do agente (6.6; EDD 2, 6 e 8).**
- Pirâmide:
  - unidade e contrato sem LLM;
  - `conftest` define a variável dos artefatos antes do import;
  - grade de pedidos × horários.
- Invariante, com o número recalculado por fora (`expm1(nível + q50 cru)`):
  - grafo == modelo, grafo == função e MCP == função;
  - o ReAct só cita números das tools.
- Teste de injeção (princípio 8, red teaming da EDD 8).
- Golden de-para nos 30 Shorts de 6 canais fora da base, com limiares de alarme folgados (cobertura de 0,6 a 0,97,
  par-a-par ≥ 0,5).
- F30.

**A8, a demo e o README (6.7).**
- Demo:
  - 8 blocos em até 7 minutos;
  - todo número dito aparece na tela, com a fonte;
  - a regra que escolhe os criativos do de-para fica escrita antes de rodar;
  - plano B: gravação, prints do Inspector e do trace, respostas em JSON;
  - bloco ao vivo sem rede, com `aquecer()`.
- README:
  - as 9 seções com números: o painel (F1), a ablação no formato da 4.8 (degrau, colunas, Spearman, Δ, IC,
    decisão), F29, F30, a frequência do flag no teste, o custo da extração por vídeo e a versão em produção;
  - testado do zero.
- A autoavaliação do hub fica preenchida.

**O teste final (F16; hub, `troubleshooting`).**
- Abertura única, nos dois cenários, com o run registrado.
- Sanidade antes de comemorar: um par-a-par ou um Spearman intra-canal "bom demais" é sinal de vazamento.

## 4. Conformidade do que já existe

| Aula | Exigência (EC = erro comum) | Evidência | Situação |
| :--- | :--- | :--- | :--- |
| 1.1 | o número nunca vem do LLM | a invariante é a A7 | hoje |
| 1.2 | teste aberto uma vez (EC2); split por data (EC4); a perda escolhida pelo efeito, não pelo nome (EC3) | `test_desenvolvimento_nao_ve_o_periodo_de_teste`, `test_avaliacao_final_abre_o_teste_com_aviso_e_registro`, `test_corta_por_data_e_nao_pela_posicao_da_linha`; EC3 é a F24 | ok, menos a F24 |
| 1.3 | `collected_at` (EC4); o teste é o período mais recente (EC2) | `test_dias_diferentes_viram_serie`; datas da #3 | ok |
| 1.4 | excluídos registrados (EC5); critérios antes das curtidas (EC2); teto no data card; erros por fonte (F33); canal fictício (F12) | `test_avaliar_candidatos_registra_incluidos_e_excluidos`; #8; `data/canais.csv` versionado, com `motivo` | ok, menos a F12 e a F33 |
| 1.5 | playlist de uploads (EC1); `quotaExceeded` para (EC2); curtida oculta é nula (EC3); upsert (EC4); `age_days` (EC5); regra de Short (EC6) | `test_listar_ids_pagina_e_para_quando_a_pagina_ja_e_conhecida`, `test_quota_exceeded_para_sem_retry`, `test_curtidas_ocultas_viram_nulo`, `test_salvar_duas_vezes_no_mesmo_dia_nao_duplica`, `test_horizontal_ou_longo_nao_e_short` | ok; F10, F14 e F34 abertas |
| 1.6 | mediana em log (EC1, EC2); só o passado (EC3, EC4) | `test_alvo_e_log1p_das_curtidas`, `test_mediana_madura_so_usa_posts_com_30_dias`, `test_perturbar_o_futuro_nao_muda_o_passado` | ok; faixas: F13 |
| 1.7 | rótulo com 30 dias (EC1); curtidas (EC2); inscritos fora (EC3); hora local (EC4); idade do rótulo (EC5) | `test_rotulo_e_o_primeiro_snapshot_com_30_dias_e_respeita_o_teto`, `test_calendario_hora_local_feriado_e_idade`, `test_idade_do_rotulo_filtra_e_fixa_na_previsao`; nenhuma config de experimento usa inscritos | ok |
| 1.8 | base fora do git (EC1); refresh de 30 dias (EC2); esquema a cada exportação (EC5); data card no template | `.gitignore` (`data/*`, `*.parquet`, `mlruns/`, `mlflow.db`, `artefatos/`); `test_renovacao_dos_30_dias_renova_e_apaga_os_que_sumiram`, `test_main_falha_alto_se_algum_dado_passou_de_30_dias`, `test_main_valida_a_base_minima_a_cada_execucao`; `docs/DATA_CARD.md` | ok; F9 e F11 abertas |
| 2.1 | baseline sem o futuro (EC1); mesmas linhas (EC3); log (EC5); o baseline mais forte (regra 1); fallback em produção | `test_mediana_madura_perturbacao`, `test_linhas_de_avaliacao`; F25 e F20 | **parcial** (a 1ª passada dizia ok) |
| 2.2 | sem proporções do próprio post (EC1); sem target encoding do canal (EC2); PCA só no treino (EC4); dedupe antes do split (EC6) | `test_pca_ajustado_so_no_treino`, `test_linhas_do_modelo_ficam_so_com_a_primeira_copia`; F35 | ok |
| 2.3 | sem KFold (EC1); split por data (EC2); gap (EC3); escolhas no treino (EC4); calibração por data (EC5); os dois esquemas (EC7) | `test_walk_forward_avanca_no_tempo_com_gap`, `test_folds_de_desenvolvimento_ficam_dentro_do_treino`, `test_canal_novo_nunca_aparece_no_treino` | ok no desenvolvimento; a avaliação final é a F16 |
| 2.4 | par-a-par intra-canal como manchete (EC1); bootstrap por canal (EC4); mínimo de 3 e peso n_c (EC5); empates (EC6) | `test_peso_posts_a_manchete_da_6`, `test_ic_par_a_par_um_canal`, `test_janela_empate_na_previsao_vale_meio_e_empate_em_y_fica_fora` | ok; o painel é a F1 |
| 2.5 | config no run (EC1); hash da base (EC2); teste uma vez (EC4); `mlflow.db` fora do git (EC8); corte nas tags (EC3); o modelo de uso rastreável (NP) | `registrar` (`log_dict` da config, `git_commit`, `hash_base`); `.gitignore`; F22 e F23 | **parcial** (a 1ª passada dizia ok) |
| 2.6 | todos menos um (EC1); controle negativo (EC2); bootstrap pareado (EC3); canais, não posts (EC4); seeds (EC5); 20 piores assistidos (EC7) | `test_todos_menos_um_tira_so_o_bloco_e_compara_com_o_completo`, `test_controle_negativo_acrescenta_ruido_ao_completo`, `test_dif_mae_pareada_reamostra_canais`; EC5 é a F5; EC7 é a M9 | ok, menos a F5; leitura: F32 |
| 3.1 | profundidade limitada (EC1); sinalizar caso fora da faixa do treino (EC4) | `num_leaves: 15`, `min_data_in_leaf: 20`; EC4 é a F37 | ok / hoje |
| 3.2 | sem OOB nem validação aleatória (EC1); early stopping (EC3); RF como segundo baseline | `test_gbm_early_stopping_refit_e_seed`; F36 | ok |
| 3.3 | early stopping temporal (EC1); learning rate fixa (EC2); árvores rasas (EC3); a perda (NP) | `val_dias: 180`, `learning_rate: 0.05`; F24 | ok, menos a F24 |
| 3.4 | refit (EC1); `num_leaves` e `bagging_freq` (EC2); sem `channel_id` (EC4); números de partida | `test_gbm_early_stopping_refit_e_seed`; configs com os números da aula | ok |
| 3.5 | `shift` + maduros (EC1); ordem por data (EC2); features na base inteira antes do split (EC3); lacuna (EC4); hora local (EC5); sem inscritos (EC6); a mesma função na tool | `test_perturbar_o_futuro_nao_muda_o_passado`, `test_ordem_das_linhas_nao_importa`, `test_historico_30_dias_entra_29_nao`, `test_paridade_treino_uso` | ok (desvio: lacuna inclusive) |
| 3.6 | PCA só no treino (EC1); cache por `video_id` + hash (EC3); sem a descrição padrão do canal (EC4) | `test_pca_ajustado_so_no_treino`, `test_cache_por_video_e_hash_do_texto`, `test_titulo_e_a_primeira_linha_da_caption` | ok |
| 4.1 | cache por etapa e versão (EC2); mesmo caminho no treino e no uso (EC3) | `test_cache_por_video_nao_repete_e_erros_ficam_fora`; EC3 é a F11 | ok, menos a F11; F2 parcial |
| 4.2 | 360p pelo lado menor (EC2); "not a bot" para (EC3); vídeo apagado (EC5); data da capa no data card (EC6) | `test_baixar_video_cache_falha_permanente_e_bloqueio`, `test_extrair_e_apagar_so_apaga_quando_tudo_deu_certo` | ok (desvio: recorte 9:16) |
| 4.3 | RGB (EC1); largura fixa (EC2); fluxo sem atravessar cortes (EC3); detector calibrado (EC4) | `test_aparencia_com_gabarito`, `test_metadados_e_sinais_no_video_sintetico`; EC4 é a F6 | ok, menos a F6 |
| 4.4 | `.pooler_output` (EC1); PCA só no treino (EC3); **sondas validadas (EC5)**; capa, gancho e vídeo separados (EC6) | `test_embeda_recortado_em_lotes_e_usa_o_cache`; o `test_sondas_relativas_tiram_a_media_da_imagem` confere a conta, não a validade (F26); o gancho é a F2 | **parcial** (a 1ª passada dava EC5 como ok) |
| 4.5 | idioma só com fala (EC1); VAD (EC3); `small` (EC4); transcrição guardada (EC5); transcrição no bloco de texto (NP) | `midia.py` (`vad_filter=True`, `language_probability >= 0.7`); `test_energia_do_audio`; o NP é a F27 | ok nos EC; falta a F27 |
| 4.6 | detectores leves (rosto, texto, objetos) | não implementados (MediaPipe aborta no Apple Silicon) | falta (F2) |
| 4.7 | o VLM não prevê (EC1); âncoras (EC2); Pydantic (EC3); backoff só em 429 e 5xx (EC4); prompt versionado (EC6); reteste | `test_rubrica_tem_6_campos_com_ancoras`, `test_nota_fora_da_escala_levanta`, `test_400_nao_repete`, `test_429_repete_e_depois_passa`, `test_teste_reteste_reproduz_a_aula`; kappa de 0,72 a 1,00 (DECISOES, 20/10) | ok; F2 e F11 abertas |
| 4.8 | PCA só no treino (EC1); mesmos splits e seeds (EC2); bootstrap pareado (EC3); colunas por prefixo (EC4); sem a flag de falha (EC5); todos menos um (EC6); a ablação decide os blocos da tool (NP) | `test_variantes_mudam_so_os_blocos`, `test_extras_entram_por_join_com_nan_e_prefixo`, `test_curinga_sem_colunas_levanta`, `test_bloco_extra_sem_nenhuma_feature_levanta`; o NP é a F17 | ok; F17 aberta |
| 5.6 | contribuições: `pred_contrib` e intervencional, aditividade, PCs somados num bloco | `test_a_conta_fecha_sem_fundo`, `test_a_conta_fecha_com_fundo_do_canal`, `test_blocos_somam_os_componentes` | ok (o `top_drivers` é a M8) |
| 6.1 | as mesmas funções de feature no treino e no uso | `test_paridade_treino_uso`, `test_paridade_com_a_lista_de_producao_e_historico_completo`, `test_salva_carrega_e_preve_igual` | ok; F17 e F18 abertas |
| 6.4 | só o passado maduro e nunca o próprio post (EC1 a EC3); um índice por setor (EC4); cosseno normalizado (EC6) | `test_mesmo_setor_e_so_o_passado_maduro`, `test_exatamente_30_dias_entra`, `test_feature_no_tempo_usa_a_regra_da_tool_e_nunca_o_proprio_post` | ok (desvio: numpy em vez de FAISS); o índice da tool é de hoje |
| 6.6 | golden: canais fora da base, posts maduros, sorteio com seed, congelado (EC3, EC4) | `test_candidatos_sao_as_reservas_revisadas_de_fora_da_base`, `test_escolhe_os_5_mais_recentes_maduros_do_periodo`, `test_sorteio_por_estrato_pula_canal_sem_5_shorts`; `data/golden/golden_v1.parquet` (30 Shorts, 6 canais, de 01/05 a 14/09/2026) | ok |
| 6.6, EDD 8 | CI sem dados e sem LLM | `.github/workflows/ci.yml` (`uv sync --frozen`, ruff e pytest) | ok |
| hub (`troubleshooting`) | LightGBM com torch no mesmo processo (OpenMP) | `test_torch_e_lightgbm_no_mesmo_processo_com_a_guarda`, `test_erro_alto_se_o_lightgbm_ja_carregou_sem_a_guarda` | ok |
| TAPI §6.2 | o ganho de cada bloco (metadados, + texto, + visual, + VLM) | `previa.ablacao` (escada, todos menos um, ruído); `test_ablar_ponta_a_ponta` | ok |

## 5. Os 9 princípios do `CLAUDE.md`

| Princípio | Situação | Evidência ou falta |
| :--- | :--- | :--- |
| 1. O número vem do modelo | hoje | a invariante da A7, que depende da F19 |
| 2. Sem vazamento | ok | histórico maduro (`test_historico_30_dias_entra_29_nao`); PCA no treino (`test_pca_ajustado_so_no_treino`); nenhuma config usa inscritos; `auditar_datas` (19/10); teste trancado (`test_desenvolvimento_nao_ve_o_periodo_de_teste`) |
| 3. Validação temporal e por canal | ok | `test_walk_forward_avanca_no_tempo_com_gap`, `test_canal_novo_nunca_aparece_no_treino`; o teste final é a F16 |
| 4. Paridade treino/uso | ok nas features de base | `test_paridade_treino_uso`; os blocos extras são a F17 |
| 5. Termos do YouTube | ok | `.gitignore`; refresh de 30 dias com teste; apagamento de 30/10 no data card; o VLM só recebe URLs publicadas. Aberta: a F11 |
| 6. Contratos | parcial | Pydantic e pandera na base (`esquema.py`); o contrato da tool é a A1; pandera na tabela de treino é a F15 |
| 7. Falha nunca silenciosa | ok, com 1 exceção | a transcrição descartada em silêncio, apesar do comentário (F27) |
| 8. Conteúdo do criativo é dado | parcial | o prompt do VLM trata fala e texto como dado (DECISOES, 18/10); o explicador e o teste de injeção são de hoje |
| 9. Fronteira com a Galaxies | ok | sem personas e sem nota de 0 a 5. As notas de 0 a 10 da rubrica são atributos de entrada, não nota do criativo |

## 6. Desvios

**Registrados, com o porquê no `DECISOES.md`:**
- a lacuna de 30 dias conta inclusive (a 3.5 usa "estritamente antes"), e a cadência não tem lacuna;
- a capa é recortada na faixa 9:16 (4.2);
- os comparáveis usam numpy em vez de FAISS (6.4, segfault);
- o áudio vai ao Whisper como array (4.5, PyAV 19);
- a rubrica tem 6 campos (o TAPI pede de 5 a 8);
- o rosto vem só pela rubrica (MediaPipe no Apple Silicon);
- o setor é código 0/1 para o shap intervencional (5.6);
- o texto usa o MiniLM-L12 da 3.6, não o mpnet que o TAPI cita;
- o baseline é a mediana do canal em log (05/10; a 2.1 usa a mediana), na versão madura, e não a "média histórica" do
  TAPI §4.1;
- o early stopping usa os últimos 180 dias do treino sem gap (as features já têm a lacuna; docstring do runner).

**Sem registro (a registrar hoje):** a regressão em inscritos (F4), a perda L2 (F24), a falta de clusterização dos
descritores (F38) e o `rel_vizinhos` fora do modelo (F43).

## 7. "Ir além" (módulo 7 e TAPI §4.2, opcionais)

| Extensão | Situação | O que fazer |
| :--- | :--- | :--- |
| Conformal (5.3) | hoje, na M6 | o TAPI põe o conformal no "ir além" e o intervalo por quantis no núcleo; a 7.1 mostra quantis cobrindo 60% de 80%, e por isso o CQR é o conserto honesto |
| RAG sobre comparáveis | parcial | o retriever existe (`comparaveis.py`); falta o índice da tool. Não chamar de "RAG" no README sem a evidência na resposta |
| What-if (7.2) | não feito | o TAPI põe "ordenação / what-if" na Tool 5. Se entrar, com o filtro de suporte e "o que o modelo acha"; se não, o README diz que a Tool 5 só ordena |
| Treino dinâmico (7.1) | não feito | o README não promete |
| LLM zero-shot × modelo (7.3) | não feito | o README não promete |
| Embeddings de vídeo e áudio (7.4) | não feito | o ritmo já está nos cortes e no movimento (Farnebäck) |
| Conformal adaptativo (7.5) | não feito | a cobertura por mês (F41) e o retreino com recalibração (F31) cobrem o essencial |

## 8. Correções da primeira passada

1. Dava a 2.5 como "ok". Faltam a data de corte no run (F22) e o rastreio do modelo de uso (F23).
2. Dava a 4.4 (EC5, sondas validadas) como "ok". O teste citado confere a conta, não a validade (F26).
3. Dava a 2.1 como "ok". Faltam o baseline mais forte (F25) e o fallback (F20).
4. Não viu:
   - que não há código do teste final (F16);
   - o bundle que recusa os extras (F17);
   - os caminhos relativos (F18);
   - o `features_used` (F19);
   - a regra de data que dispara hoje (F21);
   - a perda (F24);
   - a transcrição (F27);
   - os três entregáveis do hub (F28 a F30).
5. Contava "47 aulas nos módulos 1 a 6". São 42 nos módulos 1 a 6 e 5 no módulo 7.
6. A primeira versão desta segunda passada (09h40) também falhou: por causa da compactação da conversa, a seção 3
   ficou sem 10 itens de revisão das aulas 6.1 a 6.5 e sem a F43. Corrigido às 09h55 (veja o item 8 de "O que esta
   passada não cobriu").

## 9. Decisões para o Vinicius e ordem sugerida para hoje

**Decisões, nesta ordem** (os números são os da tabela do ROTEIRO). Cada uma vem em duas mensagens: primeiro o
problema com um exemplo, depois as opções.
1. **F2, a camada 1: decidida em 21/10 (#31, A).** Rosto com o YuNet e texto com o PaddleOCR nas 30.981 capas, sem
   objetos; a H10 dele vem antes de rodar.
2. **O próximo corte:** o dia somava ~12,5 h dele antes da auditoria, que acrescentou ~1 h; o item 13 de "Se
   atrasar" (a B da #28) segue aberto.
3. **F24, a perda (#32).** Vem antes da abertura das hipóteses, porque uma variante nova precisa de hipótese escrita
   antes de rodar.
4. **F31 e F21 juntas (#33):** a base do treino final e a regra de data do flag.
5. **F20 (#34):** o canal com menos de 3 posts, no caso à mão da A1.
6. **F27 (#35):** a transcrição.
7. **F32 (#36):** a forma de ler a ablação, na abertura.
8. **F23 (#37):** o carregamento do modelo de uso, em aprovar ou vetar.

Continuam abertas do bloco 2: #14 a #18, #19 e #23.

**Ordem sugerida:**
1. **Agora, Claude, sem decisão (~1h30):** F18, F22, F26 (antes de ler a H7), F25, F1, F13, F15, F39, F40, F33,
   F35, e os registros F4, F38 e F43.
2. **Vinicius:** as decisões 2 a 4 e a H10, depois a abertura das hipóteses (com a F32) e a escolha dos blocos.
3. **A1** (Vinicius, com F19, F20, F10 e F34), depois `prever_criativo` com F17 e F23 (Claude).
4. **M6** (Vinicius), o flag (#14, F21, F37), a M8 (Claude), as tools, a A6 e a A2 (Claude).
5. **A4:** o validador (Vinicius), o prompt, F7 e F8 (Claude).
6. **A7:** a invariante (Vinicius), a injeção e a F30 (Claude).
7. **F16, o teste final:** abertura única, com a H1 e a cobertura. Depois, F5 nas comparações decisivas e F31, o
   retreino do modelo de uso.
8. **Documentos:** README (F29, F41, F42, tabelas), model card (F3), data card final, relatório (F28), respostas da
   demo em JSON; F14 e a autoavaliação (Vinicius).

**Se o tempo apertar,** nada sai do produto. Primeiro viram limitação escrita no README os itens de evidência
adicional: F36, F12, F41 e, se ele decidir assim, F2 e F27.
