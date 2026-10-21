# Diário do projeto

Histórico das sessões. As entradas novas ficam no topo, com 3 a 5 linhas por sessão (`/fechar-sessao`). O plano
vivo está em `docs/ROTEIRO.md` e as decisões em `docs/DECISOES.md`.

## 21/10 (manhã, até ~10h) · auditoria completa (#30)

- Segunda passada da auditoria (`docs/AUDITORIA.md`): TAPI, as 10 páginas do hub, as 47 aulas, RAG e EDD lidos
  inteiros e cruzados com o código. Saíram F1 a F43. Quatro estão no caminho crítico:
  - não há código do teste final (F16);
  - o bundle recusa os blocos extras (F17);
  - os caminhos relativos quebram o MCP (F18);
  - falta o `features_used` no contrato (F19).
- Três "ok" da primeira passada foram corrigidos (2.1, 2.5, 4.4). A transcrição era descartada em silêncio, apesar
  do comentário no código (F27).
- A compactação da conversa caiu no meio do módulo 6. Ele perguntou se algo se perdeu. A conferência de 6.1 a 6.5
  contra as aulas repôs 10 itens de revisão e trouxe a F43. Lição: notas no arquivo a cada aula.
- Decidida a #31 (F2, opção A: rosto e texto nas capas, com a H10 antes de rodar). Nenhum código novo; 273 testes,
  ruff ok.
- Próxima sessão: a seção 9 da auditoria. O Claude faz os itens sem decisão; as decisões #32 a #37 e o próximo corte
  vêm uma por vez; depois, a A1.

## 20/10 (tarde e noite) e 21/10 (manhã) · hipóteses fechadas, mídia completa, leitura e auditoria

- O Vinicius escreveu a H3 e da H5 à H9 (`8e8935a`). Na H9, "não sei" virou hipótese honesta: os sinais ajudam no
  número; se acrescentam algo além da rubrica, a ablação decide. "Mais informação" não basta: o controle de ruído da
  ablação mostra por quê. A recoleta de 20/10 mediu a cauda da maturação (0% de 21 dias a anos).
- A mídia da amostra fechou (1.399 de 1.400, 0h15 de 21/10); a H9 (`601b2217`) e a ablação da amostra (14 runs,
  `554e0abb` o completo) rodaram em 2h20 (a estimativa era 10 min) e estão lacradas, com a H2 a H8 e a base toda.
- 21/10: ele percebeu que não leu nada depois do módulo 2 (as leituras dirigidas escorregaram dia a dia e o Claude
  não avisou). #29: o conceito mínimo da aula dentro de cada peça hoje, a leitura das aulas da apresentação em 22/10.
  #30: auditoria de conformidade e completude (`docs/AUDITORIA.md`): F1 a F15 até aqui; a segunda passada continua.
- Ficou para a sessão nova: terminar a auditoria, a F2 (frames e detectores, decisão dele), a A1 (o caso à mão
  espera as respostas) e o resto do dia no "Agora" do ROTEIRO.

## 19/10 (tarde e noite) · hipóteses H1, H2 e H4, régua dos blocos e base congelada

- O Vinicius escreveu o mecanismo da H1 (no feed o vídeo começa sozinho; assunto, duração e horário decidem quem fica),
  a H2 (o histórico ajuda pouco na ordem, porque é quase igual para A e B a até 14 dias, e ajuda no número: no
  Acelerados, a mediana de tudo dá 1.070 e os últimos 20 maduros, ~560) e a H4 (o texto ajuda na ordem; os embeddings
  só entram se ganharem das manuais). Decidiu a #27: o bloco fica se melhorar a manchete ou o `mae_log`, sem piorar a outra.
- Funcionou: os 6 Shorts reais do Acelerados (02 a 10/06/2025) em tabela; contextualizar o porquê do arquivo e do par
  controle × tratamento antes de perguntar; multiplicadores inventados para separar ordem e número. A confusão foi
  "histórico = mais dados para treinar" (é coluna, não linha).
- Do Claude: IC pareado do `mae_log` no `comparar`, base congelada v1 (`09f6035c11c9`), thumbnails e embeddings
  (30.981) e H7 lacrada (`f98c3e05`, falta o OpenCV). Mídia em 441/1.400; rubrica em ~973/1.400 (cota às 14h42).
  255 testes.
- Ficou para trás ~3,7 h do dia (cansaço à noite). Próxima sessão: a decisão do corte (#28), depois a H3 em diante.

## 18 e 19/10 (manhã) · M2, critério de sucesso e o pipeline do modelo

- Decididos: o bloco 1 (10 decisões, 18/10), a #6 (manchete pesando cada canal pelo nº de Shorts; a Salon Line tinha
  53% dos pares), a #5 (ganho ≥ +3 pontos com IC sem o zero, e IC sem o zero no canal novo), a #26 (o raciocínio é do
  Vinicius, o código é do Claude: o esqueleto com buraco não ensinava) e as decisões em duas mensagens. M2 e
  teach-back fechados: a mediana madura dá 49,7% (IC 49,2 a 50,1) porque empata A e B; a da 2.1 cai a 45,6% pela
  regressão à média.
- Do Claude: M3, runner com MLflow e M4 (alvo relativo, decisão de 05/10, depois de uma correção), texto, comparáveis
  (A5), SHAP, modelo de uso com paridade, golden set (30 Shorts de 6 canais de fora), CI, maturação (0% de
  crescimento depois de 21 dias), `auditar_datas`, a lista da #19, o pipeline de mídia da amostra (só do IP de casa)
  e uma revisão independente (sem vazamento; 13 correções com teste). 251 testes.
- Rodando: rubrica do VLM (819 de 1.400; o free tier corta em ~500 pedidos por dia) e thumbnails (8,9 mil de 31 mil).
  Os runs das H1 a H6 estão lacrados até as hipóteses (o último lote: `192a4f36` a `09275208`).
- Próxima sessão: o `HIPOTESES.md`, com os problemas da H3, da H5 e das H8/H9 antes; depois marco 1 e A1.

## 16 e 17/10 · M1 fechada e revisão do bloco

- A M1 está pronta: o Vinicius escreveu `split_treino_cal_teste` e `canal_novo_no_futuro`. Na base de
  desenvolvimento: treino 20.579, gap 1.025, calibração 1.882 e teste 0 (trava). No "canal novo no futuro" com a
  calibração, os grupos saem desiguais (180 a 679 Shorts; o grupo 3 perde 8.454 linhas de treino): reportar cada fold.
- Teach-back aprovado. A correção: os ajustes do modelo vêm do walk-forward dentro do treino; a calibração só mede os
  erros do modelo fechado, para o intervalo (M6). Na dúvida de sintaxe, funcionou dar a linha inteira de um bloco e
  deixá-lo escrever os outros; quatro reexplicações seguidas, com tabelas, não funcionaram.
- `/code-review` do bloco: M1 sem bug; corrigidos o fallback das duplicatas, o retry do download (httpx), o cache
  atômico e o `walk_forward` com a base inteira (`folds_de_desenvolvimento`). O Vinicius estendeu a regra das
  duplicatas para entre canais (aula 2.2). Base real: 314 cópias em 291 grupos, sem nenhuma mudança.
- Atraso de ~1 dia na Fase 1. Próxima sessão: o corte (decisão dele), depois a 2.4 e a M2.

## 14 e 15/10 · Decisões da Fase 1 e duplicatas

- Decididas a #1 (duplicatas por título, duração e thumbnail; controle: 668 pares de vídeos diferentes do mesmo canal
  têm dHash a distância mediana de 32 bits, e 99% ficam acima de 20; 10 pares da Principia conferidos à mão, todos
  certos), a #2 (treino desde 01/07/2021), a #3 (datas) e a #4 (canais fora da base para o golden set).
- 15/10: a liga antecipou a entrega para 21/10; corte #24 aplicado. As duplicatas foram implementadas: 307 cópias em
  285 grupos. Detalhes em `docs/DECISOES.md`. (Entrada escrita em 17/10, a partir do caderno de decisões.)

## 13/10 · Reta final: material mapeado e trabalho reorganizado

- O Claude leu o material da liga por inteiro: hub, as 47 aulas, RAG e EDD. O Vinicius vai ler tudo também, em duas
  velocidades. O mapa, com um cartão por aula e os
  conflitos, está em `docs/material/`.
- Os exercícios das aulas foram suspensos, porque custavam ~3 h cada, quase tudo de pandas. Agora o Vinicius escreve
  o coração das peças reais (M1–M4, M6, M8, M9, A1, A2 e A4–A8) e decide o resto; o Claude faz o encanamento. O calendário de 14 a 23/10
  está no ROTEIRO.
- O contexto foi reestruturado: CLAUDE.md enxuto; ROTEIRO, DECISOES e DIARIO; skills `/aula`, `/teach-back` e
  `/fechar-sessao`; `.claude/settings.json`. O clone fixo do material fica em `~/autoestudos`.

---

## Histórico até 12/10 (movido do CLAUDE.md em 13/10, sem cortes)

### Estado (como estava no CLAUDE.md)

- 30/09: material da liga adotado. Código da base pronto e testado (coletor, curadoria e esquema mínimo; 64 testes),
  seguindo o plano `docs/superpowers/plans/2026-09-30-base-de-shorts.md` (decisões da execução no apêndice dele).
  Falta a chave da API para avaliar os 112 candidatos, fazer a revisão dos canais, a primeira coleta e instalar o
  agendamento (Task 7 do plano).
- 05/10: chave da API criada e coleta diária agendada (launchd, 5h15). Primeira coleta dos 32 canais: 44.494
  vídeos, 10.949 Shorts, 6.772 rotulados. Revisão humana (aula 1.4): 52 canais incluídos (Localiza fora pelo
  critério 1; Garagem do Bellote dentro após contagem completa). Repositório privado criado. Estudo: Conceitos
  básicos de LLM e aulas 1.1 a 1.4 feitas (problema no README, exercício da 1.2 em `estudos/`, "o que a base não
  representa" no plano da base).
- 08/10: expansão de canais delegada ao Claude. Comando `buscar` (estratégia 4 da aula 1.4, modos canal e vídeo):
  24 termos, 614 candidatos, 18 incluídos na revisão. Base: **70 canais, 35 por setor**; criadores de beleza de 7
  para 20. Funil e decisões no `docs/PLANO_DA_BASE.md` ("Expansão por busca"). Correção de rede: `preferir_ipv4()`
  (a rota IPv6 até o Google caiu). Commitado (3 commits).
- Próximo (09/10): decisões do plano ainda abertas (teto de 400, motos). **Módulo 1 fechado em 08/10:**
  `docs/DATA_CARD.md` v0.1 com os vieses do Vinicius; a coleta diária valida a base mínima com pandera, audita a regra dos 30 dias e
  desliga em 30/10 (apagamento). Pendente até 30/10: inscritos da API versionados em `data/canais.csv`. Para o
  módulo 3: experimento sobre a idade do rótulo (variável de controle × janela da mediana), viés 3 do data card.
  Próximo estudo: 2.1.
- 08/10 (noite): conferência do material da liga contra o repositório (módulos 1 e 2 inteiros, hub, trechos de 3 a
  6). Feito: ambiente do módulo 2 (lightgbm, mlflow, pyyaml; Homebrew 7.0.9 e `libomp` instalados; os passos 2
  a 4 da aula 2.1 reproduzem os números da aula na base sintética); exportação
  **sem teto** (22.172 Shorts; o teto vira parâmetro do experimento); coluna `duplicate_group` (573 grupos, 974
  cópias, concentradas em Principia e Mundo do Automóvel para PCD, com títulos de código de anúncio); trava do
  teste em `configs/avaliacao.yaml` (provisória: desde 01/04/2026); `previa.sintetico.gerar_base()` para os
  exercícios do módulo 2. O que falta está na tabela "Pendências da conferência", abaixo.
- 09/10: estudo das métricas da 2.1 (MAE em log, Spearman global e intra-canal), com teach-back em
  `estudos/aula_2_1_metricas.md`. O que funcionou: construir a "régua" do log com curtidas (cada dobra = +0,69) e
  o exemplo de dois canais (modelo que só sabe o canal: global 0,88, intra 0). Próximo: exercício da 2.1.
- 11/10: coletas de 09 a 11/10 sem falha (70 canais, até 851 unidades de cota por dia). A primeira coleta dos 18
  canais novos trouxe 9.027 Shorts, e cada um deles tem pelo menos 106 rotulados. Base: 30.879 rotulados (beleza
  16.791, automotivo 14.088), dos quais 23.822 ficam antes da trava do teste. Exercício da 2.1 feito na base
  sintética (`estudos/aula_2_1_baselines.py`), porque na nossa a receita da aula vaza (posts imaturos) e o passo
  3 abriria o teste. Resultados: a validação escolheu k=10; nas mesmas linhas, o modelo do passo 4 perde no MAE
  para a `media_ult10` (0,711 × 0,686, −3,6%) e ganha no intra-canal (+0,16 × −0,18). A primeira tentativa dele
  usou só os 3 baselines como features (−8,5%): o que faltava era informação, e o overfitting explicava só uma
  parte. A ideia dele de usar o nível do baseline e a ordenação do modelo leva ao alvo relativo (aula 3.5).
  A tabela de baselines da nossa base vem depois da 2.3, com o split definitivo e a regra de maturidade.
- 12/10: 2.2 feita. A auditoria está em `docs/AUDITORIA_FEATURES.md`; a confusão entre treino e produção se resolveu
  quando ficou claro que a coluna "quando existe" fala do valor que está na base. O item 2, na sintética, reproduz
  a aula (madura 0,807 × `shift(1)` 0,788 de MAE). Teach-back ok. Para explicar como o modelo funciona em produção,
  funcionaram três coisas: o pedido da Lumina em passos, o laudo de imóvel ("treinado uma vez; os dados dela entram
  como entrada") e uma árvore de 2 níveis com folhas como multiplicadores da mediana. Falta a regra das duplicatas:
  na base de 70 canais são 644 grupos, com 1.089 cópias extras, e a Principia tem 920 linhas em grupos, com títulos
  de código como "CH 01". **Próxima sessão: começar pela regra das duplicatas** (explicar com exemplos da base →
  decisão dele), depois a 2.3. Ele ainda não respondeu como dividir o pandas nos exercícios (eu monto o pandas e
  ele fica com as decisões em `TODO(human)` × ele escreve tudo); a 2.2 ele fez copiando a solução da aula.
- Ritmo combinado em 08/10 (~1 semana atrás no núcleo de ML): ~2 aulas por dia, cada uma com a peça implementada.
  08–09/10 módulo 1; 10–13/10 módulos 2 e 3 → **marco: GBM batendo o baseline no teste temporal até 13/10**;
  14–16/10 `prever_criativo` + MCP mínimo; 17–20/10 intervalo, SHAP e multimodal numa amostra; 21–23/10 docs e
  demo. Fora: VLM (camada 2), LangGraph completo, módulo 7. Se o marco escorregar, corta-se o multimodal.
  Em 12/10, com a 2.1 e a 2.2 fechadas, o marco de 13/10 vai escorregar: cada exercício levou ~3 h, a maior parte
  aprendendo pandas. Replanejar com ele quando o módulo 2 fechar.

### Pendências da conferência de 08/10 (as abertas foram para o ROTEIRO)

| Pendência | Quando | Quem | Detalhe |
| :--- | :--- | :--- | :--- |
| Teto e peso dos canais nas métricas intra-canal | na 2.4 | Vinicius | num teste desde 01/04, Mobiauto, Mundo do Automóvel para PCD e Principia somam 61% dos pares do par-a-par (79% na janela de 14 dias); os 3 não têm linha no treino |
| Regra das duplicatas | **início da próxima sessão**, antes da 2.3 | Vinicius | `duplicate_group` já marca; manter uma cópia ou deixar as cópias do mesmo lado do split (aula 2.2). Principia e Mundo do Automóvel para PCD têm títulos de código de anúncio: decidir por critério escrito (aula 1.4), nunca pelas curtidas |
| Data de início do treino | na 2.3 | Vinicius | 236 rotulados são de antes de set/2020 e 273 de antes de jul/2021: vídeos verticais curtos, não Shorts |
| Datas definitivas do teste, do gap e da calibração | na 2.3 | Vinicius | ajustar `configs/avaliacao.yaml`; a trava atual (01/04/2026) é provisória |
| Congelar a base de treino | antes do primeiro experimento na base real (depois da 2.3) | Claude | versão e hash (aula 2.5); era 12/10, mas até aqui só houve exercícios na sintética |
| Conferir 50 classificações de Short a olho | até 16/10 | Claude monta a lista; Vinicius confere | anotar a concordância no data card (aula 1.5) |
| Mediana madura no código das features | módulo 3 | Vinicius | princípio 2; o tamanho da lacuna sai da maturação (aula 3.5) |
| Medir a maturação | 19 a 26/10, antes de 30/10 | Claude mede; Vinicius escolhe a lacuna | 2 a 3 semanas entre snapshots (aula 1.7); hoje há só 1 a 3 dias, então o "confirmado" do data card ainda não vale |
| `HIPOTESES.md` | na 2.5 | Claude monta o modelo; Vinicius escreve | hipótese com métrica e limiar antes de rodar |
| README: porquê do alvo e "posts anteriores **e maduros**" | no exercício da 2.2 | Vinicius | o texto é dele |
| README: critério de sucesso | na 2.3 | Vinicius | incluir "canal novo no futuro" e a cobertura do intervalo |
| Convenção de faixas | na 2.4 | Vinicius | quantis do canal (aula 1.6) × cortes fixos (aula 2.4) |
| Exercício da 1.7 (`auditar_datas`) | junto com o da 2.2 | Vinicius | a tabela já existe (`docs/AUDITORIA_FEATURES.md`, 12/10) e diz o mapa honesto de cada feature; falta escrever a função |

