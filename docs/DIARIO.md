# Diário do projeto

Histórico das sessões. As entradas novas ficam no topo, com 3 a 5 linhas por sessão (`/fechar-sessao`). O plano
vivo está em `docs/ROTEIRO.md` e as decisões em `docs/DECISOES.md`.

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

