# Módulo 1: Formulação e dados (8 aulas, 350 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/1-formulacao-e-dados/`. Estudado até 08/10.

O módulo entrega ao projeto:

- a base rotulada;
- o alvo em log, o alvo relativo e as faixas;
- as decisões sobre população e vieses;
- o coletor que a tool de histórico reaproveita.

Legenda e regras de uso: [README](README.md). Conflitos entre aulas: [conflitos](conflitos.md).

## 1.1 Do LLM ao modelo preditivo

35 min · AI Eng (arquitetura) · leitura feita · prática: feita (README)

- **Objetivo.** Mostrar por que o agente não pede o número ao LLM, que dá uma resposta plausível mas sem régua. Os
  papéis ficam divididos: o modelo treinado prevê; o LLM orquestra, extrai atributos e explica.
- **Conceitos.**
  - Gerar × prever.
  - Supervisionado é f(x)≈y em dados nunca vistos, que aqui são o futuro.
  - Uma feature só vale se existe no instante da previsão.
  - O alvo é log1p(curtidas) ou o log da razão com a mediana dos posts anteriores do canal. As faixas são derivadas
    da previsão, sem segundo modelo.
  - Metas realistas: ordenar bem e bater o baseline. Cerca de 60% de acerto par-a-par prova o conceito; o Spearman
    esperado fica entre 0,55 e 0,85; o conteúdo visual soma de +0,02 a +0,04.
- **O LLM aparece em 3 lugares:** orquestrador, extrator de atributos (VLM com rubrica) e subagente SHAP→texto.
- **Constrói.** A arquitetura de referência: 5 tools, orquestrador em LangGraph, função Python e servidor MCP. Cita
  Kayal et al. (2025): LLM zero-shot 82% contra 80% do supervisionado, e 85,5% combinados.
- **No projeto Galaxies.** O caso Lumina Beauty: cerca de 180 mil inscritos e mediana de cerca de 2.300 curtidas em
  90 dias.

  | | Short A, "Pele natural em 5 passos" | Short B, "1 produto, 20 segundos" |
  | :--- | :--- | :--- |
  | Duração | 55 s | 22 s |
  | Abertura | logo | antes/depois, meio rosto sem maquiagem |
  | Som | voiceover calmo | música animada |
  | Texto na tela | não | grande |
  | Publicação | terça, 14h | quinta, 19h |

  A pergunta é qual dos dois impulsionar, e com que confiança. O caso secundário é o Garagem 22.
- **Erros comuns.**
  1. Começar pelo agente.
  2. Deixar o LLM produzir o número "só por enquanto".
  3. Perseguir acurácia absoluta.
  4. Esquecer o "antes da publicação".
- **Exercício.** O problema no README em até 10 linhas: decisão, unidade, momento, alvo e métrica contra o baseline.
  → Feito (README, "O problema").
- **Liga com.** Conceitos de LLM (funcionamento, function calling); 1.6; 2.2; 7.3.

## 1.2 Anatomia de um problema supervisionado

45 min · ML · leitura feita · prática: feita

- **Conceitos.**
  - Cinco peças: unidade, features, alvo, modelo e perda.
  - MSE aprende a média, MAE a mediana, Huber fica entre os dois.
  - Treino ajusta, validação decide, teste é aberto uma vez.
  - O erro se decompõe em viés, variância e ruído irredutível.
  - Dados temporais não são i.i.d.
  - Em log, um resíduo de 0,5 é um fator de 1,65.
- **Constrói.** O ambiente do módulo 1 e uma grade da constante ótima por perda. Também uma árvore por profundidade:
  a melhor é 6; sem limite, treino 0,000 e validação 0,731. No LightGBM: `regression` (MSE), `regression_l1` e `huber`
  (o `alpha` é o δ).
- **No projeto Galaxies.** A e B são linhas. Só com features de canal, o modelo dá o mesmo número para os dois, o que
  é um diagnóstico e não um bug. O split aleatório promete uma precisão que A e B não terão.
- **Erros comuns.**
  1. Reportar o erro de treino.
  2. Olhar o teste várias vezes.
  3. Escolher a perda pelo nome da biblioteca.
  4. Sortear linhas para validar.
  5. Achar que dá para bater o ruído.
- **Exercício.** "Faça o split aleatório mentir": aleatório 0,567 × temporal 0,710. → Feito
  (`estudos/aula_1_2_paradoxo.py`).

## 1.3 Fontes de dados e plano da base

35 min · dados · leitura feita · prática: feita

- **Conceitos.**
  - A API é uma foto do dia da coleta: não traz retenção, CTR nem compartilhamentos.
  - SMTPD (sobretudo long-form), SMP-Video e MicroLens são fontes de apoio, nunca de teste.
  - A cota é de 10.000 unidades por dia e não acumula.
- **Constrói.**
  - `custo_coleta` e `data_de_corte` (coleta − 30 dias).
  - Metas: 5 a 15 mil Shorts rotulados, 60+ canais, pelo menos 2 setores, rótulo com 30+ dias e `collected_at` em
    toda linha.
  - `search.list` tem 100 chamadas por dia; `videos.batchGetStats` tem balde próprio.
- **No projeto Galaxies.** Para a Lumina, a base precisa de:
  - canais parecidos com ela (beleza de porte médio, marcas oficiais);
  - variedade de formato e de horário;
  - um segundo setor.
- **Erros comuns.**
  1. Trocar a sua base por uma pronta maior.
  2. Misturar a fonte de apoio no teste.
  3. Começar a coletar na semana 3.
  4. Esquecer `collected_at`.
  5. Aceitar a licença sem ler.
- **Notas.**
  - A cota zera às 04h de Brasília até 01/11 e às 05h depois.
  - A playlist de uploads mistura vídeos longos e Shorts.
  - A base real tem cerca de 2× a meta, e acima de 15 mil "o gargalo vira a extração multimodal".
- **Exercício.** `PLANO_DA_BASE.md`. → Feito (`docs/PLANO_DA_BASE.md`).

## 1.4 Curadoria de canais

40 min · dados · leitura feita · prática: feita (delegada ao Claude em 08/10)

- **Conceitos.**
  - Critérios verificáveis: setor; idioma e país; pelo menos 1 Short em 90 dias; 30+ Shorts; `likeCount` visível;
    inscritos não ocultos. Ficam fora cortes, reuploads, agregadores e conteúdo infantil.
  - Estratificar por setor × faixa de inscritos (<10 mil, 10–100 mil, 100 mil–1 mi, >1 mi) × tipo (marca ou criador).
  - Teto por canal, por exemplo os 400 Shorts mais recentes.
  - Viés de sobrevivência.
  - O `subscriberCount` vem arredondado a 3 algarismos.
- **Constrói.** `curadoria.py` (`resolver_handles`, `atualizar_em_lote`, `handles_em_textos`, `canais_em_destaque`,
  `estratificar`) e a planilha de 14 colunas, incluindo `fonte` e `motivo`.
- **No projeto Galaxies.** A célula "beleza × 100 mil–1 mi × marca" precisa ter canais. Se estiver vazia, o módulo 5
  acende a baixa confiança. Teste de sanidade: um canal fictício por faixa, e o intervalo deve ficar mais largo onde a
  base é rala.
- **Erros comuns.**
  1. Curar pelo que você conhece.
  2. Decidir os critérios depois de ver as curtidas.
  3. Gastar `search.list` para listar vídeos.
  4. Confiar em `snippet.country`.
  5. Não registrar os excluídos.
- **No Prévia.** `src/previa/curadoria.py`; `data/canais.csv` com 70 canais, 35 por setor. O teto de 400 virou
  parâmetro do experimento.

## 1.5 Construindo o coletor

60 min · dados (encanamento) · leitura feita (rápida) · prática: feita

- **Conceitos.**
  - A playlist de uploads é o ID do canal com `UU` no lugar de `UC`.
  - Páginas e lotes de IDs têm 50 itens, e cada chamada custa 1 unidade.
  - Não existe `isShort`. A regra é vídeo vertical ou quadrado com até 3 min desde 15/10/2024 (60 s antes), com
    +1 s de folga.
  - `likeCount` ausente vira `None`, nunca 0.
  - Retry só em erro transitório; `quotaExceeded` para a coleta.
- **Constrói.**
  - `Cliente`, com cota no fuso de Los Angeles, limite de 9.000 e tenacity.
  - `parse_video`.
  - DuckDB com `videos` (upsert) e `snapshots` (chave `(video_id, collected_date)`).
  - `exportar_rotulados`, que pega o primeiro snapshot com 30+ dias.
  - Cron às 05h.
- **No projeto Galaxies.** O coletor é metade da tool de histórico do módulo 6. Short horizontal não é pontuado.
  Curtidas ocultas viram nulo, sem inventar mediana.
- **Erros comuns.**
  1. Listar vídeos com `search.list`.
  2. Insistir em `quotaExceeded`.
  3. Transformar curtida ausente em zero.
  4. Anexar em vez de fazer upsert.
  5. Não guardar `collected_at` e `age_days`.
  6. Confiar no truque não documentado.
- **Exercício.** 3 testes de casos-limite e a checagem manual de cerca de 50 classificações de Short para o data card.
- **No Prévia.** Coletor pronto (`src/previa/coletor.py`, launchd às 5h15). **Pendente:** conferir 50 classificações.
  O Claude monta a lista e o Vinicius confere até 16/10.

## 1.6 Engajamento é cauda longa

45 min · ML · leitura feita · prática: feita

- **Conceitos.**
  - Curtidas são aproximadamente log-normais: resuma pela mediana.
  - Em log1p, 0,1 ≈ 10% e 0,69 ≈ 2×.
  - Alvo relativo: `y_rel = log((likes+1)/(mediana dos anteriores+1))`, com histórico mínimo de 10 posts.
  - Janela expansiva × janela de 90 dias.
  - Faixas por quantis do canal: abaixo (<p25), na média (p25–p75), acima (p75–p95) e viral (>p95).
- **Constrói.** `alvo_relativo(df, min_hist=10)` com `groupby` + `shift(1)` + `expanding(min_periods=10)`. A versão
  vazada usa `transform("median")`: +0,23 honesto contra −0,28 vazado.
- **No projeto Galaxies.** A pergunta é qual criativo supera mais a mediana recente do canal. A faixa comunica, mas não
  substitui o intervalo.
- **Erros comuns.**
  1. Resumir pela média.
  2. Treinar em curtidas brutas com MSE.
  3. Calcular a mediana com a base inteira.
  4. Esquecer o `shift(1)`.
  5. Tratar "viral" como número absoluto.
- **Exercício.** Feito (`estudos/aula_1_6_janelas.py`). **Pendente:** a convenção de faixas, 1.6 × 2.4 (ver
  [conflitos](conflitos.md)).

## 1.7 O tempo mente

45 min · ML · leitura feita · prática: Claude (`auditar_datas`)

- **Conceitos.**
  - Maturação: os 30 dias são convenção, a verificar na própria base.
  - Idade do rótulo variável: aceitar, controlar (feature no treino, fixada em 30 na previsão) ou restringir.
  - As views mudaram de regra em 31/03/2025 e em 24/08/2026, por isso o alvo são curtidas.
  - Inscritos, título e descrição são a versão do dia da coleta.
  - Drift não é sazonalidade.
  - Hora e dia da semana vão no fuso local.
- **Constrói.** `anotar_tempo` (`age_days`, `rotulavel`, regimes R0/R1/R2, `hora_local` via `tz_convert`) e
  `auditar_datas(df, observado_em)`.
- **No projeto Galaxies.** Os inscritos de hoje são legítimos para prever A e B em produção, mas não no treino
  (training-serving skew). "Terça 14h" é horário de Brasília. A previsão vale para as curtidas aos 30 dias.
- **Erros comuns.**
  1. Rotular Shorts jovens.
  2. Usar views como alvo.
  3. Usar os inscritos da coleta como feature de posts antigos.
  4. Calcular hora e dia em UTC.
  5. Ignorar a idade do rótulo.
- **No Prévia.**
  - A tabela de auditoria está feita (`docs/AUDITORIA_FEATURES.md`).
  - `auditar_datas` é do Claude, na Fase 1.
  - A maturação o Claude mede em 19 a 22/10, e o Vinicius confirma a lacuna.
  - **Pendente:** a idade do rótulo (viés 3 do data card), decisão #21 do ROTEIRO. A escolha é entre variável de
    controle fixada em 30 e restringir a idade, decidida por experimento na M3/M4.

## 1.8 Documentação, ética e termos

45 min · produto · leitura feita · prática: feita

- **Conceitos.**
  - Data card de 11 seções; a principal é "o que esta base NÃO representa".
  - Esquema mínimo de 8 campos: Pydantic valida o registro, pandera valida a tabela.
  - Políticas da API:
    - III.E.4.d: até 30 dias para dados não autorizados;
    - III.E.4.h: métricas derivadas;
    - III.E.1: não guardar vídeo;
    - III.E.6: scraping.
  - No free tier do Gemini, o conteúdo pode ser usado pelo Google e lido por revisores.
  - LGPD: detectar se há rosto pode; reconhecimento facial não.
- **Constrói.** `DATA_CARD.md`, `ChannelStats`, `Post`, `BaseShorts`, `auditoria_de_retencao` e `apagar`, além de um
  checklist de 12 itens.
- **No projeto Galaxies.** A e B inéditos são confidenciais, então nada de free tier. A resposta é transparente:
  "estimativa do modelo, treinado em N Shorts de M canais...". Os limites do data card alimentam a baixa confiança.
- **Erros comuns.**
  1. Publicar a base.
  2. Guardar snapshots indefinidamente.
  3. Tratar métricas derivadas como inofensivas.
  4. Mandar criativos de cliente para o free tier.
  5. Validar só no primeiro dia.
- **No Prévia.**
  - `docs/DATA_CARD.md` v0.1 e `src/previa/esquema.py`.
  - A coleta valida com pandera e audita a retenção.
  - Os dados brutos são apagados em 30/10.
