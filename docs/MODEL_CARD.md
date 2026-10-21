# Model card: Prévia (rascunho de 21/10/2026)

> Rascunho do Claude, a revisar pelo Vinicius. Segue o template do hub (`projetos/galaxies/templates`), a aula 6.5
> (o recurso `model://card`: "qual modelo é esse, medido como, e onde ele não funciona"), a 6.7 (checklist de
> entrega) e a 7.4 (registrar a licença de cada checkpoint). O formato é o de Mitchell et al. (2019),
> [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993). Será servido pelo servidor MCP como
> `model://card`. Os números têm fonte; o que depende de run lacrado, do teste (abertura única) ou de uma peça ainda
> por fazer fica como `PREENCHER:`, com o que vai ali.

## Modelo e versão

- **Nome e tarefa:** Prévia, regressão do alvo relativo ao canal para um YouTube Short de beleza ou automotivo,
  antes da publicação, com intervalo de 80%, ranking entre criativos e explicação.
- **Algoritmo:** LightGBM 4.7.0 (`pyproject.toml`), regressão. Parâmetros de partida da aula 3.4: `learning_rate`
  0,05, 15 folhas, `min_data_in_leaf` 20, `feature_fraction` 0,9, `bagging_fraction` 0,8 com `bagging_freq` 1,
  `lambda_l2` 1, seed 0, early stopping nos últimos 180 dias do treino (paciência 100, até 5.000 árvores), refit com
  o número de árvores escolhido (`configs/experimentos/opcao_a_texto.yaml`, README seção 5). A perda é L2 em todas as configs
  até agora; a Huber (δ = 1) foi testada como variante (H11, run `bfba1020`, lacrado) e a perda final se decide na
  abertura (decisão #32). <!-- PREENCHER: a perda do modelo final, pelo resultado da H11 -->
- **Intervalo:** quantis 0,1, 0,5 e 0,9 com ajuste conformal (CQR), calibrados por data (peça M6 do ROTEIRO, aulas 5.2
  e 5.3). O número da tool sai do q50 da M6 (decisão #32 registra essa frase; o desenho do intervalo é da M6, ainda
  por fazer). <!-- PREENCHER: confirmar o q50 como número da tool e os hiperparâmetros dos quantis e o q̂ da calibração, da M6 -->
- **Versão em produção:** <!-- PREENCHER: `gbm-<nome>-<data>-<hash>` e o run id do MLflow do modelo de uso -->
- **Qual modelo serve (decisão #33, 21/10):** o teste final mede o modelo treinado até 01/01/2026 (fim exclusivo em 02/01; `configs/avaliacao.yaml`) na base congelada
  v1, uma vez. O modelo de uso é a mesma receita (features e config) retreinada numa v2 da coleta diária, com Shorts
  até ~jun/2026, e o intervalo recalibrado numa janela recente e madura fora desse treino (~jul a set/2026). Se às
  22h a v2 não estiver pronta, a entrega sai com o modelo de janeiro. Este card vale para os dois casos:
  <!-- PREENCHER: qual dos dois foi entregue, o fim do treino e da calibração dele, e o run id -->
- **Reprodução:** `uv run python -m previa.experimento configs/experimentos/<config>.yaml` registra no MLflow
  (`previa-curtidas`) o commit, o hash da base, a config e as previsões fora da amostra.
- **Responsável e contato:** Vinicius Castiglia, repositório `case-galaxies` (privado).

## Uso pretendido

- **Quem usa:** o time de marketing de uma marca, via a persona da Galaxies (Agente Nexus), ou um cliente MCP.
- **Para quê:** comparar criativos de um canal com histórico, antes de publicar, para decidir qual Short impulsionar
  (README, seção 1). O Prévia entrega a ordem entre os criativos (P(A > B)) e o número esperado de curtidas com o
  intervalo de 80%.
- **Unidade:** um Short de um canal já existente, com ao menos 10 Shorts publicados e maduros (30+ dias) antes.
  Com menos de 10 a previsão sai com o flag `canal_novo`; com menos de 3 posts maduros, a tool dá um número pela
  cadeia de cold start da aula 2.1, com intervalo largo e o texto dizendo que é uma referência (decisões #20 e #34).

## Fora de escopo (uso não pretendido)

- Vídeos longos, outras plataformas, outros setores além de beleza e automotivo, outros países e idiomas além de
  canais brasileiros em português.
- Prever views, retenção, CTR ou impressões. O alvo é curtidas.
- Decisão automática sem revisão humana. A saída é uma estimativa do modelo, não uma garantia.
- Canais sem histórico tratados como se o número fosse firme: o `confidence_flag` tem de ir junto.
- Simular personas ou dar nota de 0 a 5 (fronteira com a Galaxies).
- Explicação causal: os fatores dizem o que o modelo aprendeu, não o que causa curtidas.
- Usar o número como produzido pelo LLM: o LLM orquestra, extrai atributos e redige; o número vem sempre do modelo.

## Dados de treino e corte por data

Data card completo: [`DATA_CARD.md`](DATA_CARD.md). Base congelada v1 (`data/base_v1.parquet`, exportação de
19/10/2026, 06h46, hash de desenvolvimento `09f6035c11c9`).

| Bloco | Datas de publicação | Shorts | Uso |
| :--- | :--- | ---: | :--- |
| Treino | 01/07/2021 a 01/01/2026 | 20.579 | ajuste; walk-forward de 2025 para as escolhas |
| Gap | jan/2026 | 1.025 | maturidade do alvo e contágio na fronteira |
| Calibração | 01/02 a 31/03/2026 | 1.882 | o intervalo (CQR) |
| Teste | 01/04 a 14/09/2026 | 7.265 | uma vez, no fim (`carregar_base(final=True)`) |

- **Volume:** 31.303 Shorts rotulados de 70 canais (35 por setor), 14.290 de automotivo e 17.013 de beleza; 320 cópias
  entre 30.983 criativos (só a primeira cópia vira linha de treino e de teste). Fonte: DATA_CARD.md.
- **Antes de jul/2021:** 336 Shorts ficam só no histórico do canal (decisão #2).
- **Modelo de uso (v2):** <!-- PREENCHER: contagem de Shorts e canais da v2, e o fim do treino, se a v2 for entregue
  (decisão #33) -->
- **Termos dos dados:** YouTube API Services Terms of Service e Developer Policies. Uso educacional e interno,
  sem redistribuição. Os dados brutos são apagados em 30/10/2026 (DATA_CARD.md, "Licença e termos").

## Alvo

`y_rel = log1p(curtidas) − mediana madura do canal`, com 10+ posts maduros (decisões #20 e de 19/10). A mediana
madura usa só posts publicados ao menos 30 dias antes, calculada sobre todos os posts do canal. A previsão em
curtidas é a saída do modelo somada a essa mediana, de volta da escala log. O rótulo é o `likeCount` do primeiro
snapshot com 30 dias ou mais de idade. Curtidas ocultas viram `None`, nunca 0 (3,7% da base de desenvolvimento).

## Features

Prefixo por bloco (aula 4.8). Fontes: README seção 5 e `configs/experimentos/`.

| Bloco | Exemplos | Onde existe |
| :--- | :--- | :--- |
| Histórico do canal | `hist_n`, `hist_mediana`, `hist_med5`, `hist_med20`, `tendencia`, só com posts maduros | base toda |
| Calendário e canal | `hora_local`, `dia_semana_local`, `feriado`, `idade_canal_dias` | base toda |
| Metadados | `duracao_s`, `setor` | base toda |
| Texto do título | 10 manuais (`txt_*`) + PCA de 16 do MiniLM | base toda |
| Capa | PCA de 16 do SigLIP 2, 10 sondas, brilho, contraste, saturação, colorfulness (`thumb_*`) | base toda |
| Rosto e texto na capa | `det_*` (YuNet e o detector de texto PP-OCRv3 do opencv_zoo; decisão #31) | base toda, quando a H10 for medida |
| Rubrica do VLM | gancho, energia e ritmo, rosto, produto em close, antes/depois, texto na tela (`vlm_*`) | amostra de 1.400 |
| Vídeo e áudio | cortes, movimento e aparência dos frames, fala e energia do áudio | amostra de 1.400 |

<!-- PREENCHER: quais blocos entraram no modelo final e o total de features, pela ablação e pelas H2 a H10 (lacradas);
a quantidade por bloco no formato do template do hub -->

Fora do modelo, de propósito: inscritos (são os do dia da coleta, 1.7), views, curtidas e comentários do próprio
vídeo, o idioma e a transcrição do Whisper (decisão #35, registrada como limitação). O texto na capa é só detectado
(tem ou não, quantas caixas, área), não lido (DECISOES, 21/10, "Texto na capa sem o PaddleOCR"). Os PCAs e as escalas são ajustados só no treino. A mesma função calcula cada feature no
treino e na previsão.

## Métricas

Régua (decisões #5, #6, #27): a manchete é o par-a-par intra-canal entre Shorts do mesmo canal publicados a até 14
dias um do outro, com cada canal pesando pelo número de Shorts dele; ao lado, o `mae_log` (quanto o número erra, em
log), o Spearman intra-canal e a cobertura do intervalo. Todo IC 95% é bootstrap de canais inteiros, pareado quando
compara dois modelos.

**Critério de sucesso (decisão #5):** o ganho do GBM sobre a mediana madura na manchete ser de ao menos +3 pontos,
com IC sem o zero, no teste temporal e também no "canal novo no futuro". Os ~60% da Galaxies são referência.

**Baselines (decisão #6, walk-forward de 2025, janela de 14 dias):**

| Baseline | Manchete | IC 95% |
| :--- | ---: | :--- |
| Mediana madura do canal | 49,7% | 49,2% a 50,1% |
| Mediana da aula 2.1 (com posts imaturos) | 45,6% | 44,0% a 46,7% |

Média dos últimos 10 posts maduros: manchete de 0,501, empate estatístico com a mediana madura (decisão de 21/10,
F25).

**Teste temporal (abertura única, 01/04 a 14/09/2026, 7.265 Shorts):**

| Medida | Mediana madura | Prévia | Ganho [IC 95%] |
| :--- | ---: | ---: | ---: |
| Manchete (par-a-par, 14 dias) | <!-- PREENCHER: do teste --> | <!-- PREENCHER --> | <!-- PREENCHER --> |
| `mae_log` | <!-- PREENCHER --> | <!-- PREENCHER --> | <!-- PREENCHER --> |
| Spearman intra-canal | <!-- PREENCHER --> | <!-- PREENCHER --> | <!-- PREENCHER --> |

Run id e comando: <!-- PREENCHER: run id do MLflow da H1 e o comando que o reproduz -->

**Canal novo no futuro** (4 grupos de canais fora do treino, testados em 2025): <!-- PREENCHER: manchete, `mae_log`
e ganho com IC, do runner -->

**Ablação por blocos** (aula 4.8, walk-forward de 2025): <!-- PREENCHER: a tabela de ablação, lacrada -->

**Por setor e por faixa de inscritos:** <!-- PREENCHER: manchete e `mae_log` em beleza e automotivo, e por faixa
(só com vídeos dos últimos 90 dias, pelo viés dos inscritos) -->

**Golden set** (30 Shorts de 6 canais de fora da base, decisão #4, de 01/04 a 14/09/2026): <!-- PREENCHER: cobertura
e par-a-par -->

**Onde o modelo erra:** <!-- PREENCHER (M9): os 20 piores erros, classificados -->

## Cobertura do intervalo

Intervalo nominal de 80%. Critério da M6: cobertura de 80% ± 3 pontos em dados permutáveis (ROTEIRO). A calibração
é por data, em fev e mar/2026 para o modelo de janeiro (1.882 Shorts); no modelo de uso da v2, em ~jul a set/2026
(decisão #33). A garantia conformal é marginal: vale na média, não por canal (aula 5.3).

| Recorte | Cobertura | n |
| :--- | ---: | ---: |
| Teste, total | <!-- PREENCHER: do teste --> | 7.265 |
| Teste, beleza | <!-- PREENCHER --> | <!-- PREENCHER --> |
| Teste, automotivo | <!-- PREENCHER --> | <!-- PREENCHER --> |
| Canal com menos de 10 posts maduros | <!-- PREENCHER --> | <!-- PREENCHER --> |
| Modelo de uso (v2), na janela de calibração | <!-- PREENCHER: se a v2 for entregue (#33) --> | <!-- PREENCHER --> |

## `confidence_flag`

Conjunto de regras explícitas, cada uma com código e frase (aula 5.4); as regras abaixo são candidatas, não finais. A decisão #14 (regras e
limiares) ainda está aberta no ROTEIRO. Estão decididos só o `canal_novo` (#34) e a regra de data contada do fim do
treino (#33). Proposta, não decisão: o critério da aula, de o erro médio dos casos sinalizados ser maior que o dos
não sinalizados no teste temporal, e a taxa-alvo de disparo de 15% a 35% (ROTEIRO, #14).
Regras candidatas (a proposta do ROTEIRO são as regras 1, 2 e 6, mais a 4 com embeddings; as regras 3 e 5 da aula
não estão nela):

1. intervalo largo (largura em log acima do percentil 80 de referência, fixado na calibração);
2. canal novo (menos de 10 posts maduros);
3. setor fora do treino;
4. criativo fora da distribuição (distância ao vizinho mais próximo, limiar no percentil 99 da calibração);
5. extração falhou (VLM ou vídeo);
6. data de publicação fora do período: mais de 180 dias depois do fim do treino (decisão #33: conta do fim do
   treino do modelo servido).

<!-- PREENCHER: as regras e os limiares finais (decisão #14, ainda aberta no ROTEIRO), a taxa de disparo no teste e o
erro médio dos casos sinalizados contra os não sinalizados -->

Com o modelo de janeiro, a regra 6 acenderia em todas as previsões de hoje; por isso a #33 manda servir a v2 se
ela estiver pronta às 22h; senão, a entrega sai com o modelo de janeiro e o README diz por quê. Nos dois casos, o
card diz com todas as letras: o envelhecimento do modelo vale para toda previsão e é limitação, não só
flag (decisão #33). Retreinar a cada ~3 meses é a recomendação (decisão #33).

## Limitações e vieses

Resumo; o detalhe está no DATA_CARD.md ("O que esta base NÃO representa" e "Vieses") e no README (seção 8).

- **Canais novos e pequenos:** o alvo relativo depende de 10+ posts maduros. Abaixo disso, flag `canal_novo` e
  intervalo largo; esse regime não foi avaliado (decisão #34).
- **Idade do rótulo:** na base de desenvolvimento, a mediana é de 636 dias e 85% dos rótulos foram medidos com mais de
  300; no teste, de 1 a 6 meses. O modelo aprende curtidas de vida inteira e é avaliado nas de poucos meses (H5).
- **Inscritos:** só a foto do dia da coleta; ficam fora das features, e a análise por faixa pode estar com a faixa
  errada.
- **Mix de setor e tipo:** no automotivo, 83% dos Shorts rotulados vêm de criadores; na beleza, 50% de criadores e 50%
  de marcas. Uma diferença de erro entre setores pode ser do tipo de canal.
- **Mídia paga:** a API não diz se o Short foi impulsionado.
- **Multimodal numa amostra:** rubrica do VLM e sinais de vídeo e áudio existem em 1.400 Shorts; em 2025 isso dá só
  ~206 pares a até 14 dias para medir o ganho, e o resultado é inconclusivo por tamanho.
- **Transcrição e idioma da fala fora do modelo** (decisão #35); na amostra, 52 Shorts em inglês (~4%).
- **Capa da data da coleta:** o canal pode tê-la trocado depois da publicação.
- **Drift:** nada depois de ~19/09/2026; mudanças do algoritmo do YouTube depois disso ficam fora.
- **Associação, não causa:** os fatores (SHAP) explicam o modelo, não o mundo.
- **P(A > B)** supõe erros independentes entre A e B. <!-- PREENCHER: confirmar com o que a Tool 5 fizer -->
- **Concentração:** 13 canais têm 62% das 20.579 linhas de treino (a Salon Line, 2.082; o canal mediano, 166); o
  modelo pesa cada linha igual (decisão de 20/10, H6).
- **Sondas da capa:** `roda` (AUC 0,670 contra a rubrica do VLM) não passa de 0,7 (decisão de 21/10, F26); o AUC é
  piso, porque o rótulo também erra.
- **Rubrica do VLM:** o kappa (0,72 a 1,00 no reteste de 50) mede a consistência do VLM com ele mesmo, não o acerto.
- **Segurança:** legenda, texto na tela e fala são dados, nunca instrução (princípio 8). <!-- PREENCHER: resultado
do teste de injeção, A7 -->

## Licenças dos checkpoints e termos

Cada licença vem do model card ou do repositório oficial do checkpoint, lida em 21/10/2026 (menos os termos do YouTube, só citados: o link não foi relido nesta sessão). O YouTube e o Gemini
são termos de serviço, não licenças de pesos.

| Componente | Uso no Prévia | Licença | Fonte |
| :--- | :--- | :--- | :--- |
| SigLIP 2 (`google/siglip2-base-patch16-224`) | embedding da capa (768 dimensões, CPU) | Apache-2.0 | <https://huggingface.co/google/siglip2-base-patch16-224> |
| Whisper `small`, via faster-whisper 1.2.1 (`Systran/faster-whisper-small`) | fala e densidade de palavras na amostra | MIT | <https://huggingface.co/Systran/faster-whisper-small>; pesos originais da OpenAI: <https://github.com/openai/whisper/blob/main/LICENSE> (MIT) |
| `paraphrase-multilingual-MiniLM-L12-v2` | embedding do título (384 dimensões) | Apache-2.0 | <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2> |
| YuNet (`face_detection_yunet_2023mar.onnx`, opencv_zoo) | presença, número e área de rosto na capa | MIT | <https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE> |
| `text_detection_en_ppocrv3_2023may.onnx` (opencv_zoo) | caixas de texto na capa (só detecta, não lê) | Apache-2.0 | <https://github.com/opencv/opencv_zoo/tree/main/models/text_detection_ppocr> |
| Gemini (`gemini-3.5-flash-lite`) | rubrica do VLM e explicador (API, sem pesos) | Gemini API Additional Terms of Service (atualização de 28/04/2026, em vigor desde 23/03/2026) | <https://ai.google.dev/gemini-api/terms> |
| YouTube Data API v3 | origem dos dados | YouTube API Services Terms of Service e Developer Policies | <https://developers.google.com/youtube/terms/api-services-terms-of-service> |

- **Gemini, free tier:** pelos termos, o conteúdo enviado e as respostas podem ser usados pelo Google para melhorar
  seus produtos, com leitura humana possível, e não se deve enviar informação sensível, confidencial ou pessoal. Por
  isso o Prévia só enviou URLs de Shorts já publicados, e um criativo não publicado nunca vai ao free tier
  (princípio 5). <!-- PREENCHER: se o explicador ou a tool servirem criativo não publicado, em que plano da API
  isso roda -->
- **MiniLM e SigLIP 2 em uso comercial:** Apache-2.0 permite. Se a Galaxies for usar comercialmente, a decisão é
  dela; a aula 7.4 pede o registro, não a escolha.

## Contato e versão

- **Responsável:** Vinicius Castiglia. **Repositório:** `case-galaxies` (privado).
- **v0 (21/10/2026):** rascunho do Claude, a revisar. <!-- PREENCHER: a versão final, com o nome do modelo, a data e o
  run id, depois da H1 e da decisão #33 -->
