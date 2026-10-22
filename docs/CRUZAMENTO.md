# Cruzamento: a auditoria (#30) contra o que foi entregue

Feito em 22/10/2026 pelo Claude, a pedido do Vinicius, para ele validar o que faltou e escolher o que aplicar. A
auditoria é a `docs/AUDITORIA.md` (21/10, 09h55: F1 a F43 e os requisitos da seção 3 para as peças de 21/10). O
entregue é o `main` em `5d6175a` (o `e982e80` depois dele só mexe em CLAUDE, ROTEIRO, DECISOES e DIARIO).

**Verificação (22/10):** `uv run pytest`: 670 passed em 121 s; `uv run ruff check .`: All checks passed.

## O que este cruzamento não cobriu

1. **Não reli o TAPI, o hub nem as aulas.** O cruzamento é auditoria × código. Se a auditoria deixou passar algo,
   este documento também deixa; o "cruzamento novo" do ROTEIRO (TAPI, hub e aulas × código) é outro trabalho.
2. **Evidência por leitura do código e `grep`.** Não rodei a tool com o modelo real, o MCP num host nem o agente com
   o Gemini. Os números (runs) vêm dos documentos; não reabri o MLflow.
3. **Não vi o vídeo entregue nem o repositório no GitHub**, só o checkout local.
4. **Seção 4 da auditoria (aula a aula):** reconferi só as linhas que não estavam "ok" em 21/10; as "ok" valem pelos
   testes, que passam.

## Resumo

- **Dos 43 achados:** 32 entregues, 4 parciais, 3 não feitos e declarados no README, e 4 não feitos sem declaração
  completa.
- **Os que pesam na defesa** (não feitos ou parciais, sem declaração):
  - **F26:** as sondas da capa foram validadas, mas o modelo usa as 10. A `roda` reprovou (AUC 0,670) e 5 não têm
    rótulo; a aula 4.4 manda ficar só com as que passam de ~0,7.
  - **F37:** o `fora_da_distribuicao` nunca é calculado (o README diz) e a faixa de valores por feature não está no
    pacote (o README não diz).
  - **F10:** com as curtidas ocultas, a tool não diz isso; o motivo do flag fala em "só 0 posts maduros".
  - **F28 e F8:** não há relatório de experimento nem juiz da explicação, e o README não diz.
- **Documentos entregues desatualizados:**
  - a `AUTOAVALIACAO.md` (entregável do hub) é o rascunho da tarde de 21/10: 24 linhas com `PREENCHER:`/`V:` e "falta"
    em itens que ficaram prontos à noite (explicador, demo, plano B);
  - a `AUDITORIA_FEATURES.md`, que o README cita como prova de que nada vem depois da publicação, não tem nenhuma
    linha da capa, o único bloco de conteúdo do modelo final;
  - o README perdeu, na reestruturação final, a frase "a instalação não foi testada do zero".
- **A2 sem host:** o MCP nunca foi testado no Inspector nem no Claude Code (`claude mcp add`). É o marco 2 do
  ROTEIRO e o "teste da banca" da auditoria; o README instrui o comando sem ele ter sido rodado.

## Como ler

- **Estado:** **ok** (entregue, com evidência), **parcial**, **não feito (declarado)** quando o README ou o data
  card dizem, e **não feito** quando nada diz.
- **Ação proposta** (para você validar):
  - **E**, aplicável agora: encanamento, teste ou texto, sem mexer no número do modelo;
  - **N**, mexeria no número: o teste já abriu, então não dá para medir de novo; vira limitação ou extensão
    depois de 23/10;
  - **V**, decisão sua;
  - **D**, documento: o Claude escreve e você revisa;
  - **—**, nada a fazer.

## 1. Os achados F1 a F43

| # | O que a auditoria pedia | Na entrega | Evidência | Ação |
| :--- | :--- | :--- | :--- | :--- |
| F1 | painel de métricas com IC | ok | `experimento.painel` (`experimento.py:377`); README §7 | — |
| F2 | camada 1 do multimodal | parcial (decidida, #31) | rosto e texto nas capas (`detectores.py`); H10 não ficou; gancho e objetos fora, declarados no README | — |
| F3 | model card e `model://card` | ok | `docs/MODEL_CARD.md` (licenças na §"Licenças"); `servidor_mcp.py` | D: o card se diz "v0, rascunho, a revisar" (`MODEL_CARD.md:277`) |
| F4 | registrar: sem regressão nos inscritos | ok | `DECISOES.md:108` | — |
| F5 | repetir as comparações decisivas em outras seeds | não feito (declarado) | README §8 | E: rodar no desenvolvimento (walk-forward), sem tocar o teste |
| F6 | detector de cortes calibrado | não feito | o bloco de vídeo não entrou no modelo, então não muda o número; o README final não cita | D: uma linha no model card |
| F7 | a explicação só usa os blocos aprovados | ok | `blocos_argumento: [capa]` (`configs/explicacao.yaml`); `previsao._aprovados` | — |
| F8 | juiz da explicação (outra família, temperatura 0, ~20 rótulos) | não feito | o README diz que o validador não vê a redação, mas não diz que falta o juiz | V: fazer (~1 h + 15 min seus rotulando) ou declarar |
| F9 | proveniência na resposta | ok | `previsao._proveniencia`; "estimativa do modelo, não métrica do YouTube" | — |
| F10 | curtidas ocultas: a tool diz isso | parcial | o post oculto vira NaN e não conta (`test_curtidas_ocultas_nao_contam_f10`), mas o motivo diz "o canal tem só 0 posts maduros" (`flag.py`, `canal_novo`). A decisão da A1 (`DECISOES.md:120`) pedia "a tool diz isso" | E: um motivo que diga "curtidas ocultas", sem mudar o número |
| F11 | criativo não publicado: regra da capa, VLM nunca no free tier | ok | capa = `cover` ou a imagem do criativo (`passo_extrair`); o explicador LLM só com `publicado=True`. A tool não olha o vídeo, e o modelo final não usa vídeo | — |
| F12 | sanidade: um canal fictício por faixa de tamanho | não feito (declarado) | README §8 | E: pelo `channel_stats`, sem dados do teste (~20 min) |
| F13 | faixas pelos quantis do canal | ok | `metricas.faixa_do_canal` | — |
| F14 | conferência dos 50 Shorts | ok | 49/50 (#19) | — |
| F15 | pandera na tabela de treino | ok | `validar_tabela_de_treino` (`modelo.py`, runner) | — |
| F16 | código do teste final | ok | `final.py`; run `6b1f7e77` | — |
| F17 | a tool calcula os blocos que a ablação aprovou | ok | `capa.extras_da_capa`; só a capa ficou | — |
| F18 | caminhos absolutos e variáveis de ambiente | ok | `caminhos.py` | — |
| F19 | `features_used` | ok | `schemas.py:213` | — |
| F20 | canal com menos de 3 posts: a cadeia | ok no código; parcial no README | `ancora.py` (#34); o README diz "regime não avaliado", mas não quantos posts caem em cada degrau, que a F20 pedia | E: contar os degraus (lê o teste já aberto, sem decisão de modelo) |
| F21 | a regra de data do flag | ok | modelo de uso v2, até 31/05/2026; a regra só volta a disparar perto de 27/11/2026 | — |
| F22 | data de corte no run | ok | `conferir_cortes` (`experimento.py:659`) | — |
| F23 | o modelo de uso rastreável | ok | `registrar_bundle`; `model_version` em toda resposta | — |
| F24 | a perda do GBM | ok | Huber (#32, H11) | — |
| F25 | baselines mais fortes | ok | `baselines.py` (k = 10, empate com a mediana madura) | — |
| F26 | sondas validadas antes de virar feature | parcial | AUC medido (`sondas.py`; `DECISOES.md:115`): rosto 0,787, falando 0,871, texto 0,703, `produto_beleza` 0,814; `roda` 0,670 reprova e 5 não têm rótulo. O `final.yaml` usa `sonda_*`, as 10. O peso está na **explicação**: nas 6 respostas da demo, "produto beleza" aparecia nos 3 canais automotivos | E (22/10): a explicação só nomeia as validadas no setor; o modelo continua com as 10 (tirá-las mudaria o número), declarado no README |
| F27 | a transcrição | ok (decidida, #35) | `FORA_DO_MODELO` (`experimento.py:136`); comentário corrigido; README §8 | — |
| F28 | relatório de experimento (template do hub) | não feito | nenhum arquivo; o README não diz | D/V: o Claude rascunha a partir do `HIPOTESES.md` e do model card; você conclui |
| F29 | comparação com a literatura | ok | README §7 | — |
| F30 | taxas do agente com LLM real e tracing | ok | `avaliacao_agente.py` (`mlflow.langchain.autolog`); 22/24 | — |
| F31 | base do treino final | ok | v1 no teste, v2 no uso (#33) | — |
| F32 | as três leituras da ablação | ok | README §7 | — |
| F33 | erros por fonte do canal | ok | `erros.origem_do_canal` | — |
| F34 | horizontal ou mais de 180 s | ok | `schemas.Criativo._confere` | — |
| F35 | importância por ganho no run | ok | `avisar_importancia_dominante`; o limite está no código, não na config | E: polimento, levar o limite para a config |
| F36 | random forest como segundo baseline | não feito (declarado) | README §8 | E: no desenvolvimento, sem tocar o teste |
| F37 | faixa de valores do treino por feature, insumo do `fora_da_distribuicao` | não feito | o pacote não tem a faixa nem o `sim_min`; `limiares_flag["dist_knn"]` é sempre None (`modelo.py`), e a tool passa `dist_knn=None` (`previsao.py:367`). O README declara só o `fora_da_distribuicao` | V: implementar (muda o flag, não o número, e o teste mediu o flag sem essa regra) ou declarar |
| F38 | registrar: sem clusterização | ok | `DECISOES.md:109` | — |
| F39 | `langchain-google-genai` | ok | `pyproject.toml:16` | — |
| F40 | silenciar o aviso do shap | ok | `pyproject.toml:68-70` | — |
| F41 | cobertura por mês | ok | `final.py:196`; README §7 | — |
| F42 | limitações: URL no VLM, varredura, recall | ok | `DATA_CARD.md:209-216`; README §8 | D: os itens estão marcados "complemento do Claude, a revisar" |
| F43 | registrar: `rel_vizinhos` fora | ok | `DECISOES.md:110` | — |

## 2. Os requisitos da seção 3 que não estão nos F

Só o que falta ou diverge; o resto da seção 3 está entregue (contrato, M6, M8, A4, A6, o teste final).

| Peça | O que pedia | Na entrega | Ação |
| :--- | :--- | :--- | :--- |
| A2 | Inspector e `claude mcp add` com caminhos absolutos; "o número dito no Claude Code é exatamente o `predicted_metric`" (marco 2) | não feito; o README instrui o comando, mas ele nunca rodou (nada no DIARIO ou nas DECISOES) | E: rodar com você (o `claude mcp add` é seu, com `!`); serve de ensaio para 23/10 |
| A2 | dogfooding com `MultiServerMCPClient` numa sessão | não feito (declarado) | E: um teste |
| A2 | perguntar à Galaxies qual versão do protocolo o cliente da persona fala | não feito | V: pergunta para a apresentação |
| Tool 1 | cache pelo SHA-256 do conteúdo | não feito (`capa.py` sem cache) | E |
| Tool 2 | índice com título e capa e o PCA do treino, com o encoder versionado | parcial: só a capa, com o embedding cru (`indice.py`); "só a capa" está declarado | V: deixar como está (declarado) ou incluir o título |
| Tool 5 | dizer "uma escala para ordenar" | parcial: a frase diz o complemento e a independência | E: uma frase |
| 6.3 | as sete perguntas por tool (custo e limites) | parcial: as descrições do MCP não dizem custo nem limite | E: texto em `configs/mcp.yaml` |
| A7 | golden de-para com alarme (cobertura de 0,6 a 0,97; par-a-par ≥ 0,5) | parcial: o resultado (24/30 e 41/58) está no model card, mas não há checagem automática; a demo só imprime | E: alarme na demo |
| A7 | grade de pedidos × horários | não feito | E: um teste parametrizado |
| Flag | taxa de 15% a 35% | 61,4% no teste, quase todo pela regra de data (declarado) | — |
| A8 | README: custo da extração por vídeo | não feito | E: medir o tempo da capa por pedido |
| A8 | README: ablação no formato da 4.8 (com Spearman por degrau) | parcial: Δ e IC nas duas réguas, sem Spearman | — (a régua #27 é a das duas métricas) |
| A8 | README testado do zero | não feito, e a declaração saiu do README final | D: repor a frase nas limitações |
| A8 | plano B com prints do Inspector e do trace | parcial: vídeo e respostas em JSON (`data/demo/respostas.json`) | E: os prints saem do item do Inspector |
| A8 | autoavaliação do hub preenchida | desatualizada: o rascunho da tarde de 21/10 | D: atualizar com o estado final; os `V:` são seus |

## 3. Achados novos (fora da auditoria)

1. **`AUDITORIA_FEATURES.md` sem a capa.** É o exercício da 2.2 (seu, de 12/10), e o README o cita como "quando cada
   feature existe". As features do modelo final (SigLIP, sondas, aparência da capa) não aparecem. A capa é a da data da
   coleta (out/2026), não a da publicação; o `DATA_CARD.md:206` diz isso, marcado "a revisar". → D/V: a linha da capa
   é sua.
2. **`configs/multimodal.yaml:20`** ainda diz "a validar" nas sondas. → E.
3. **README §1:** "com menos de 10 Shorts **anteriores**", mas a regra conta posts **maduros**. → D: uma palavra.
4. **Pendências da revisão final de 21/10** (ROTEIRO, "Pendências do Claude"): `faixa_do_canal` com escalar devolve
   array 0-d; `k_escolhido` depende da ordem de `ks`; `painel_do_run` e `previa.erros` não chamam `conferir_cortes`;
   e mais quatro polimentos. → E.

## 4. Princípios, desvios e "ir além"

- **Os 9 princípios:** todos cumpridos. Os pontos frágeis são o 7 (falha nunca silenciosa), por causa da F10 (o
  motivo engana) e do `fora_da_distribuicao` (que não roda, mas está declarado). O 4 (paridade) agora cobre a capa
  (`test_paridade_com_o_caminho_do_treino` em `tests/test_capa.py`, `test_paridade_treino_uso_com_extras`).
- **Desvios sem registro em 21/10** (F4, F24, F38, F43): todos registrados. Desvios novos depois da auditoria: o número
  pontual em vez do q50 (#40, registrado) e o índice só pela capa (declarado no README).
- **"Ir além":** o conformal entrou; o what-if, o treino dinâmico, o duelo com LLM e o embedding de vídeo não. O README
  não promete nenhum; os quatro estão na §9 (Evolução).

## 5. Proposta para validar

Ordenada pelo que a apresentação de 23/10 mais ganha. Cada item tem a ação da tabela; nada aqui mexe no número do
modelo.

1. **MCP num host de verdade (A2):** Inspector e `claude mcp add`, com o número conferido contra o
   `predicted_metric`. Os prints viram o plano B. *E, com você.*
2. **Documentos entregues que estão errados ou velhos:** a autoavaliação, a capa no `AUDITORIA_FEATURES`, a frase do
   "do zero", o "anteriores/maduros" e os "a revisar" dos cards. *D, você revisa.*
3. **Limitações que faltam no README §8:** F26 (sondas), F37 (a faixa por feature), F8 (juiz), F28 (relatório) e
   F6, se você decidir não fazer. *V, depois D.*
4. **Correções pequenas da tool:** o motivo das curtidas ocultas (F10), "uma escala para ordenar", custo e limites nas
   descrições, o alarme do golden na demo. *E.*
5. **Evidência extra, só no desenvolvimento:** F5 (seeds), F36 (random forest) e F12 (canais fictícios). *E; o
   resultado não troca o modelo, só reforça ou enfraquece o argumento.*
6. **Decisões suas:** F8 (fazer o juiz ou declarar), F37 (implementar ou declarar) e a Tool 2 (título no índice ou
   declarar).
7. **Polimento:** cache da Tool 1, grade de horários, as pendências da revisão final e o limite da F35 na config.

## 6. Contra o TAPI (o que a auditoria não pesou)

A auditoria conferiu item a item; lida contra o espírito do TAPI, a maior distância é uma só:

- **A tool lê a capa, não o vídeo.** O TAPI diz que "a Galaxies precisa ver o agente processando o vídeo", e a Tool 1
  do diagrama é "VLM + Whisper". O vídeo, o áudio e a rubrica foram extraídos e medidos numa amostra de 1.400 Shorts,
  sem ganho mensurável (H8, H9), e ficaram fora do modelo; o VLM do free tier também não pode receber criativo não
  publicado (princípio 5). Não muda sem mexer no modelo: em 22/10, virou limitação escrita no README e no model card.
- **A explicação, que o TAPI chama de obrigatória**, tinha fatores com nomes não validados (a F26 acima); consertado
  em 22/10 sem mudar o número.
- **A acurácia de referência (~60% da Galaxies)** é outra régua: a manchete (55,3%) compara Shorts do mesmo canal a
  até 14 dias, onde o baseline fica em 49,7%. Não é falta; é o recorte da métrica.

## 7. Aplicado em 22/10 (branch `feat/refino`)

Plano: `docs/superpowers/plans/2026-10-22-refino.md`. Nenhuma mudança no modelo: o golden (24 de 30 e 41 de 58) e o
número de cada um dos 30 Shorts ficaram iguais antes e depois de cada task.

| Item | O que foi feito |
| :--- | :--- |
| F26 (explicação) | só as sondas validadas no setor falam pelo nome; as outras somam no fator da capa |
| F10 | o motivo `canal_novo` diz quantos posts têm as curtidas ocultas |
| F37 (parte) | o `fora_da_distribuicao` é calculado (limiar 0,357, o p99 da calibração do modelo em uso); a faixa por feature continua sem fazer, declarada |
| Tool 1 | cache da capa pelo SHA-256 do conteúdo |
| Tool 5 e 6.3 | "uma escala para ordenar"; custo (~0,4 s por criativo com capa nova) e limites nas descrições do MCP |
| A2 | o MCP pelo adaptador do LangGraph, lançado de outro diretório, com o modelo real: o mesmo número da função. O Inspector e o `claude mcp add` ficam para o Vinicius |
| A7 | golden reproduzível com alarme (`previa.demo --golden`); grade de horários |
| Revisão final | a sanidade lida na escala do modelo; a autoavaliação com a hipótese "antes de rodar" como parcial (H2 e H4); o limiar do OOD amarrado ao pacote em uso; o texto das curtidas ocultas sugere enviar as curtidas; o passo a passo de 30/10 com as pastas que faltavam |
| F12 | sanidade com canais fictícios por tamanho (`--sanidade`; tabela no model card): o número segue o canal; a largura do intervalo, em log, é a mesma nos três tamanhos |
| F6, F8, F28 | F6 e F8 declarados no README e no model card; F28: `docs/RELATORIO_EXPERIMENTO.md` |
| Documentos | model card v1, data card, autoavaliação final, a capa na auditoria das features, o README |
| Polimento | as pendências da revisão de 21/10 e o id curto nos comandos (o `bincount` do painel fica fora) |
| F5, F36 | seeds 0, 1 e 2 para a capa e a Huber (passam nas três) e o random forest com defaults (o final o supera), no desenvolvimento |
| Achado novo | o `HIPOTESES.md` entregue tinha "Resultado:" e "Decisão:" vazios da H2 à H11; preenchidos a partir da abertura de 21/10 |

Continuam por fazer (opcionais, sem mexer no modelo): o Inspector e o `claude mcp add`; o juiz da explicação (F8), que
precisa de rótulos do Vinicius.
