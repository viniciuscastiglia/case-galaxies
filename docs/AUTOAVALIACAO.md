# Autoavaliação do Prévia (checklist do hub, `projetos/galaxies/autoavaliacao`)

Estado de 22/10/2026, depois da entrega. Segue os itens do hub na ordem e no agrupamento dele (núcleo, depois "ir
além"). Cada linha traz a evidência: um arquivo, um teste, um run do MLflow ou um comando que a reproduz.

- **Status:** feito, parcial ou não feito.
- **Runs citados:** `6b1f7e77` é o teste final, aberto uma única vez (01/04 a 14/09/2026); `e684ac8c` é o modelo
  final no desenvolvimento (walk-forward de 2025); `5373921a` é o modelo em uso; `dc6cd141` é a avaliação do agente.

## Camadas do multimodal (quadro do hub)

| Camada | Status | Evidência |
| :--- | :--- | :--- |
| 1 · Mínima: frames + SigLIP, faster-whisper, rosto/texto/objetos, métricas OpenCV, PySceneDetect, tudo no GBM | parcial | SigLIP 2, 10 sondas e a aparência da capa na base toda (`visual.py`); rosto (YuNet) e texto (detector PP-OCRv3) na capa, na base toda (`detectores.py`); frames, cortes (PySceneDetect), movimento (Farnebäck), aparência dos frames e faster-whisper numa amostra de 1.400 Shorts (`midia.py`). Não há detector de objetos. Cada bloco foi testado no GBM (`HIPOTESES.md`, H7 a H10); só a capa ficou no modelo |
| 2 · Recomendada: + VLM com rubrica JSON + clusterização | parcial | rubrica de 6 campos com âncoras pelo Gemini na amostra (`vlm.py`; `test_rubrica_tem_6_campos_com_ancoras`; kappa de 0,72 a 1,00 no reteste de 50); inconclusiva no GBM (H8). Sem clusterização: a rubrica tem campos fixos, sem descritores livres para agrupar (`docs/DECISOES.md`, 21/10) |
| 3 · Ambiciosa | não feito | o README não promete |

## 1 · Agente em funcionamento

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Recebe criativo + contexto e devolve previsão, intervalo, ranking e justificativa | feito | `previsao.prever_criativo` e `previsao.comparar_criativos` (o ranking e P(B > A) de cada par); `tests/test_previsao.py` |
| Função Python documentada com contrato Pydantic (`predicted_metric`, `interval_low/high`, `confidence_flag`, `relative_to_channel`, `top_drivers`, `explanation`, `comparables`) | feito | `schemas.py` (`PredictionRequest`, `PredictionResponse`, mais `features_used` e `model_version`); `tests/test_schemas.py` (recusas de data sem fuso, setor fora, campo extra, data a mais de 90 dias, vídeo horizontal ou longo) |
| Servidor MCP funcionando (Inspector ou cliente) | feito | `servidor_mcp.py`: 2 tools, 3 resources e 1 prompt; `tests/test_servidor_mcp.py` com o cliente do FastMCP; por stdio com o modelo real (`uv run python -m previa.demo --mcp`) e pelo adaptador do LangGraph, lançado de outro diretório (`--langchain`). Não foi rodado no Inspector nem no Claude Code |
| Agente LangGraph com tools de extração, histórico/comparáveis, predição, explicabilidade e ordenação | feito | `agente.py`: grafo fixo extrair → contexto → prever → explicar → responder, cada nó chamando o mesmo passo da função; `create_agent` com as 2 tools no nível do pedido; com o Gemini real, trajetória certa em 22 de 24 perguntas (run `dc6cd141`) |
| O número vem sempre do modelo, com teste que prova | feito | `test_invariante_o_numero_pelo_mcp_e_o_da_funcao_e_o_do_modelo`, `test_invariante_grafo_igual_funcao_igual_modelo`, `test_agente_so_cita_o_numero_da_tool_que_e_o_da_funcao_pura`; a injeção na legenda não muda o número |
| Explicação validada: só cita números e fatores da saída do modelo | feito | `explicador.validar` (números no formato brasileiro, fatores contidos no `top_drivers`, termos proibidos, persona); uma nova tentativa e, se falhar, o template; 6 de 6 aprovadas com o Gemini real; `tests/test_explicador.py`. Não há juiz independente (limitação no README) |

## 2 · Modelo preditivo treinado

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Base documentada (data card): origem, período, volume, termos, setores, o que não representa | feito | `docs/DATA_CARD.md` |
| Features por bloco, com a garantia de que existem antes da publicação | feito | `docs/AUDITORIA_FEATURES.md` (inclusive a capa do modelo final); `test_historico_30_dias_entra_29_nao`, `test_pca_ajustado_so_no_treino`, `test_paridade_treino_uso_com_extras`; o contrato da tabela de treino confere a lacuna de 30 dias (`validar_tabela_de_treino`) |
| Baseline mínimo (mediana/média histórica do canal) implementado e reportado | feito | `baselines.py` (mediana madura, média dos últimos k e mediana de 90 dias); 49,7% na métrica principal; desvio registrado: mediana em log, não média |
| GBM batendo o baseline em validação temporal | feito | teste final (run `6b1f7e77`): 55,3% contra 49,7%, +5,6 pontos [+3,6; +7,1]; canais nunca vistos +3,5 [+1,9; +4,9]; critério fixado antes, cumprido nos dois |
| Artefatos versionados (MLflow), com a data de corte e a versão da base | feito | cada run grava commit, hash da base, config, data de corte e a avaliação usada (`experimento.registrar`); o modelo em uso é artefato do run `5373921a`, e toda resposta traz a versão e o run |

## 3 · Experimento

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Hipótese escrita antes de rodar | parcial | `HIPOTESES.md`, H1 a H11. H3 e H5 a H11 foram commitadas antes da abertura dos resultados, que ficaram lacrados (a ordem dos commits é a prova). H1, H2 e H4 entraram no commit do runner (19/10, 13h19), dez minutos depois de os primeiros runs da H2 e da H4 começarem, numa versão do runner que ainda imprimia o resultado: para elas, o "antes" não é demonstrável pelo histórico |
| Validação temporal (principal) e por canal (não vistos) | feito | `splits.py` (`test_walk_forward_avanca_no_tempo_com_gap`, `test_canal_novo_nunca_aparece_no_treino`); o teste final nos dois cenários |
| Métricas de erro: MAE em log e MAPE, com a interpretação correta | feito | no teste: `mae_log` 0,925 (erro típico de ×2,52, um fator multiplicativo) e MAPE em log de 25,6% [18,7%; 35,6%] (run `6b1f7e77`) |
| Métricas de ordenação: Spearman global e intra-canal, par-a-par | feito | no teste: par-a-par 55,3%; Spearman global 0,75 [0,55; 0,86]; Spearman intra-canal 0,18 (baseline 0,05) |
| Cobertura do intervalo medida no teste (geral e por setor) | feito | 76,5% no total (68,9% antes do ajuste conformal), 80,6% no automotivo, 71,4% na beleza, e por mês de 84,3% a 65,7%. Abaixo de 80%: limitação no README, com o retreino como resposta |
| Ganho da camada multimodal por ablação, com IC | feito | `previa.ablacao` (escada, todos menos um e ruído); a capa soma +2,3 pontos [+1,2; +3,3], sólida nas três leituras; a rubrica e o vídeo são inconclusivos na amostra (README, seção 7) |
| Análise de onde o modelo erra | feito | `previa.erros`: os 20 piores do walk-forward (run `e684ac8c`) são subestimações de 416× a 2.249×; 12 com sinal de impulsionamento, 3 virais de verdade e 5 ambíguos |

## 4 · Demonstração

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Executável sobre um conjunto de criativos | feito | `uv run python -m previa.demo` |
| Criativos de canais fora do treino, com métrica real conhecida ("de-para") | feito | 30 Shorts de 6 canais fora da base: 24 de 30 dentro do intervalo e 41 de 58 pares na ordem certa (`uv run python -m previa.demo --golden`, com alarme) |
| Mostra previsão, evidências (comparáveis, `top_drivers`) e explicação | feito | a demo imprime o número, o intervalo, o flag, os fatores, a versão e o A ou B com P(B > A) |
| Plano B: gravação e respostas salvas | feito | o vídeo da entrega e as respostas em `data/demo/respostas.json` (fora do git). Não há prints do Inspector |

## 5 · README

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Problema e recorte | feito | seção 1 |
| Base: origem, período, termos, limites | feito | seção 2 e `docs/DATA_CARD.md` |
| Arquitetura | feito | seção 3 (diagrama e o papel do LLM) |
| Instalação e execução (reprodutível do zero) | parcial | seção 4; o CI roda os testes num clone limpo, mas a instalação não foi testada numa máquina limpa |
| Modelos e configurações (versões fixadas) | feito | seção 5 e `pyproject.toml` |
| Metodologia | feito | seção 6 |
| Resultados (tabela com baseline e ablação) | feito | seção 7 |
| Limitações | feito | seção 8 |
| Evolução | feito | seção 9 |

## Ir além (opcional)

| Extensão | Status | Evidência |
| :--- | :--- | :--- |
| Treinamento dinâmico (7.1) | não feito | o README não promete |
| Intervalos calibrados por conformal (CQR) | feito | `intervalo.py`: CQR calibrado por data; cobertura de 68,9% para 76,5% no teste |
| RAG sobre criativos comparáveis | parcial | `comparaveis.py`: os Shorts parecidos, do passado maduro e do mesmo setor, entram como evidência na resposta; não viram feature, e o recall não foi medido. O README não chama isso de RAG |
| Loop de otimização what-if (7.2) | não feito | o README diz que a comparação só ordena |
| Comparação LLM zero-shot vs. modelo (7.3) | não feito | listada em "Evolução" |
| Embeddings de vídeo completo (X-CLIP/VideoMAE) e áudio (CLAP) | não feito | o ritmo vem dos cortes e do movimento, só na amostra |

## O que costuma diferenciar os melhores projetos (as 5 dicas do hub)

| Critério | Status | Evidência |
| :--- | :--- | :--- |
| Avaliação honesta: split temporal, sem vazamento, otimismo do CV aleatório medido | parcial | split temporal com gap, teste aberto uma vez, travas e testes contra vazamento. O otimismo do CV aleatório não foi medido (o KFold é proibido no projeto) |
| Resultados com IC em vez de número solto | feito | todo IC é bootstrap de canais inteiros, pareado quando compara dois modelos |
| Análise de erros com exemplos concretos | feito | os 20 piores, com link e categoria |
| Explicação que não inventa | feito | o validador e o template; a explicação só cita pelo nome as sondas da capa validadas |
| README que qualquer pessoa roda do zero | parcial | o início rápido e o CI; sem teste numa máquina limpa |

## Contagem (itens do checklist do hub: 37)

Núcleo: 6 + 5 + 7 + 4 + 9 = 31 itens. Ir além: 6 itens. As camadas e as 5 dicas não entram na contagem.

| Status | Núcleo | Ir além | Total |
| :--- | ---: | ---: | ---: |
| Feito | 29 | 1 | 30 |
| Parcial | 2 | 1 | 3 |
| Não feito | 0 | 4 | 4 |
| **Total** | **31** | **6** | **37** |
