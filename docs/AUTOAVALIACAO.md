# Autoavaliação do Prévia (checklist do hub, `projetos/galaxies/autoavaliacao`)

## Como ler

Rascunho do Claude, de 21/10/2026, para o Vinicius revisar (decisão #38, "Se atrasar", item 14). Segue os itens do hub
na ordem e no agrupamento dele (núcleo, depois "ir além").

- **Status:** feito, parcial, falta, ou depois do teste. "Falta" quer dizer que não há evidência no repositório hoje.
- **`PREENCHER:`** a resposta depende de resultado lacrado (H2 a H11, ablações) ou do teste final (abertura única).
  Nenhum número foi lido para escrever este rascunho.
- **`V:`** o julgamento é só do Vinicius.

**O que eu não verifiquei (pontos cegos):**
1. Nenhum resultado: não abri métricas do MLflow, `data/logs`, `data/ablacao` nem `carregar_base(final=True)`.
2. Não rodei `uv run pytest` nem `ruff` agora. A contagem de testes (273 em 09h30, 363 no ROTEIRO às 18h) é de
   documentos, não minha.
3. As evidências de teste foram conferidas por `grep` no nome, não executadas.
4. Não segui o README numa máquina limpa. O CI só rodou uma vez (segundo o ROTEIRO), e não o vi.
5. Há outros agentes editando o checkout (`final.py`, `tests/test_final.py` e `configs/avaliacao.yaml` ainda não
   estão commitados). O que depende deles está marcado "não commitado".
6. O estado das peças A1, M6, flag, tools, A6, A2, A4, A7 e A8 vem de `ls src/previa` e de `grep` por `FastMCP`,
   `StateGraph`, `create_agent` e `prever_criativo` (nenhum aparece em `src/`). Se alguma entrou depois, as linhas
   da seção 1 estão defasadas.
7. Não li o `DIARIO.md` nem as pastas de `.superpowers/sdd`.

---

## Camadas do multimodal (quadro do hub)

| Camada | Status | Evidência |
| :--- | :--- | :--- |
| 1 · Mínima: frames + SigLIP, faster-whisper, rosto/texto/objetos, métricas OpenCV, PySceneDetect, tudo no GBM | parcial | SigLIP da capa e sondas (`visual.py`, `aa2cb9a`); faster-whisper (`midia.py:10`, `pyproject.toml:8`); PySceneDetect (`midia.py:182`) e Farnebäck (`midia.py:204`); rosto YuNet e texto PP-OCRv3 só na capa (`2764361`, `2f13956`, #31); **objetos não feitos**; SigLIP dos frames (gancho) não feito; transcrição e idioma fora do modelo (#35). `PREENCHER:` quais blocos ficam no modelo final, pelas H7 a H10 |
| 2 · Recomendada: + VLM com rubrica JSON + clusterização | parcial | rubrica do VLM com 6 campos (`vlm.py`, `test_rubrica_tem_6_campos_com_ancoras`, kappa 0,72 a 1,00 no DECISOES de 20/10); **sem clusterização**, registrado como desvio (F38). `PREENCHER:` H8 |
| 3 · Ambiciosa | falta | não feita; o README não promete (AUDITORIA seção 7) |

---

## 1 · Agente em funcionamento

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Recebe criativo + contexto e devolve previsão, intervalo, ranking e justificativa | **feito** | `previsao.prever_criativo` e `previsao.comparar_criativos` (`4aec332`): número do modelo com o intervalo do CQR (M6), flag, `top_drivers` (M8), comparáveis, explicação por template e, na comparação, o ranking e P(B > A) de cada par (6.3); testes em `tests/test_previsao.py`. O intervalo mede 76,5% no teste (F41) |
| Função Python documentada com contrato Pydantic (`predicted_metric`, `interval_low/high`, `confidence_flag`, `relative_to_channel`, `top_drivers`, `explanation`, `comparables`) | **feito** | Pydantic na base (`esquema.py:18,26`: `ChannelStats`, `Post`) e o contrato da tool. A1 feita em 21/10 (decisão de 21/10, `src/previa/previsao.py`, `tests/test_previsao.py`). O contrato da tool é a A1 (`.superpowers/sdd/2026-10-20-a1/caso_a_mao.md`, F19, F20, #34). `V:` o contrato é seu |
| Servidor MCP funcionando (Inspector ou cliente) | **feito** | `src/previa/servidor_mcp.py` (`8ca876b`): 2 tools, 3 resources e o prompt; 13 testes com o `fastmcp.Client` em memória (`tests/test_servidor_mcp.py`); fumaça por stdio com o pacote real e o golden (`previa.demo --mcp`: "igual ao da função: True") |
| Agente LangGraph com tools de extração, histórico/comparáveis, predição, explicabilidade e ordenação | **parcial** | `src/previa/agente.py` (`6594d9e`): grafo fixo extrair → prever → explicar → responder (estado JSON, `trilha`, `InMemorySaver`) e `create_agent` com as 2 tools no nível do pedido; o explicador da A4 (`explicador.py`) com validador. Parcial: extração, contexto e predição ficam dentro de um nó (`prever_criativo`), não em tools separadas |
| O número vem sempre do modelo, com teste que prova | **feito** | a invariante (A7): o número do MCP, do grafo e do agente é o da função e o `expm1` do modelo (`tests/test_servidor_mcp.py`, `tests/test_agente.py`); a injeção na legenda não muda o número |
| Explicação validada: só cita números e fatores da saída do modelo | **falta** | o validador é da A4 (AUDITORIA 3, "A4"). `top_drivers` (M8) existe e é testado (`tests/test_drivers.py:53-146`), mas não há explicador nem validador. `PREENCHER:` aprovação na 1ª tentativa e fidelidade do juiz (F8) |

## 2 · Modelo preditivo treinado

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Base documentada (data card): origem, período, volume, termos, setores, o que não representa | **feito** | `docs/DATA_CARD.md` (seções Origem, Período, Volume, Licença e termos, Setores, "O que esta base NÃO representa", Vieses). Base v2 congelada (`c0b41ed`, hash `96fd13a28156`). A concordância da conferência dos 50 Shorts está no data card (#19/F14, 49 de 50 = 98%, 21/10). Pendente: conferir que o texto cita a v2 (o título ainda diz "v1, 20/10") |
| Features por bloco, com a garantia de que existem antes da publicação | **parcial** | `docs/AUDITORIA_FEATURES.md`; `DATA_CARD.md` seção "Features derivadas"; testes `test_historico_30_dias_entra_29_nao` (`tests/test_features.py:60`), `test_pca_ajustado_so_no_treino` (`tests/test_experimento.py:215`), `test_desenvolvimento_nao_ve_o_periodo_de_teste` (`tests/test_dados.py:27`). `PREENCHER:` no `MODEL_CARD.md:96`, os blocos que entram no modelo final (H2 a H10) |
| Baseline mínimo (mediana/média histórica do canal) implementado e reportado | **feito** | `baselines.py` (`mediana_madura` `:53`, `media_ultimos_k` `:89`, `mediana_janela` `:106`), `test_mediana_madura_perturbacao` (`tests/test_baselines.py:60`), `08ba1b3`. README `:216` traz o baseline em 49,7%. Desvio registrado: mediana madura em log, não "média" (AUDITORIA 6) |
| GBM batendo o baseline em validação temporal | **feito** | teste final aberto uma vez em 21/10 (run `6b1f7e77`, commit `3f47754`, `uv run python -m previa.final configs/experimentos/final.yaml --abrir-o-teste`): temporal 55,3% contra 49,7%, ganho +5,6 pontos [IC +3,6; +7,1]; canal novo 53,2%, +3,5 [+1,9; +4,9]; o critério da #5 foi cumprido nos dois (DECISOES de 21/10). Desenvolvimento: H2 não fica e H11 fica (`abertura.md`). `mae_log` do teste: 0,925 contra 0,999 (7,5% menor [1,8%; 12,7%]) |
| Artefatos versionados (MLflow), com a data de corte e a versão da base | **feito** | `registrar` grava `git_commit`, `hash_base`, config e a tag de corte (`9b26709`, F22); `registrar_bundle` e o run id no `model_version` (`d9c3a5b`, `modelo.py:94,121`, #37). O modelo de uso está treinado e o pacote é artefato do run `5373921a` (versão `gbm-final-20261021-e3b7614`, base cortada `52fd60e2ae9b`, DECISOES de 21/10); o teste está no run `6b1f7e77` |

## 3 · Experimento

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Hipótese escrita antes de rodar | **feito** | `HIPOTESES.md` H1 a H11, commits `8e8935a` (H3, H5 a H9), `e3ab61a` (H10) e `b83b927` (H11, #32). A ordem dos commits é a prova. `V:` confirme que H2 e H4 foram escritas antes dos runs (não conferi as datas) |
| Validação temporal (principal) e por canal (não vistos) | **feito** | `splits.py`; `test_walk_forward_avanca_no_tempo_com_gap` (`tests/test_splits.py:109`), `test_canal_novo_nunca_aparece_no_treino` (`:210`); teste final com os dois cenários em `final.avaliar_final` (`test_avaliar_final_devolve_os_dois_cenarios`, não commitado). Valores do teste na linha do GBM (run `6b1f7e77`) |
| Métricas de erro: MAE em log e MAPE, com a interpretação correta | **parcial** | `metricas.mae_log` (`:14`), `mape` (`:22`), painel com IC por canal (`27a4d56`). No teste (run `6b1f7e77`): `mae_log` 0,925, e^MAE ×2,52 [2,21; 3,06]. `V:` explicar que e^MAE é um fator multiplicativo, não percentual |
| Métricas de ordenação: Spearman global e intra-canal, par-a-par | **parcial** | `spearman_global` (`:29`), `spearman_intra_canal` (`:63`), `pairwise_accuracy` (`:70`), manchete na janela de 14 dias (`pairwise_janela` `:271`, `16e9f5d`). Manchete do teste: 55,3% (temporal) e 53,2% (canal novo), run `6b1f7e77`, e a coluna do Prévia na comparação com a literatura (README seção 7). Spearman do teste: global 0,75 [0,55; 0,86] e intra-canal 0,18 (temporal); 0,73 e 0,13 (canal novo) |
| Cobertura do intervalo medida no teste (geral e por setor) | **feito** | M6 feita (`src/previa/intervalo.py`, CQR com q̂ = 0,229 em log, calibração por data). No teste (run `6b1f7e77`): cobertura 76,5% (alvo 80%; crua 68,9%), automotivo 80,6%, beleza 71,4%, 10 a 50 posts 63,2%, por mês de 84,3% (abril) a 65,7% (setembro), F41. Abaixo de 80% ± 3: entra como limitação no README, com o retreino (v2) como resposta |
| Ganho da camada multimodal por ablação, com IC | **parcial** | código pronto: `ablacao.py` (escada, todos menos um, ruído; `test_todos_menos_um_tira_so_o_bloco_e_compara_com_o_completo`, `tests/test_ablacao.py:55`; `test_controle_negativo_acrescenta_ruido_ao_completo`, `:64`), IC pareado por canal (`tests/test_metricas.py:329`), abertura com as três leituras (`97b4ef3`, #36). `PREENCHER:` a tabela de ablação, que sai de `previa.abertura` (runs lacrados); só o bloco sólido nas três leituras vira argumento |
| Análise de onde o modelo erra | **parcial** | `erros.py` (cortes por origem do canal, `piores(n=20)`; `ecb59bb`). A M9 (os 20 piores, classificados) é dele, com pré-triagem do Claude (#38). `PREENCHER:` os 20 piores erros (`README.md:229`, `MODEL_CARD.md:145`). `V:` a classificação |

## 4 · Demonstração

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Executável sobre um conjunto de criativos | **falta** | não há script de demo; depende da A1 e da `prever_criativo`. A8 define 8 blocos em até 7 minutos (AUDITORIA 3, "A8") |
| Criativos de canais fora do treino, com métrica real conhecida ("de-para") | **parcial** | `data/golden/golden_v1.parquet` (30 Shorts, 6 canais fora da base, 01/05 a 14/09/2026) e `historico_v1.parquet`; `golden.py`, `test_candidatos_sao_as_reservas_revisadas_de_fora_da_base`, `test_escolhe_os_5_mais_recentes_maduros_do_periodo`. Falta o modelo rodar sobre eles. `PREENCHER:` cobertura e par-a-par do golden (`README.md:226`). `V:` a regra que escolhe os criativos da demo, escrita antes de rodar |
| Mostra previsão, evidências (comparáveis, `top_drivers`) e explicação | **falta** | peças soltas: `comparaveis.py` (`test_mesmo_setor_e_so_o_passado_maduro`), `drivers.py`. Falta a integração (A1) e a explicação (A4). Todo número dito deve aparecer na tela, com a fonte |
| Plano B: gravação e respostas salvas | **falta** | nada gravado. A8: gravação, prints do Inspector e do trace, respostas em JSON |

## 5 · README

Estado de `README.md` (300 linhas), 9 seções. Há 19 linhas com `PREENCHER` no README e 46 em `README.md`, `DATA_CARD.md` e
`MODEL_CARD.md` somados (contagem por `grep`, em 21/10).

| Item | Status | Evidência |
| :--- | :--- | :--- |
| Problema e recorte | **feito** (texto dele) | seção 1 (`README.md:15`), texto do Vinicius de 05/10. `V:` o porquê do alvo e o critério de sucesso são seus (ROTEIRO, "README") |
| Base: origem, período, termos, limites | **feito** | seção 2 (`:46`) e `DATA_CARD.md`. `PREENCHER:` o README ainda aponta o data card como v0.1 (`:48`); atualizar para a v2; a conferência dos 50 Shorts (98%) já entrou no data card |
| Arquitetura | **parcial** | seção 3 (`:72`) descreve o desenho, mas os nomes do contrato, do servidor e do grafo esperam as peças (`:74`, `PREENCHER`) |
| Instalação e execução (reprodutível do zero) | **parcial** | seção 4 (`:96`); `uv sync`, `pytest` e a ordem da coleta estão. O comando do servidor MCP aponta um módulo que não existe (`:126-130`). **Não testado do zero** |
| Modelos e configurações (versões fixadas) | **parcial** | seção 5 (`:139`); versões fixas em `pyproject.toml` (`mlflow==3.16.1`, `pydantic==2.13.5`, `scenedetect==0.7.1`). intervalo (M6), blocos finais e versão em produção (`gbm-final-20261021-e3b7614`, run `5373921a`) já estão no README seção 5 (21/10) |
| Metodologia | **feito** (rascunho) | seção 6 (`:166`); `PREENCHER:` nº de testes e o resultado das peças A1, A4 e A7 (`:201-203`) |
| Resultados (tabela com baseline e ablação) | **parcial** | seção 7 (`:210`). Entraram em 21/10 a tabela de ablação (Δ, IC, decisão), o teste final nos dois cenários (run `6b1f7e77`), a cobertura e o flag, e a coluna do Prévia na literatura (F29). Seguem `PREENCHER:` o `mae_log` e o Spearman do teste, o golden e a M9. Escreve o `V:` do README (`:243`). O texto desta seção é dele (#38) |
| Limitações | **parcial** | seção 8 (`:251`) é concreta (idade do rótulo, inscritos, mix de setor, mídia paga, amostra, drift, memória do VLM, coletor sem varredura, recall dos comparáveis, F42). Entraram a frequência do flag no teste (61,4%) e a cobertura por mês (F41). `PREENCHER:` o P(A>B) da Tool 5. Faltam a transcrição e o idioma fora (#35), o objeto não detectado e a clusterização ausente. Texto dele (#38) |
| Evolução | **feito** (rascunho) | seção 9 (`:279`). `V:` confirme a ordem (`:281`) |

---

## Ir além (opcional)

| Extensão | Status | Evidência |
| :--- | :--- | :--- |
| Treinamento dinâmico (7.1) | **falta** | não feito; o README não promete (AUDITORIA 7) |
| Intervalos calibrados por conformal (CQR) | **feito** | M6: CQR à mão, calibração por data (1.843 posts), q̂ = 0,229 em log; cobertura crua 68,9% e com CQR 76,5% no teste (run `6b1f7e77`) |
| RAG sobre criativos comparáveis | **parcial** | recuperação existe (`comparaveis.py`, `IndiceComparaveis` `:31`, `vizinhos_no_tempo` `:62`); o índice da tool não está montado. Os comparáveis entram só como evidência (F43). Não chamar de RAG no README sem a evidência na resposta |
| Loop de otimização what-if (7.2) | **falta** | não feito; ou o README diz que a Tool 5 só ordena |
| Comparação LLM zero-shot vs. modelo (7.3) | **falta** | não feito; listado em "Evolução" (`README.md:285`) |
| Embeddings de vídeo completo (X-CLIP/VideoMAE) e áudio (CLAP) | **falta** | não feito; o ritmo vem dos cortes e do movimento (Farnebäck, `midia.py:204`) |

## O que costuma diferenciar os melhores projetos (as 5 dicas do hub)

| Critério | Status | Evidência |
| :--- | :--- | :--- |
| Avaliação honesta: split temporal, sem vazamento, otimismo do CV aleatório medido | **parcial** | split temporal e sem vazamento: ver seção 3 e os princípios 2 e 3 do CLAUDE.md. **O otimismo do CV aleatório nunca foi medido** (KFold é proibido, `splits.py`). Se quiser o argumento, é um experimento a mais |
| Resultados com IC em vez de número solto | **feito** (no código) | IC por bootstrap de canal em tudo (`metricas.py:134-244`, painel `27a4d56`). `PREENCHER:` valores |
| Análise de erros com exemplos concretos | **parcial** | M9 (acima) |
| Explicação que não inventa | **falta** | A4 (seção 1) |
| README que qualquer pessoa roda do zero | **parcial** | seção 5, linha "Instalação"; CI sem dados e sem LLM (`.github/workflows/ci.yml`) |

---

## Contagem (itens do checklist do hub: 37)

Núcleo: 6 + 5 + 7 + 4 + 9 = 31 itens. Ir além: 6 itens. As camadas e as 5 dicas não entram na contagem.

| Status | Itens |
| :--- | ---: |
| Feito | 8 |
| Parcial | 14 |
| Depois do teste (só a linha "GBM bate o baseline") | 1 |
| Falta | 14 |
| **Total** | **37** |

Atualização de 21/10 (noite): teste final, intervalo (M6), flag medido no teste, contrato A1 e pacote no MLflow passaram a "feito"; depois, o servidor MCP, o agente (A6), o explicador (A4), a demo e a M9 entraram (`8ca876b` a `fb709dc`). A contagem de `PREENCHER:` abaixo é a da primeira passada e está defasada.

Os `PREENCHER:` e os `V:` aparecem dentro de linhas de qualquer status. Contagem por ocorrência neste arquivo:
`PREENCHER:` 25; `V:` 12.

## O que muda se hoje sair o que falta

Ordem de fechamento sugerida (do ROTEIRO): A1, depois `prever_criativo`, M6, A2, A6, A4, A7, F16 (teste final), as
tabelas dos documentos. Cada peça que sair muda a linha de cima da seção 1, 3 ou 4. Se às 22h a A1, a M6 ou a A2 não
estiverem de pé, as linhas "falta" viram limitação escrita no README (nada sai do produto, mas o texto não pode
prometer).
