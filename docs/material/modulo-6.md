# Módulo 6: Agente preditivo e MCP (7 aulas, 375 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/6-agente-preditivo-e-mcp/`.

O encadeamento do módulo:

- **6.1:** arquitetura (função pura + contrato; MCP e LangGraph são adaptadores finos);
- **6.2:** grafo determinístico, com ReAct só para perguntas abertas;
- **6.3:** as 5 tools;
- **6.4:** comparáveis sem vazamento;
- **6.5:** MCP;
- **6.6:** testes ("o número chega intacto");
- **6.7:** demo e README.

No Prévia, o módulo inteiro é AI Engineering na mão do Vinicius. O pacote do material se chama `galaxies_tool`; o
nosso é `previa`.

**Ambiente do módulo.** pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, shap 0.52.0,
pyarrow 25.0.1, faiss-cpu 1.15.1, pydantic 2.13.5, langgraph 1.2.12, langchain 1.4.2, langchain-core 1.6.5,
langchain-mcp-adapters 0.3.2, **fastmcp 3.4.7, mcp 1.30.0** e mlflow 3.16.1.

**LangGraph é obrigatório?** Pela própria 6.2, não: a função Python e o MCP bastam. Mas o hub o põe no núcleo, então no
Prévia ele entra na versão mínima (A6).

**Testes indispensáveis** (6.6 e checklist da 6.7):

- contrato recusa entradas inválidas e aceita as válidas;
- invariante: o número do agente é o número do modelo, e o MCP devolve o mesmo que a função;
- golden set "de-para" congelado;
- servidor testado no Inspector e num host real.

**Checklist "Antes de 23/10"** (6.7):

1. contrato;
2. invariante;
3. golden set congelado;
4. MCP no Inspector e num host;
5. versões fixadas;
6. `model://card`;
7. README de 9 seções testado em máquina limpa;
8. demo ensaiada em até 7 min;
9. plano B;
10. autoavaliação.

## 6.1 O LLM orquestra, o modelo prevê

50 min · AI Eng · leitura profunda · prática: **Vinicius escreve A1**

- **Conceitos.**
  - O LLM fica nas bordas; o caminho do número é determinístico.
  - Ports and adapters: 4 portas devolvem o mesmo número.
  - O contrato é a defesa: `Literal` de setor; `AwareDatetime` (19h em SP é 22h em UTC); `extra="forbid"`;
    `channel_id` ou `channel_stats`.
  - As mensagens de erro do Pydantic chegam ao LLM, que se corrige sozinho.
  - Esqueleto andante; alvo relativo; `lru_cache(maxsize=1)` em `carregar_modelo`.
- **Constrói.**
  - Árvore de arquivos: `schemas.py`, `modelo.py`, `api.py`, `tools.py`, `comparaveis.py`, `agente.py`,
    `servidor_mcp.py`, `tests/`.
  - `PredictionRequest` e `PredictionResponse` (12 campos, com `_intervalo_coerente`); `Motivo`, `ConfidenceFlag`,
    `Driver`, `Fator`, `Explicacao` e `Comparable`.
  - `Bundle`; `treinar`, com CQR e limiares no p80; `model_version`.
- **No projeto Galaxies.** A v0 não vê o vídeo e dá 2.113 para B; com a Tool 1, B chega a ≈4.240. A estrutura não
  muda. Na v0, `baixa_confianca=true` com `extracao_falhou` é honesto.
- **Erros comuns.**
  1. Pedir o número ao LLM "só por enquanto".
  2. Contrato frouxo.
  3. Lógica de negócio no adaptador.
  4. Carregar o modelo a cada chamada ou na importação.
  5. Resposta sem `model_version`.
- **Exercício.** Endurecer o contrato: `planned_datetime` até 90 dias no futuro, e campos proibidos com
  `extra="forbid"`. → Entra na A1.
- **Liga com.** Function calling (Conceitos de LLM), 1.1 e 5.3.

## 6.2 LangGraph em 2026

55 min · AI Eng · leitura profunda · prática: **Vinicius escreve A6**

- **Conceitos.**
  - Pergunta fechada → grafo fixo; pergunta aberta → ReAct. O LLM decide *quais* criativos, nunca *como* o número
    sai.
  - `create_react_agent` está descontinuado: use `langchain.agents.create_agent`.
  - O estado guarda JSON, não objetos Pydantic.
  - Os passos são injetados (`Passos`).
- **Constrói.**
  - `Estado` (TypedDict com o reducer `trilha`) e `construir_grafo(passos, explicador, checkpointer)`.
  - Nós `extrair`, `embutir_texto`, `contexto`, `prever`, `explicar` e `responder`; aresta condicional
    `rota_inicial`.
  - `InMemorySaver` (exige `thread_id`), `stream_mode`, `get_state_history`.
  - O agente ReAct, com o LLM falso `FakeComTools`.
  - `tests/test_agente.py`.
- **No projeto Galaxies.** A persona chama `comparar_criativos`; cada pedido percorre o grafo fixo. Os intervalos de A
  e B se sobrepõem, e a resposta precisa dizer isso.
- **Erros comuns.**
  1. Deixar o ReAct decidir a ordem dos passos do número.
  2. Expor ao LLM uma tool que recebe features.
  3. Guardar objetos Pydantic no estado.
  4. Esquecer o `thread_id`.
  5. Importar `create_react_agent`.
  6. Testar só com o LLM real.
- **Liga com.** EDD 6 (avaliação de agentes) e Function calling. O exercício (`marcar_revisao`) fica para depois.

## 6.3 As cinco tools do agente

60 min · AI Eng + ML · leitura profunda · prática: o Vinicius escreve as descrições dentro da A2; o Claude escreve os
corpos

- **Conceitos.**
  - A tool tem dois leitores: o LLM (nome, docstring, schema) e você às 23h.
  - Sete perguntas por tool: nome, docstring, entrada, saída, erros, idempotência e cache, custo e limites.
  - A Tool 3 é a única fonte do número.
  - Exponha tools no nível do pedido, nunca das features.
- **Constrói.**
  - Tool 1 `extrair_atributos`: cache por SHA-256 e timeout de 120 s.
  - Tool 2 `historico_e_comparaveis`: tudo filtrado pela data planejada.
  - Tool 3 `prever`: `montar_features` e `sinal_de_confianca`.
  - Tool 4 `explicar`: SHAP, `top_drivers`, payload, template sem LLM, `validar` e subagente opcional.
  - Tool 5 `ordenar_criativos` com `prob_supera` (normal no log, `np.hypot`).
  - `api.py`: `prever_criativo`, `comparar_criativos`, `what_if` e `aquecer`.
  - `ToolCallLimitMiddleware(run_limit=6)`.
- **No projeto Galaxies.** B 4.237 (1.655–9.180); A 1.477; P(B>A) = 0,884, que deixa cerca de 12% de chance de A
  render mais.
- **Erros comuns.**
  1. Expor ao LLM tool de passo interno.
  2. Erro técnico cru.
  3. Cache pelo nome do arquivo.
  4. Features montadas de um jeito no treino e de outro na tool.
  5. LLM reescrevendo a explicação sem validador.
  6. Ler P(A>B) como certeza.
- **Notas.** A docstring é o prompt da tool: um exemplo de data com fuso evita o erro mais comum. Referência:
  "Writing effective tools for agents" (Anthropic).

## 6.4 Comparáveis: retrieval de criativos

50 min · ML + retrieval · leitura profunda · prática: **Vinicius escreve A5**

- **Conceitos.**
  - Recuperar, depois prever (MMRA).
  - Cosseno em vetores normalizados é produto interno.
  - Filtros: mesmo setor e post publicado pelo menos 30 dias antes.
  - Dois papéis: evidência e a feature `rel_vizinhos`.
  - A versão vazada parece melhor em tudo.
- **Constrói.** `IndiceComparaveis.buscar(emb, setor, antes_de, k=10)` com `faiss.IndexFlatIP`, mais
  `adicionar_rel_vizinhos`. Na ablação com `rel_vizinhos`, o par-a-par vai de 0,638 a 0,672.
- **No projeto Galaxies.** Os vizinhos de B ficam em torno de 1,66×; os de A, em 0,68×. Comparáveis e atributos vêm do
  mesmo vídeo, então não são evidências independentes.
- **Erros comuns.**
  1. Vizinhos do futuro.
  2. Vizinhos imaturos.
  3. O próprio post como vizinho.
  4. Misturar setores.
  5. Embeddings de versões diferentes.
  6. Cosseno sem normalizar.
- **Notas.**
  - **No Mac:** `faiss-cpu` com `lightgbm` dá segfault (dois `libomp`). Use `OMP_NUM_THREADS=1` no `__init__.py`, ou o
    `NearestNeighbors(metric="cosine")` do scikit-learn, que é o exercício da aula.
  - Não ajuste o k no teste.
  - Se a ablação não mostrar ganho, os comparáveis ficam só como evidência.
- **Liga com.** RAG 4, 5 e 6, mais 2.2 e 1.7.
- **No Prévia.** A5: a busca com os filtros aplicados antes do kNN, no estilo do exercício `buscar_sklearn`.

## 6.5 MCP do zero

60 min · AI Eng · leitura profunda · prática: **Vinicius escreve A2**

- **Conceitos.**
  - Host, cliente e servidor.
  - Tools (o LLM decide), resources (contexto da aplicação), prompts (o usuário escolhe).
  - stdio (nada no stdout) × Streamable HTTP.
  - A spec 2026-07-28 é stateless, mas o stack da trilha fala a versão 2025-11-25.
  - **Não coloque LLM no servidor.**
- **Constrói.**
  - **Pegadinha das versões:** fixe `fastmcp==3.4.7`, `mcp==1.30.0` e `langchain-mcp-adapters==0.3.2`, e ignore o
    "Update available".
  - `servidor_mcp.py`: tools `prever_criativo` e `comparar_criativos`, com annotations e timeout; `ToolError`.
    Resources `model://card` e os schemas; o prompt `decidir_entre_criativos`; `aquecer()`.
  - Testes em memória com `fastmcp.Client(mcp)` e `@pytest.mark.anyio`.
  - Inspector: `npx @modelcontextprotocol/inspector`, apontando para o arquivo e com `-e PYTHONPATH`.
  - Claude Code: `claude mcp add --transport stdio ...`.
  - Dogfooding com `MultiServerMCPClient` e `client.session`; sem sessão, cada chamada leva ~10 s.
- **No projeto Galaxies.** Teste da banca: conectar ao Claude e perguntar "qual dos dois Shorts eu impulsiono?". O
  número dito tem de ser exatamente o `predicted_metric`.
- **Erros comuns.**
  1. `print()` num servidor stdio.
  2. Caminhos relativos.
  3. Seguir o "Update available".
  4. Uma sessão por chamada.
  5. LLM dentro do servidor.
  6. HTTP aberto sem autenticação.
- **Notas.** Sem `instructions`, o LLM do cliente não sabe que não deve alterar o número.

## 6.6 Testando e avaliando o agente

55 min · AI Eng · leitura profunda · prática: **Vinicius escreve A7**

- **Conceitos.**
  - Pirâmide: unidade, contrato, invariante, golden set, trajetória e fidelidade.
  - Tudo o que pode ser testado sem LLM é testado sem LLM.
  - O que quebra é o caminho do número, não o modelo.
  - Golden set: canais fora do treino, posts maduros, arquivo congelado. Os limiares são folgados: é alarme, não
    métrica.
- **Constrói.**
  - `conftest.py`, que treina o modelo sintético.
  - `test_schemas.py` (inclui hora local) e `test_contrato.py` (16 combinações, ida e volta por JSON,
    `low ≤ pred ≤ high`).
  - `test_invariante.py`: `numero_do_modelo` recalcula o q50 cru por fora.
  - `golden.py`, com 36 criativos e os limiares: Spearman > 0,6, cobertura de 0,6 a 0,97, par-a-par ≥ 0,5.
  - `rodar_avaliacao.py` (trajetória e fidelidade) e `rastrear.py` (`mlflow.langchain.autolog`).
- **No projeto Galaxies.** O golden set responde "e quando vocês acertaram?": 3 criativos de canais fora do treino,
  escolhidos antes de ver o resultado, com a cobertura e o par-a-par do conjunto.
- **Erros comuns.**
  1. Testar só o modelo.
  2. Testar tudo com LLM real.
  3. Golden set com canais do treino.
  4. Ajustar o modelo olhando o golden set.
  5. Limiares apertados demais.
  6. Não rastrear.
- **Notas.** Com LLM real, rode de 20 a 30 perguntas com armadilhas e reporte duas taxas: trajetória e fidelidade.
- **Liga com.** EDD 2 (golden datasets), EDD 3 (métricas determinísticas) e EDD 6 (trajetória × fidelidade).
- **No Prévia.** A7: a invariante e a injeção na legenda. O Claude faz o contrato, o golden set e o teste de regressão
  (o exercício da aula).

## 6.7 Demo e README que convencem

45 min · produto · leitura rápida · prática: **Vinicius escreve A8**

- **Conceitos.** A demo é um argumento, na ordem da dúvida:

  | Tempo | Bloco |
  | :--- | :--- |
  | 0:00 | problema |
  | 0:40 | base |
  | 1:20 | baseline × modelo |
  | 2:00 | multimodal |
  | 2:40 | ao vivo, com de-para |
  | 4:10 | explicação |
  | 4:50 | MCP |
  | 5:40–7:00 | limitações |

  Os criativos do de-para são escolhidos por uma regra definida antes.
- **Constrói.**
  - `ablacao.py` e `demo_de_para.py` (um criativo por setor).
  - `app.py` com Streamlit, opcional.
  - README com 9 seções: problema; base; arquitetura; instalação; modelos; metodologia; resultados; limitações;
    evolução.
- **Erros comuns.**
  1. Abrir pela arquitetura.
  2. Escolher o de-para depois de ver o resultado.
  3. Dizer um número sem a fonte na tela.
  4. Depender da rede.
  5. Esconder as limitações.
  6. README que só funciona na máquina de quem escreveu.
- **Notas.**
  - Grave a demo na véspera e salve as respostas em JSON.
  - Se estourar os 7 min, corte a arquitetura, nunca as limitações.
  - Se 2 de 3 caírem fora do intervalo, contextualize com a cobertura do conjunto.
