# Trilha Eval-Driven Development (8 aulas curtas, cerca de 8 mil palavras, ~55 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/eval-driven-development/`. As aulas não têm exercício nem "Erros comuns"
formais, então os cartões registram as armadilhas que o texto cita.

**No Prévia.** A EDD é a base dos testes do agente e da explicação: o validador da A4, os testes da A7, o golden set e
o CI. Pré-requisito declarado de 5.7, 6.2, 6.6 e 7.3. Ler na Fase 1.

## 1. Introdução ao EDD

- **Ideia.** O "vibe check" (testar 3 a 5 casos no olhômetro) não escala e esconde regressões. A EDD é o TDD da era
  probabilística: defina o critério de sucesso antes de implementar.
- **Ciclo:** métricas → golden dataset (50 a 500 casos) → baseline → iteração guiada por dados → prevenção de
  regressão.
- **Trilema:** qualidade × custo × latência. Meça para escolher o modelo menor quando ele bastar.
- **No Prévia.** O validador e os testes do agente são o "TDD" do explicador.

## 2. Golden datasets

- **Ideia.** Um caso de teste tem `query`, `ground_truth`, `contexts` e `metadata` (categoria e dificuldade, para
  segmentar a avaliação).
- **Três fontes:** especialistas; logs de produção, sem dados pessoais; geração sintética com LLM, que pede amostragem
  humana por causa do "eco cognitivo".
- **Contaminação:** a lei de Goodhart e o overfitting de prompt. Guarde de 20 a 30% como holdout, atualize sempre e
  varie as formulações.
- **Tamanho:** de 20 a 50 casos ao prototipar; de 100 a 250 para PRs; 500 ou mais para o release.
- **No Prévia.**
  - O golden set "de-para" da 6.6: canais fora do treino, posts maduros, arquivo congelado.
  - O golden set da explicação: de 20 a 40 payloads (5.7).
  - Liga com a 2.2 (vazamento e contaminação).

## 3. Métricas determinísticas e heurísticas

- **Ideia.** A primeira linha de defesa custa R$ 0, roda em microssegundos e é 100% reprodutível.
  - Validação de schema com Pydantic.
  - Métricas léxicas: exact match, ROUGE, BLEU e Levenshtein. São cegas à semântica: falham com paráfrases e com
    negação.
  - Asserções de negócio e testes negativos: termos proibidos, links obrigatórios, tamanho mínimo.
- **Regra:** comece pelo determinístico e só depois chame o juiz neural.
- **No Prévia.** O validador de fidelidade da 5.7 e da A4: os números citados têm de estar no payload, os fatores
  precisam ter o nome exato, o intervalo é obrigatório e a ressalva de baixa confiança vem na frente.

## 4. LLM-as-a-judge e G-Eval

- **Ideia.** Um LLM avalia com rubrica rígida. Há três modos: nota individual, comparação pareada, e com ou sem
  gabarito. O G-Eval é rubrica + raciocínio passo a passo + nota.
- **Vieses do juiz:**
  - posição: favorece a primeira opção, e se mitiga invertendo a ordem;
  - prolixidade: penalize na rubrica;
  - autopromoção: favorece a própria família, e se mitiga com um comitê de famílias diferentes.

  Use temperatura 0.
- **No Prévia.** O juiz de fidelidade da 5.7 é offline, com modelo diferente do explicador e calibrado com cerca de 20
  rótulos humanos. Opcional na entrega. A inversão de ordem reaparece na 7.3.

## 5. Avaliando pipelines RAG: a tríade

- **Ideia.** A tríade separa três relações:
  - relevância e recall do contexto (pergunta ↔ contexto);
  - fidelidade (contexto ↔ resposta), que é a métrica anti-alucinação, medida por afirmações atômicas sustentadas
    sobre o total;
  - relevância da resposta (pergunta ↔ resposta).

  Uma matriz de diagnóstico diz o que consertar. Aumentar o `top_k` só traz ruído (lost in the middle).
- **No Prévia.** A fidelidade da explicação: o payload do modelo faz o papel do contexto recuperado (5.7).

## 6. Avaliação de agentes e tool calling

- **Ideia.** Por chamada: acerto na escolha da tool, validade dos parâmetros contra o schema e sequência das chamadas.
  Por trajetória: sucesso da tarefa, eficiência de passos, loops e autocorreção. Isolamento com mocks e sandboxes;
  conferir o estado final.
- **Benchmarks:** SWE-bench, WebArena e AgentBench.
- **No Prévia.** Os testes do grafo com LLM falso (6.2) e a separação entre trajetória e fidelidade (6.6). Na A7:
  "número do agente = número do modelo" e o teste de injeção.

## 7. Frameworks e ferramentas

- **Ferramentas:**
  - DeepEval: o "pytest para LLMs", com métricas como relevância e fidelidade e limiares;
  - Ragas: a tríade do RAG e dados sintéticos;
  - Promptfoo: matriz de prompts × modelos em YAML, e red teaming;
  - Arize Phoenix: tracing.
- **No Prévia.** O pytest com fakes basta para a entrega. O DeepEval é uma opção de evolução. O tracing já existe no
  MLflow (`mlflow.langchain.autolog`, 6.6).

## 8. CI/CD e evals contínuos

- **Ideia.** Evals offline (golden set no CI, barrando regressões) × online (amostras de produção, drift).
- **GitHub Actions como portão do PR.**
- **Feedback:** explícito (positivo ou negativo, o negativo vira caso do golden set) e implícito.
- **Red teaming:** prompt injection, jailbreak e vazamento de dados pessoais, toxicidade. **É a única aula do material
  que cita prompt injection.**
- **5 mandamentos:**
  1. nunca mude o prompt sem benchmark;
  2. comece determinístico;
  3. mantenha o golden set vivo;
  4. isole para diagnosticar;
  5. automatize no CI.
- **No Prévia.** O workflow do GitHub Actions roda `uv run pytest` e o ruff sem dados reais nem LLM (Claude, Fase 5).
  O teste de injeção na legenda entra na A4 e na A7 (princípio 8).
