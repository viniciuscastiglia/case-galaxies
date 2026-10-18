# Mapa do material da liga

O mapa resume o material do case Galaxies (Inteli Academy), lido por inteiro pelo Claude em 13/10/2026: três
leitores em paralelo para os módulos 1 a 6 e leitura direta do resto. São o hub do case, as 47
aulas de ML Preditivo e Agentes, as 8 de RAG e Rerankers e as 8 de Eval-Driven Development. Conceitos básicos de LLM
foi estudada em 05/10 e não tem cartão.

**O cartão é índice e checklist, não substituto da aula.** Para implementar uma peça, o Claude lê a aula inteira no
clone (`/aula X.Y`). O material vence qualquer resumo daqui.

- Clone: `~/autoestudos`, atualizado com `git -C ~/autoestudos pull --ff-only`.
- Trilha: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/<módulo>/<aula>/index.mdx`.
- Hub: `~/autoestudos/autoestudos/docs/projetos/galaxies/`.
- Site: https://inteliacademyclub.github.io/autoestudos/projetos/galaxies/ (`/aulas/` dá 404).

## Arquivos

| Arquivo | Conteúdo |
| :--- | :--- |
| [hub.md](hub.md) | escopo, contrato da tool, entregáveis, roadmap, Stack 2026, templates, autoavaliação, troubleshooting |
| [modulo-1.md](modulo-1.md) | formulação e dados (1.1–1.8) |
| [modulo-2.md](modulo-2.md) | avaliação honesta (2.1–2.6) |
| [modulo-3.md](modulo-3.md) | gradient boosting (3.1–3.6) |
| [modulo-4.md](modulo-4.md) | do vídeo às features (4.1–4.8) |
| [modulo-5.md](modulo-5.md) | incerteza e explicabilidade (5.1–5.7) |
| [modulo-6.md](modulo-6.md) | agente preditivo e MCP (6.1–6.7) |
| [modulo-7.md](modulo-7.md) | ir além (7.1–7.5, opcional) |
| [rag.md](rag.md) | RAG e Rerankers (8 aulas) |
| [edd.md](edd.md) | Eval-Driven Development (8 aulas) |
| [conflitos.md](conflitos.md) | onde o material se contradiz ou silencia, com recomendação e dono da decisão |

## Como ler um cartão

Toda aula de ML tem as seções "No projeto Galaxies", "Erros comuns", "Exercício", "Resumo" e "Referências". O cartão
traz os conceitos, o que a aula constrói (funções, números e versões), o que ela pede ao projeto, **todos** os erros
comuns, o exercício e o que fazemos no Prévia.

O cabeçalho de cada cartão diz o tempo oficial, o eixo, a velocidade de leitura e a prática.

**Prática:**
- **Vinicius escreve (M#/A#):** o coração da peça real, com `TODO(human)` e um teste-contrato pronto. Lista em
  `docs/ROTEIRO.md`.
- **o Vinicius decide:** uma decisão dele, que o Claude implementa e registra em `docs/DECISOES.md`.
- **Claude:** encanamento. O Vinicius lê um resumo de 3 linhas e faz teach-back só do que vai apresentar.
- **depois de 23/10:** o exercício da aula na base sintética, na trilha de consolidação.

**Velocidade de leitura:**
- **profunda:** o núcleo de ML e AI Engineering;
- **rápida:** "No projeto Galaxies", "Erros comuns" e "Resumo", com o resto depois da entrega;
- **depois de 23/10:** o módulo 7.

## Ordem de leitura até 21/10

Os tempos oficiais do material incluem interativos e exercícios. Só lendo, conte uns 75% deles.

| Fase | Datas | Leitura | Tempo oficial |
| :--- | :--- | :--- | :--- |
| 1 | 14–16/10 | 2.3, 2.4, 2.5, 2.6 + RAG (8, rápidas) | 210 min + ~65 min |
| 2 | 17–18/10 | 3.4, 3.5, 6.1 (profunda); 3.1–3.3, 3.6 (rápida) | 300 + 50 min |
| 3 | 18–19/10 | 6.5 + EDD 2–6 (profunda); 6.2, 6.3, EDD 1, 7, 8 (rápida) | 175 + ~65 min |
| 4 | 19–20/10 | 4.8, 5.2, 5.3 (profunda); 4.1–4.7, 5.1, 5.4 (rápida) | 400 + 175 min |
| 5 | 19–21/10 | 5.6, 5.7, 6.6 (profunda); 5.5, 6.4 (rápida) | 240 min |
| 6 | 21/10 | 6.7 (rápida) | 45 min |

Velocidades replanejadas em 15/10 pela entrega em 21/10 (decisão #24): leitura profunda só nas aulas que sustentam uma
peça do Vinicius.
| depois | a partir de 24/10 | módulo 7; a parte rápida do módulo 4; os exercícios adiados | 240 min + |

## Pré-requisitos entre trilhas (declarados no cabeçalho das aulas)

- **EDD** vem antes de:
  - 5.7: LLM-as-a-judge e tríade do RAG;
  - 6.2: avaliação de agentes;
  - 6.6: golden datasets, métricas determinísticas e avaliação de agentes;
  - 7.3: LLM-as-a-judge.

  A 2.2 cita golden datasets.
- **RAG** vem antes de:
  - 3.6: embeddings;
  - 4.4: embeddings;
  - 6.4: embeddings, vector databases e retrieval.
- **Conceitos de LLM** vem antes de:
  - 1.1: funcionamento;
  - 4.7, 6.1, 6.2 e 6.5: function calling;
  - 5.7: alucinação.

## O que é entrega

Lista do hub e do checklist da 6.7; detalhes em [hub.md](hub.md).

- contrato Pydantic, `prever_criativo` e servidor MCP, testados no Inspector e num host real;
- invariante "o número do agente é o número do modelo";
- golden set "de-para" com canais fora do treino;
- baseline × GBM em validação temporal (e por canal), com métricas de ordenação e IC;
- ablação multimodal com IC;
- intervalo de 80% com cobertura medida, total e por setor;
- `confidence_flag`;
- SHAP com explicação validada;
- README de 9 seções;
- demo de até 7 min com plano B;
- autoavaliação.
