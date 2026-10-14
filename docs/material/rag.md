# Trilha RAG e Rerankers (8 aulas curtas, cerca de 11 mil palavras, ~75 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/rag-e-rerankers/`. As aulas não têm exercício nem "Erros comuns"
formais, então os cartões registram as armadilhas que o texto cita.

**No Prévia.** O RAG aparece na tool de comparáveis (6.4), que recupera Shorts parecidos e publicados antes. Aparece
também nos embeddings de texto (3.6) e visuais (4.4), e na analogia da fidelidade da explicação (EDD 5): o payload faz
o papel do contexto recuperado. Não há chunking nem reranker no projeto. Ler antes da 3.6 e da 6.4 (Fase 1).

## 1. Introdução ao RAG

- **Ideia.** Recuperar antes de gerar: o modelo consulta uma base externa em vez de depender só do que aprendeu.
- **Por que os LLMs precisam disso:** corte de conhecimento, alucinação com tom confiante e falta de dados do domínio.
- **Prompting × RAG × fine-tuning:** o fine-tuning muda comportamento; o RAG muda o que o modelo sabe na hora da
  resposta. Ainda há a rastreabilidade das fontes.
- **No Prévia.** É a mesma lógica dos comparáveis: "criativos parecidos renderam X" vira evidência rastreável.

## 2. Componentes do RAG

- **Ideia.** O pipeline tem base de conhecimento, recuperador (embeddings), camada de integração (o prompt
  aumentado), gerador (o LLM) e, opcionalmente, um reranker.
- **Gerações:** naive, advanced (reranking, híbrido, reescrita de query) e modular.
- **Janela de contexto:** "lost in the middle". O objetivo não é passar mais documentos, e sim os certos, em pequena
  quantidade e na ordem certa.
- **Armadilha:** base de má qualidade. Garbage in, garbage out.

## 3. Chunking

- **Ideia.** Dividir o documento em pedaços: tamanho fixo, por estrutura (parágrafo ou sentença), recursivo
  (`RecursiveCharacterTextSplitter`), semântico, por estrutura do documento, hierárquico, agêntico ou enriquecido com
  metadados.
- **Ponto de partida:** 500 a 1.000 tokens, com overlap de 10 a 20% (100 a 200 tokens). É iterativo: medir com
  Recall@K num conjunto de perguntas.
- **No Prévia.** Não se aplica, porque legendas são curtas. Leitura de contexto.

## 4. Embeddings

- **Ideia.** O significado vira uma posição no espaço vetorial. Os transformers geram vetores que dependem do
  contexto. A similaridade de cosseno é invariante à magnitude; o produto interno exige vetores normalizados.
- **Modelos:** para português, use os multilíngues (`multilingual-e5-large`, `bge-m3`). Referência: o MTEB.
- **Armadilhas:**
  - embedding é compressão com perda, então documentos longos perdem mais;
  - o bi-encoder não sabe qual query vai buscar o documento;
  - **use o mesmo modelo na indexação e na consulta**;
  - prefixos `query:`/`passage:` na família E5;
  - nunca passe do limite de tokens.
- **No Prévia.** O `paraphrase-multilingual-MiniLM-L12-v2` (3.6) e o SigLIP 2 (4.4). Versione o encoder junto com o
  índice dos comparáveis (6.4).

## 5. Vector databases

- **Ideia.** Busca exata (ENN) ou aproximada (ANN): HNSW, IVF ou PQ. Até cerca de 50 mil vetores, a busca exata é
  viável.
- **Bancos:** Pinecone, Qdrant, Weaviate, Milvus, FAISS (biblioteca, sem servidor) e pgvector.
- **Recursos:** filtro por metadados; híbrido nativo.
- **Memória:** 1.536 dimensões em float32 ocupam 6 KB por vetor.
- **No Prévia.** Os ~31 mil Shorts cabem em busca exata (`IndexFlatIP` ou `NearestNeighbors`). O filtro de metadados é
  o filtro de setor e de "publicado antes", aplicado antes da busca (A5).

## 6. Retrieval

- **Ideia.** Recall × precision (`recall@K`, `precision@K`).
- **Tipos de busca:**
  - vetorial: semântica, mas fraca com códigos e nomes;
  - lexical (BM25): forte com termos exatos;
  - híbrida com RRF (k = 60).
- **Lost in the middle:** menos documentos, com os melhores no topo.
- **Otimizações:** expansão e reescrita de query, parent-child, filtro de metadados.
- **Métricas:** Recall@K, MRR, NDCG e hit rate. Um conjunto de 50 a 100 queries já paga o esforço.
- **No Prévia.** O filtro de metadados (setor, maturidade, sem o próprio post) é o que impede o vazamento. A ablação
  com `rel_vizinhos` é a métrica de valor (6.4).

## 7. Rerankers

- **Ideia.** O bi-encoder é rápido e tem recall; o cross-encoder é preciso, porque lê query e documento juntos. Daí a
  busca em dois estágios: 25 a 50 candidatos, reduzidos a 5.
- **Modelos:** `bge-reranker-v2-m3` (multilíngue), Cohere e Voyage; também RRF, ColBERT e rerankers via LLM.
- **Quando não usar:** volume altíssimo, base pequena ou MVP. Primeiro meça um baseline sem reranker.
- **No Prévia.** Não usado: os comparáveis são poucos, e a ordem vem da similaridade.

## 8. Hybrid search

- **Ideia.** A busca vetorial e a lexical falham de jeitos complementares. Fusão por RRF (k = 60) ou por soma
  ponderada com alpha: de 0,7 a 1,0 para linguagem natural, de 0,4 a 0,7 para termos técnicos, de 0,0 a 0,3 para
  códigos.
- **Embeddings esparsos:** SPLADE.
- **Pipeline de referência:** híbrido com top 50, depois cross-encoder até 5, depois o LLM. A precisão vai de ~0,45 a
  ~0,65–0,72.
- **Custo:** dois índices sincronizados.
- **No Prévia.** Extensão possível para os comparáveis (hashtags e nomes de produto exatos mais a semântica).
  Registrar como "evolução" no README.
