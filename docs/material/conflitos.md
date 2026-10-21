# Conflitos e lacunas do material

Lugares em que as aulas se contradizem ou ficam em silêncio. A regra do projeto é que o material vence, mas quando
ele diverge de si mesmo alguém decide.

- **Coluna "Dono":** quem decide. **V** é o Vinicius; **C** é o Claude, que decide e registra.
- **Coluna "Recomendação":** a minha, que o Vinicius pode mudar.

O que é decisão do Vinicius vai para a tabela "Decisões pendentes" do `docs/ROTEIRO.md`. Depois de decidido, vai para
`docs/DECISOES.md`.

## Decisões do Vinicius

| # | Conflito | Onde | Recomendação | Dono | Quando |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Faixas: quantis do histórico do canal (p25/p75/p95) × cortes fixos no alvo relativo (−0,5/0,5/1,5) | 1.6 × 2.4 | quantis do canal: a faixa fala a língua do canal da marca; os cortes fixos ignoram que cada canal tem a sua dispersão | V | Fase 1 |
| 2 | Lacuna do histórico: só posts maduros (30 dias) × defasagem de 7 dias × lacuna como experimento (0/7/30) | 2.2 × 7.1 × 3.5 | 30 dias (princípio 2) até a medição da maturação (19–22/10); rodar o experimento 0/7/30 da 3.5 junto com a M3/M4 | V | Fases 2 e 4 |
| 3 | Modelo principal: CatBoost × LightGBM | 3.4 × 3.5–6.x | LightGBM: é o que 3.5, 3.6, 4.8, 5.x e 6.x usam (quantis, `pred_contrib`, CQR à mão) e já está instalado; o CatBoost vira experimento opcional | V | Fase 2 |
| 4 | Critério de sucesso: ~60% par-a-par × ganho ≥ +0,03 com IC pareado × IC sem 50% | 1.1/2.1/2.4 × 2.5 × 2.4 | pré-registrar em `HIPOTESES.md`: par-a-par intra-canal do GBM acima do baseline, com IC pareado por canal sem zero; os ~60% entram como referência | V | Fase 1 |
| 5 | Tamanho do teste: 3 a 6 meses recentes, antes do último mês × 15–20% × 20% | 2.3 × 1.3 × 7.1 | 3 a 6 meses (2.3), começando em 01/04/2026 ou depois, que é o que nunca foi olhado | V | Fase 1 |
| 6 | VLM: obrigatório na camada leve × "recomendado" × fora (decisão de 08/10) | índice do hub e 4.1 × autoavaliação × CLAUDE.md antigo | rubrica curta (5 a 8 campos) via Gemini por URL, numa amostra, se a Fase 4 estiver no ritmo; senão, limitação documentada | V | Fase 4 |
| 7 | Background do SHAP: intervencional com background do canal × `tree_path_dependent` sem background | 5.6 × 6.3 | o da 6.3, porque o alvo do Prévia já é relativo ao canal, mais o `assert` de aditividade da 5.6 | V | Fase 5 |
| 8 | Explicação pelo MCP: o servidor não tem LLM, então a explicação é o template; o explicador LLM fica fora do MCP | 6.5 × 5.7 | seguir a 6.5: template no servidor; explicador LLM só no grafo; o README diz isso | V | Fase 5 |
| 9 | Limiares de histórico: 30 Shorts para o canal entrar; 10 posts para o alvo relativo; 3 para o cold start | 1.4 × 1.6 × 2.1 | manter os três, um em cada papel; `canal_novo` (5.4) com menos de 10 (#20 do ROTEIRO) | V | Fase 1 |
| 10 | Teto de 400 Shorts por canal | 1.4 × código da 1.5 (sem teto) | já decidido em 08/10: é parâmetro do experimento; resta o valor e a versão manchete (#8 do ROTEIRO). O peso dos 3 canais dominantes é a #6 | V | Fase 1 |
| 12 | Regra de "fora da distribuição": distância ao 10º vizinho num PCA padronizado (p99) × similaridade de cosseno ao vizinho mais próximo (p1) | 5.4 × 6.x | a da 6.x, no mesmo espaço dos comparáveis (entra na #14 do ROTEIRO, regras do flag) | V | Fase 4 |
| 13 | Payload do explicador: o da 5.7 tem mais campos que o `ConfidenceFlag` da 6.1; a 6.3 perdeu a checagem "sem ressalvas" | 5.7 × 6.3 | o payload da 6.3, mais a checagem de ressalvas da 5.7 (#23 do ROTEIRO) | V | Fase 5 |
| 14 | Setores: o material usa `Literal` com 3 setores; o Prévia tem 2 | 6.1, 5.4 | `Literal["beleza", "automotivo"]`, e a regra `setor_desconhecido` vira erro de validação do contrato (#22 do ROTEIRO, na A1) | V | Fase 3 |
| 19 | Carregamento do modelo de uso. A 2.5 e a 7.1 pedem o MLflow (`log_model`, URI no config, "nunca o último .pkl da pasta", `@champion` promovido por pessoa); a 6.1, a 6.5 e a 6.6 carregam de uma pasta de artefatos com variável de ambiente (`GALAXIES_ARTEFATOS`) | 2.5, 7.1 × 6.1, 6.5, 6.6 | a 6.x (pasta mais variável de ambiente), com o bundle registrado como artefato do run de treino final e o run id no `model_version` (F23 da auditoria; #37 do ROTEIRO) | V | Fase 5 |

## Decisões do Claude (registradas em DECISOES)

| # | Conflito | Onde | Decisão |
| :--- | :--- | :--- | :--- |
| 11 | Unidade do bootstrap: post × canal | 3.2/3.4 × 2.4/2.6/4.8 | canal, sempre (posts do mesmo canal não são independentes) |
| 15 | Nomes de colunas: o coletor (`video_id`, `likes`), o briefing (`post_id`, `metric_target`), o módulo 2 (`canal`, `data_pub`, `y`) e a 4.8 (`id_canal`) | 1.5 × 1.8 × 2.x × 4.8 | a base mínima do Prévia (`post_id`, `channel_id`, `published_at`...) é a referência; o código da aula se adapta a ela |
| 16 | Segfault no macOS: importar `lightgbm` antes de `torch` × wrapper com `n_jobs=1` × processos separados × `OMP_NUM_THREADS=1` | 4.1 × 3.6 × 4.8 × 6.4 | extrair com torch num processo e treinar noutro (cache em parquet); `OMP_NUM_THREADS=1` no `__init__` quando o FAISS entrar. 20/10, medido no Mac M4: importar o `lightgbm` antes não basta (segfault no 1º `encode`, também só prevendo); no processo que serve a tool, `previa.openmp.um_runtime()` fixa `OMP_NUM_THREADS=1` antes dos imports |
| 17 | Faixa de complexidade do GBM: profundidade 3–6 ou 15–63 folhas × 7–63 folhas ou profundidade 4–8 | 3.3 × 3.4 | partir de 15 folhas e `min_child_samples` de 20 a 40, com early stopping temporal; ajustar só na validação |
| 18 | `min_child_samples` igual nos três quantis; sem early stopping por quantil; quantil com interpolação | código da 6.1 × 5.2/5.3 | seguir a 5.2 e a 5.3: folhas maiores nos extremos, early stopping com `metric="quantile"`, quantil por estatística de ordem |
| 19 | Onde mora o modelo de uso: registro do MLflow (`models:/...`) × pasta de artefatos com variável de ambiente | 2.5 NP e 7.1 × 6.1, 6.5 e 6.6 | decisão #37 (21/10, A): a tool lê da pasta (`GALAXIES_ARTEFATOS`), que funciona em qualquer processo MCP; o mesmo pacote é registrado como artefato do run do treino final, e o run id vai no `model_version` de toda resposta |

## Lacunas

| Lacuna | Onde | O que fazemos |
| :--- | :--- | :--- |
| Prompt injection não aparece na trilha de ML; só a EDD 8 cita, como red teaming | 5.7, 6.x | princípio 8: a legenda vai ao LLM como dado delimitado; teste com legenda que contém instruções (A4 e A7); o título dos comparáveis vai truncado ao payload |
| LangGraph: a própria 6.2 o trata como opcional (o MCP usa a função pura), mas o hub o põe no núcleo | 6.2 × hub | versão mínima do `StateGraph` (A6); o ReAct é opcional |
| Números diferentes para o mesmo caso Lumina entre aulas (4.380 × 3.762 × 4.237) | 5.x × 6.x | ilustrativos; não citar como resultado do Prévia |
