# Caderno de decisões

Cada decisão do projeto, com quem decidiu, a aula que a sustenta e o porquê. É a base da seção "Metodologia" do
README e da defesa na demo. Quem decide: **V** = Vinicius; **C** = Claude (encanamento ou tarefa delegada). As
decisões pendentes ficam em `docs/ROTEIRO.md`; os conflitos do material, em `docs/material/conflitos.md`.

Para registrar: uma linha por decisão, a mais nova embaixo. Quando a decisão mudar, a linha antiga fica e uma nova
linha diz o que mudou e por quê.

| Data | Decisão | Quem | Aula | Por quê | Onde está |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 24–27/09 | Setores: beleza e automotivo | V | hub (escopo: ≥2 de 3) | moda e beleza se misturam em canais brasileiros e geram ruído de rótulo; o contraste entre beleza e automotivo torna informativo o experimento de especialista (D7 em `docs/arquivo/PROJECT_v0.md`) | `CLAUDE.md`, `data/canais.csv` |
| 24–27/09 | Recorte: canais brasileiros, em português | V | 1.4 (critério 2) | público homogêneo; a persona é brasileira | `docs/PLANO_DA_BASE.md` |
| 24–27/09 | Alvo: curtidas, não views | V | 1.7 | as views de Shorts mudaram de regra em 31/03/2025 e 24/08/2026 | `README.md` |
| 24–27/09 | Nome Prévia, pacote `previa`, repositório privado no GitHub | V | — | — | `pyproject.toml` |
| 30/09 | O material da liga é a especificação; o desenho anterior foi arquivado | V | — | material oficial, completo e com versões testadas | `docs/arquivo/PROJECT_v0.md` |
| 30/09 | Critérios de inclusão escritos antes de olhar resultados: setor, idioma e país, atividade em 90 dias, 30+ Shorts, curtidas e inscritos visíveis, exclusões | V + C | 1.4 | evitar viés de seleção | `docs/PLANO_DA_BASE.md` |
| 30/09 | Coletor incremental pela playlist de uploads; DuckDB com snapshots diários; Pydantic no registro e pandera na tabela | C | 1.5, 1.8 | encanamento que segue as aulas; idempotente e auditável | `src/previa/coletor.py`, `src/previa/esquema.py` |
| 30/09 | Regra de Short: até 60 s antes de 15/10/2024, até 180 s depois, e não horizontal (formato desconhecido conta como Short) | C | 1.5 | a regra documentada da aula; o `isShort` não existe | `docs/DATA_CARD.md` (Coleta) |
| 05/10 | Revisão humana: 52 canais incluídos (Localiza fora pelo critério 1; Garagem do Bellote dentro depois da contagem completa) | V | 1.4 | critérios escritos | `data/canais.csv` |
| 05/10 | Problema no README: unidade com 10+ Shorts anteriores; alvo = log da razão entre as curtidas e a mediana dos posts anteriores, medido aos 30+ dias; sucesso = par-a-par, Spearman e MAE em log contra a mediana do canal, com validação temporal | V | 1.1, 1.6 | — | `README.md` |
| 08/10 | Expansão por busca para 70 canais, 35 por setor (criadores de beleza de 7 para 20) | C (delegada) | 1.4 (estratégia 4) | a base estava abaixo da meta de 60 canais e quase sem criadores | `docs/PLANO_DA_BASE.md` (Expansão por busca) |
| 08/10 | Exportação sem teto; o teto vira parâmetro do experimento, só nas linhas de treino. O valor e a versão manchete são do Vinicius (#8 do ROTEIRO) | C | 1.4 | o histórico do canal precisa de todos os posts (paridade treino/uso) | `docs/DATA_CARD.md` (Coleta) |
| 08/10 | `duplicate_group` marca cópias (título normalizado + duração); nada é apagado; a regra fica para a 2.2/2.3 | C | 2.2 | decidir por critério escrito, nunca pelas curtidas | `src/previa/esquema.py` |
| 08/10 | Trava provisória do teste: nada publicado a partir de 01/04/2026 entra no desenvolvimento | C | 1.2, 2.3 | reservar o teste antes de olhar qualquer métrica; as datas definitivas são da Fase 1 | `configs/avaliacao.yaml`, `previa.dados.carregar_base` |
| 08/10 | A coleta diária valida com pandera, audita a regra dos 30 dias e desliga em 30/10 (apagamento) | C | 1.8 | termos da API (III.E.4.d) | `scripts/coletar.sh`, `src/previa/coletor.py` |
| 08/10 | Data card v0.1, com os vieses escritos pelo Vinicius | V | 1.8 | — | `docs/DATA_CARD.md` |
| 12/10 | Auditoria de features: inscritos da coleta são risco temporal (retreinar sem eles e comparar); `target_age_days` é candidata a variável de controle (fixada em 30 na previsão); views, curtidas e comentários do vídeo ficam fora | V | 2.2, 1.7 | treino e produção calculam cada feature do mesmo jeito | `docs/AUDITORIA_FEATURES.md` |
| 13/10 | Reta final: os exercícios das aulas ficam para depois de 23/10. O Vinicius escreve o coração das peças reais (M1–M4, M6, M8, M9, A1, A2 e A4–A8) e decide o resto; o Claude faz o encanamento | V | — | tempo até 23/10; foco de carreira em ML e AI Engineering; protagonismo | `docs/ROTEIRO.md` |
| 13/10 | Bootstrap sempre por canal | C | 2.4, 2.6, 4.8 | posts do mesmo canal não são independentes | `docs/material/conflitos.md` (#11) |
| 13/10 | Nomes de colunas: a base mínima do Prévia é a referência, e o código das aulas se adapta a ela | C | 1.8 | um só vocabulário no repositório | `docs/material/conflitos.md` (#15) |
| 13/10 | macOS: extração com torch num processo e treino em outro (cache em parquet); `OMP_NUM_THREADS=1` quando o FAISS entrar | C | 4.8, 6.4 | evita os segfaults conhecidos de OpenMP | `docs/material/conflitos.md` (#16) |
| 13/10 | GBM: partir de 15 folhas e `min_child_samples` de 20 a 40, com early stopping temporal; quantis com folhas maiores nos extremos, early stopping por quantil e quantil conformal por estatística de ordem | C | 3.3, 3.4, 5.2, 5.3 | seguir as aulas onde o código da 6.1 simplificou | `docs/material/conflitos.md` (#17, #18) |
