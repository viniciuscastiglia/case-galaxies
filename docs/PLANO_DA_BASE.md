# Plano da base

> Exercício da [aula 1.3](https://inteliacademyclub.github.io/autoestudos/aulas/ml-preditivo-e-agentes/formulacao-e-dados/fontes-de-dados-e-plano-da-base/)
> da trilha da liga. Rascunho do Claude em 30/09/2026, com números **propostos**. As linhas marcadas com
> **[decisão do Vinicius]** esperam confirmação na revisão dos canais.

A base tem 60 a 70 canais brasileiros de beleza e automotivo, coletados todo dia pela YouTube Data API. O rótulo de
cada Short é o primeiro snapshot com 30 dias ou mais.

## Metas

| Meta | Valor proposto | Por quê |
| :--- | :--- | :--- |
| Canais incluídos | 60 a 70, cerca de metade em cada setor | Com poucos canais, o modelo aprende "quem é quem" e não "o que funciona" (aula 1.3) |
| Marcas | pelo menos 10 por setor **[decisão do Vinicius]** | O cliente da Galaxies é a marca (aula 1.4) |
| Faixas de inscritos | nenhuma faixa vazia entre 10 mil e 1 milhão, em cada setor | A Lumina tem 180 mil inscritos; sem canais desse porte, a previsão vira extrapolação |
| Volume | 5 a 15 mil Shorts rotulados | Abaixo de 5 mil, o teste temporal fica pequeno demais (aula 1.3) |
| Teto por canal | 400 Shorts rotulados mais recentes **[decisão do Vinicius]** | Um canal que posta 5 Shorts por dia não pode dominar a base (aula 1.4) |
| Rótulo | primeiro snapshot com 30 dias ou mais | Curtidas ainda crescem nas primeiras semanas (aula 1.7) |
| Setor automotivo | carros, picapes e SUVs: montadoras, varejo e serviços, revistas e criadores; motos ficam fora **[decisão do Vinicius]** | Setor homogêneo e coerente com o que a rubrica do VLM vai perguntar (carro em cena, interior, direção) |

## Critérios de inclusão

Escritos antes de olhar resultados (aula 1.4). A avaliação automática (`python -m previa.curadoria avaliar`) mede os
itens 3 a 6; os itens 1, 2 e 7 são conferidos na revisão humana.

1. **Setor:** a maioria dos Shorts do canal é de beleza ou de automotivo.
2. **Idioma e país:** conteúdo em português, para público brasileiro. O campo de país da API é opcional, então a
   conferência final é humana.
3. **Atividade:** pelo menos 1 Short nos últimos 90 dias.
4. **Histórico:** pelo menos 30 Shorts.
5. **Curtidas visíveis:** `likeCount` presente na maioria dos Shorts.
6. **Inscritos visíveis:** o canal não esconde o número de inscritos.
7. **Exclusões:** canais de cortes ou reuploads, compilações, agregadores e conteúdo infantil.

## Fontes

| Fonte | Papel | O que não representa |
| :--- | :--- | :--- |
| YouTube Data API v3 | base principal: treino, validação e teste | retenção, CTR e impressões (só o dono do canal vê); estatísticas do dia da publicação |
| SMTPD | apoio opcional, para praticar alvo em log e baselines | Shorts (a duração média é de ~30 min), marcas brasileiras, o período depois de 2025 |
| MicroLens-100K | apoio opcional, para ensaiar a extração multimodal (semana 3) | YouTube, português, fracassos (só vídeos-semente com mais de 10 mil curtidas) |

## Custo em cota

| Etapa | Unidades | Quando |
| :--- | ---: | :--- |
| Avaliação dos ~110 candidatos (até 10 páginas de uploads + 10 lotes de vídeos cada) | até ~2.200 | uma vez |
| Primeira coleta completa: 65 canais com 300 uploads cada (`custo_coleta` da aula 1.3) | 845 | uma vez |
| Coleta diária: uma página por canal, lotes dos vídeos novos, snapshots dos jovens, renovação dos 30 dias | ~100 a 300 | todo dia |

A cota diária é de 10 mil unidades. O coletor para sozinho em 9 mil, para sobrar margem.

## Cronograma

| Data | Marco | Corte de publicação |
| :--- | :--- | :--- |
| 30/09 ou 01/10 | chave da API criada; avaliação automática dos candidatos | — |
| até 02/10 | revisão dos canais (Vinicius) e primeira coleta completa | 02/09 |
| todo dia, 05h15 | coleta incremental, snapshots dos Shorts jovens e renovação dos dados | — |
| 12/10 | congelar a base de treino | 12/09 |
| 19/10 | última coleta de rótulos | 19/09 |
| 30/10 | apagar os dados brutos (termos do YouTube, aula 1.8) | — |

## Riscos

1. **Poucas marcas brasileiras com 30 Shorts ou mais.** Mitigação: expandir por colabs e canais em destaque. Se ainda
   faltar, documentar no data card, sem afrouxar critério depois de ver as curtidas.
2. **Handles errados na lista inicial**, que foi feita de memória. Mitigação: busca de reserva pelo nome do canal
   (até 90 buscas por dia) e correção manual na revisão.
3. **Viés de sobrevivência:** uma lista de memória favorece canais grandes. Mitigação: expansão por colabs e destaques,
   e a coluna `fonte`, que permite medir depois se o modelo erra mais nos canais de uma origem.
4. **Mac dormindo às 05h15.** Mitigação: o launchd roda a coleta quando o Mac acorda, e a coleta é idempotente.
