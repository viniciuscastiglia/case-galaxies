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
| Teto por canal | 400 Shorts rotulados mais recentes **[decisão do Vinicius]**; desde 08/10 aplicado nas linhas de treino, não na exportação | Um canal que posta 5 Shorts por dia não pode dominar a base (aula 1.4) |
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

## Expansão por busca (08/10)

Decisão do Claude, delegada pelo Vinicius em 08/10. Com 52 canais, a base estava abaixo da meta (60+) e sem nenhum
criador de beleza com menos de 100 mil inscritos. As estratégias gratuitas da aula 1.4 (handles citados nas
descrições e canais em destaque) renderam 30 candidatos e nenhuma inclusão: a maioria dos @ citados é do Instagram.

- **Estratégia 4 da aula 1.4:** `python -m previa.curadoria buscar` lê `data/termos_busca.csv` e usa o `search.list`
  só para descobrir canais, no balde próprio de 100 chamadas por dia. Cada busca fica em `data/buscas.csv`.
- **Termos de nicho, não genéricos:** o ranking de relevância favorece canais grandes. "Maquiagem" devolve as
  gigantes que já estão na base; "maquiagem de farmácia" ou "cabelo cacheado finalização" puxam criadores menores.
- **Modo vídeo** (fora da caixa, dentro da regra da aula): busca vídeos curtos dos últimos 90 dias sobre o termo e
  fica com o canal de quem postou. Acha quem posta Shorts com frequência, que é o que os critérios 3 e 4 pedem.
- **Pré-filtro com metadados** (1 unidade a cada 50 canais): país diferente de BR, inscritos ocultos ou menos de 30
  vídeos no total não passariam nos critérios, e não gastam a avaliação completa (até 21 unidades por canal).

**Regra de seleção**, escrita antes de ver quem passou na avaliação:

1. Só entra quem passa nos critérios automáticos (3 a 6) e na revisão (1, 2 e 7), feita pela descrição do canal e
   pelos títulos dos uploads recentes.
2. Teto do plano: até 70 canais, cerca de metade em cada setor (até 35 por setor).
3. Prioridade das vagas: criadores de beleza entre 10 mil e 1 milhão de inscritos (a lacuna principal), depois
   criadores de automotivo entre 10 e 100 mil, depois o que equilibrar setor × tipo × faixa.
4. Dentro de um estrato: no máximo 3 canais vindos do mesmo termo, para variar os nichos, e a ordem da busca como
   desempate. Curtidas e visualizações nunca entram na escolha.
5. Quem passa mas fica sem vaga vira `não` com o motivo "reserva". A reserva serve para os criativos de teste da
   demo, que o briefing pede de canais fora do treino.

**Resultado (08/10).** 24 termos (18 de beleza, 6 de automotivo), 24 chamadas de busca e ~8.500 unidades da cota
geral na avaliação.

| Etapa | Canais |
| :--- | ---: |
| Candidatos novos da busca, depois do pré-filtro | 614 |
| Reprovados nos critérios automáticos (53 com menos de 30 Shorts, 34 inativos, 11 com curtidas ocultas) | 98 |
| Passaram nos critérios automáticos | 516 |
| Revistos (os 3 primeiros de cada termo nos estratos prioritários) | 65 |
| Recusados na revisão: inglês, fora do setor, compilações, divulgação de apostas | 22 |
| **Incluídos** | **18** |
| Reserva revista / reserva sem revisão | 25 / 451 |

Na prática, a regra 4 virou uma divisão das 13 vagas de beleza por nicho (maquiagem 4, pele 3, cabelo 3, perfumaria
2, cuidados masculinos 1) e por faixa (6 entre 100 mil e 1 milhão, 7 entre 10 e 100 mil), na ordem da busca. Essa
divisão foi decidida depois da revisão, sem olhar curtidas. As 5 vagas de automotivo foram para 4 criadores entre 10 e
100 mil e uma loja de estética automotiva, porque a célula de marcas de automotivo nessa faixa tinha só 1 canal.

A base passou de 52 para **70 canais, 35 por setor**. Os criadores de beleza foram de 7 para 20, e a faixa de 10 a 100
mil, que estava vazia, agora tem 7 criadores de beleza. A primeira coleta dos 18 canais novos fica para a coleta
diária de 09/10, com a cota cheia.

## Fontes

| Fonte | Papel | O que não representa |
| :--- | :--- | :--- |
| YouTube Data API v3 | base principal: treino, validação e teste | retenção, CTR e impressões (só o dono do canal vê); estatísticas do dia da publicação |
| SMTPD | apoio opcional, para praticar alvo em log e baselines | Shorts (a duração média é de ~30 min), marcas brasileiras, o período depois de 2025 |
| MicroLens-100K | apoio opcional, para ensaiar a extração multimodal (semana 3) | YouTube, português, fracassos (só vídeos-semente com mais de 10 mil curtidas) |

## O que a base não representa

Lista do Vinicius em 05/10 (exercício da aula 1.3), com os complementos da revisão. Vai para o data card (aula 1.8).

- **Vídeos longos:** só Shorts, por escopo.
- **Canais com menos de 30 Shorts:** uma marca que acabou de começar a postar Shorts está fora do que o modelo viu.
  Mitigação: previsões para canais com pouco histórico saem marcadas com baixa confiança (aula 5.4).
- **Retenção, CTR e impressões:** só o dono do canal vê. Não podem ser features nem aparecer na explicação.
- **O período depois de ~19/09/2026:** mudanças no algoritmo ou nas tendências depois do corte de rótulos (drift,
  aula 1.2).
- **Outros setores, países e idiomas:** só beleza e automotivo, canais brasileiros, em português.
- **Canais pequenos:** a lista inicial foi feita de memória e favorece canais grandes (viés de sobrevivência, risco 3).
  A busca de 08/10 trouxe criadores entre 10 e 100 mil inscritos, mas ainda não há nenhum criador com menos de 10 mil.
  A própria busca tem viés: o ranking de relevância do YouTube decide quem aparece primeiro.
- **Mídia paga:** a API não diz se um Short foi impulsionado, então parte das curtidas pode vir de anúncio.

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
5. **Rede:** em 08/10 a rota IPv6 até o Google parou de responder, e cada chamada da API esperava quatro timeouts
   antes de tentar o IPv4 (o `curl` da checagem de rede não sofre com isso). Mitigação: `preferir_ipv4()` no início
   do coletor e da curadoria, com o IPv6 de reserva.
