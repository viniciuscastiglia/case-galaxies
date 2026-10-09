# Métricas da aula 2.1: teach-back (09/10/2026)

## Com as minhas palavras

A média de erro absoluto pega todos os erros dos modelos e faz a média deles, seja pra baixo ou pra cima.
Utilizamos essa métrica em log e não em valor bruto porque a comparação de um canal que tem 100 curtidas e de um que
tem 100000 tem que ser a mesma individualmente. Spearman global mede a ordem com todos os canais misturados,
Spearman intra-canal mede a ordem dentro de 1 canal. Pro nosso problema o intra-canal é melhor, porque a Lumina tem
1 canal e dentro desse canal quer saber qual Short vai performar melhor.

## Retorno do Claude (para a versão do README)

- **Por que log:** errar pela metade pesa o mesmo num canal de 100 curtidas e num de 100 mil. Em valor bruto, os
  erros dos canais grandes engoliriam a média.
- **Por que o global não basta:** ele sai alto só de saber que canal grande tem mais curtidas. Um modelo pode ir
  bem no global e não saber escolher entre dois Shorts do mesmo canal.
- **Intra-canal:** a ordem dentro de cada canal, depois a média entre os canais.
- **Em língua de gente:** MAE em log de 0,72 = a previsão costuma errar pela metade ou pelo dobro.

## A régua do log (o que fez o 0,69 fazer sentido)

| Curtidas | `y = log1p(curtidas)` |
| ---: | ---: |
| 100 | 4,6 |
| 200 | 5,3 |
| 400 | 6,0 |
| 800 | 6,7 |
| 1.600 | 7,4 |

Cada dobra anda 0,69 na régua. O MAE em log é a distância média, nessa régua, entre o previsto e o real: 0,72 é
pouco mais de um passo de dobra.

## Tabela da aula 2.1, lida

| | MAE (log) | Spearman global | Spearman intra |
| :--- | ---: | ---: | ---: |
| Mediana do canal | 0,792 | 0,840 | −0,197 |
| LightGBM | 0,722 | 0,864 | 0,157 |

O LightGBM sabe o tamanho dos canais tão bem quanto o baseline (global igual) e começou a saber qual post de um
mesmo canal rende mais (intra de negativo para positivo). O intra negativo do baseline vem da regressão à média: a
mediana sobe depois de um post bom, e o seguinte costuma voltar ao normal.
