# Relatório de experimento: o Prévia ordena criativos melhor que o histórico do canal?

Segue o template de relatório do hub (hipótese, setup, resultados, análise de erros e conclusão). Todo número tem o
run do MLflow que o reproduz; o detalhe de cada hipótese está em [`HIPOTESES.md`](../HIPOTESES.md), e o de cada
métrica, no [model card](MODEL_CARD.md).

## Hipótese

**H1, escrita antes de rodar:** entre dois Shorts do mesmo canal publicados a até 14 dias um do outro, o Prévia acerta
qual rende mais curtidas mais vezes que o baseline (a mediana das curtidas dos Shorts maduros anteriores do canal).

**Critério de sucesso, fixado antes do teste (decisão #5):** um ganho de pelo menos +3 pontos na métrica principal,
com IC 95% sem o zero, no teste temporal, e um ganho com IC sem o zero em canais nunca vistos.

As hipóteses de bloco (H2 a H11) perguntam, uma a uma, se cada grupo de features ou escolha de treino melhora o
modelo. Cada uma tem mecanismo, métrica e limiar escritos antes do run.

## Setup

- **Base:** 31.303 Shorts rotulados de 70 canais brasileiros de beleza e automotivo (YouTube Data API v3), congelada
  na versão v1 ([data card](DATA_CARD.md)). Rótulo: as curtidas no primeiro snapshot com 30+ dias de idade.
- **Alvo:** o log da razão entre as curtidas do Short e a mediana madura do canal.
- **Split por data:** treino de jul/2021 a jan/2026, gap de 30 dias, calibração em fev e mar/2026 e teste de 01/04 a
  14/09/2026, aberto uma única vez. As escolhas foram feitas no walk-forward de 2025, dentro do treino.
- **Dois cenários:** o temporal (o futuro de todos os canais) e o de canais nunca vistos (grupos de canais fora do
  treino).
- **Métrica principal:** o acerto par a par dentro do canal, na janela de 14 dias, com cada canal pesando pelo seu
  número de Shorts. Ao lado: o MAE em log, o Spearman e a cobertura do intervalo de 80%.
- **Incerteza:** todo IC 95% é bootstrap de canais inteiros, pareado quando compara dois modelos.
- **Régua dos blocos (decisão #27):** um bloco fica se melhorar a métrica principal ou o MAE em log com IC sem o
  zero, sem piorar a outra.
- **Modelo:** LightGBM com perda de Huber, early stopping nos últimos 180 dias do treino, seed fixa
  (`configs/experimentos/final.yaml`).

## Resultados

**Teste final** (run `6b1f7e77`, 7.075 Shorts de 70 canais):

| Cenário | Baseline | Prévia | Δ [IC 95%] |
| :--- | ---: | ---: | ---: |
| Temporal | 49,7% | 55,3% | +5,6 pontos [+3,6; +7,1] |
| Canais nunca vistos | 49,7% | 53,2% | +3,5 pontos [+1,9; +4,9] |

O critério de sucesso foi cumprido nos dois cenários. O erro do número caiu 7,5% [1,8%; 12,7%] no MAE em log (0,925
contra 0,999; erro típico de ×2,52), e o Spearman global é 0,75 [0,55; 0,86], dentro da faixa de 0,55 a 0,85 da
literatura e abaixo do alarme de vazamento. O intervalo de 80% cobriu 76,5%, abaixo do alvo, caindo de 84% em abril
para 66% em setembro.

**Ablação** (walk-forward de 2025; Δ em pontos na métrica principal e no MAE em log):

| Bloco ou escolha | Δ métrica principal | Δ MAE em log | Ficou? |
| :--- | ---: | ---: | :--- |
| Histórico do canal (H2) | −0,3 [−1,6; +0,6] | −0,011 [−0,032; +0,008] | não |
| Texto do título (H4) | +0,3 [−0,4; +1,0] | +0,004 [−0,006; +0,015] | não |
| Idade do rótulo como controle (H5) | +0,3 [−0,4; +1,2] | −0,018 [−0,028; −0,007] | sim |
| Capa: SigLIP 2, sondas, aparência (H7) | +2,3 [+1,2; +3,3] | −0,012 [−0,023; −0,003] | sim |
| Rubrica do VLM, na amostra (H8) | −5,5 [−14,9; +3,4] | −0,005 [−0,021; +0,010] | não (inconclusivo) |
| Vídeo e áudio, na amostra (H9) | −3,9 [−12,7; +4,8] | +0,000 [−0,014; +0,015] | não (inconclusivo) |
| Rosto e texto na capa, por detectores (H10) | −0,5 [−1,1; +0,3] | −0,002 [−0,013; +0,006] | não |
| Perda de Huber no lugar da L2 (H11) | +2,0 [+1,2; +3,0] | −0,043 [−0,062; −0,026] | sim |

A capa é o único bloco de conteúdo que se sustenta nas três leituras da ablação (o degrau, a remoção e acima do
ruído). A rubrica e o vídeo só existem numa amostra de 1.400 Shorts, cerca de 200 pares na janela, e o IC é largo
demais para concluir.

## Análise de erros

Os 20 piores erros do walk-forward (run `e684ac8c`, `uv run python -m previa.erros e684ac8c`) são todos
subestimações, de 416× a 2.249×. Pela regra escrita antes de olhar (views 100× acima da mediana do canal e menos de 5
comentários por mil curtidas), 12 têm o sinal de impulsionamento pago, 3 são virais com conversa real e 5 são
ambíguos. Nenhum vem de canal novo, de alvo imaturo ou de dado quebrado. A lista, com o link de cada Short, fica em
`data/erros/` (fora do git, pelos termos da API).

## Conclusão

O Prévia ordena os criativos de um canal melhor que o histórico dele: +5,6 pontos no futuro e +3,5 em canais nunca
vistos, com o critério fixado antes do teste. O ganho de conteúdo vem da capa; o vídeo, o áudio e a rubrica do VLM não
puderam ser medidos com precisão, porque só existem numa amostra. O maior erro é a mídia paga, que a API não mostra, e
o intervalo envelhece com o modelo: a resposta foi retreinar o modelo em uso (run `5373921a`), que não tem um teste
próprio. Os próximos passos, em ordem de impacto esperado, estão na seção 9 do README.
