# Prévia

Prevê, antes da publicação, as curtidas de um YouTube Short de beleza ou automotivo de um canal brasileiro, com
intervalo, ranking entre criativos e explicação, exposto como função Python e servidor MCP para a Galaxies.

## O problema (aula 1.1)

1. **Decisão:** o resultado numérico do modelo preditivo vai apoiar o time de marketing da marca, que está usando a
   plataforma da Galaxies (via Agente Nexus), a decidir qual Short, dentre N que a marca quiser comparar, deve ser
   impulsionado antes da publicação.
2. **Unidade:** um Short de um canal já existente, com pelo menos 10 Shorts publicados anteriormente. Se tiver menos
   de 10, não será rejeitado, mas marcado com baixa confiança.
3. **Momento:** antes da publicação do vídeo. Nesse momento já se sabe o canal onde o vídeo será postado, o histórico
   desse canal até a data (somente posts anteriores ao vídeo), o vídeo, a legenda, o horário planejado e o setor.
4. **Alvo:** o log da razão entre as curtidas do post e a mediana dos posts anteriores do canal, medido quando o vídeo
   tiver pelo menos 30 dias de postado. <!-- TODO(Vinicius): o porquê do alvo, depois da aula 1.6 -->
5. **Sucesso:** para avaliar a ordenação, acurácia par a par e Spearman; para o erro, MAE em log (quanto o número
   previsto erra em média). O modelo deve bater o baseline "todo vídeo do canal vai ter a mediana do canal". A
   validação segue a ordem temporal: o modelo é treinado no passado e validado e testado em ordem cronológica.
