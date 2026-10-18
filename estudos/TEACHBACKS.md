# Teach-backs

O banco de explicações do Vinicius: cada peça que ele vai apresentar, em 3 a 5 linhas, nas palavras dele, com a data
e a aula. Serve para ensaiar a demo e responder à banca. Formato do `/teach-back`: as linhas dele exatamente como ele
escreveu, e a correção do Claude em uma linha.

Antes deste arquivo existir:

- **2.1, métricas (09/10):** `estudos/aula_2_1_metricas.md`.
- **2.2, vazamento (12/10):** teach-back aprovado, registrado no `docs/DIARIO.md`.

<!-- novas entradas abaixo, a mais recente em cima -->

## M1, splits por data e "canal novo no futuro" (aula 2.3) · 17/10/2026

> na primeira funcao estamos fazendo a divisao de todos os dados que nos temos em treino, calibracao e teste, o treino comeca em uma data que nós definimos, a data que o youtube lancou os youtube shorts e vai ate o recorte de calibracao, porem entre o treino e a calibracao temos um gap de 30 dias, que é o tempo que um shorts leva pra ser considerado maduro, momento em que as curtidas comecam a establizar, a calibracao termina na data que comeca o periodo de teste, a parte mais recente da base de dados destinada justamente para fazermos o teste final, importante lembrar que depois do teste nao podemos ajustar o modelo baseado no resultado do teste, todos os ajustes ocorrem com os dados que estao da calibracao. Essa funcao seria o melhor cenario pra um canal a conhecido pelo previa, em que vamos prever como um shorts nunca visto desse canal vai performar. Na segunda funcao estamos simulando uma divisao da nossa base de dados destinada a um canal que o previa nunca viu, estamos treinando o modelo deixando canais de fora propositalmente de forma que conseguimos medir a performance dele em canais nunca vistos no treino. Fazemos isso dividindo nossa base de dados ainda de maneira temporal, mas tambem em 4 grupos, de forma que 1 dos grupos tenha canais nunca vistos pelo modelo, que vao ficar escondidos dele para o momento de teste, dessa forma, quando a Lumina, empresa que o previa nunca viu, perguntar se o short A ou B vai performar melhor, a segunda funcao da a resposta mais confiavel para essa pergunta

**Correção:** os ajustes do modelo (hiperparâmetros, features) são escolhidos no walk-forward dentro do treino (aula
2.3, erro comum 4). A calibração só mede os erros do modelo já fechado, para dar a largura do intervalo de 80% (M6);
escolher o modelo olhando para ela deixa o intervalo estreito demais.

**Pergunta:** escolher as folhas olhando a calibração deixa o intervalo mais largo ou mais estreito? **Ele:** mais
estreito, porque o modelo se adapta ao conjunto que mede a largura; e, com poucos dados, há overfitting. Certo. O
mecanismo: a configuração vencedora acertou a calibração em parte por sorte, então os erros medidos ali ficam menores
que os de abril em diante, e o intervalo cobre menos que 80%. Os dois pontos dele são o mesmo efeito.
