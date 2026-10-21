# Hipóteses do Prévia

Escritas **antes** de rodar (aula 2.5: o "jardim dos caminhos que se bifurcam"). Cada hipótese diz o que se espera,
por quê, e como se decide. Depois de rodar, entram o resultado (com o run id do MLflow) e a decisão; o que estava
escrito antes não muda.

Regras de todas:

- **Métrica:** a manchete da #6, o par-a-par na janela de 14 dias com cada canal pesando pelo nº de Shorts dele,
  com IC 95% do bootstrap pareado por canal (aula 2.4).
- **Régua dos blocos, H2 a H6 (#27):** duas réguas, porque o Prévia entrega a ordem e o número (aula 2.4). A
  variante fica se melhorar a manchete **ou** o erro em curtidas (`mae_log`), com IC 95% pareado por canal sem o
  zero, **sem piorar** a outra (o IC da outra não fica inteiro do lado ruim). Os +3 pontos valem só para a H1.
- **Onde se decide:** as escolhas de modelo (H2 a H8) nos folds de walk-forward de 2025, dentro do treino (a H6
  também no canal novo, decisão de 20/10); o teste
  (abr a set/2026) abre uma vez, no fim, só para a H1 (aula 2.5, erro comum 4).
- **Controle e tratamento** usam a mesma base, os mesmos cortes, a mesma seed e o mesmo modelo, mudando uma coisa.

## H1: o Prévia ordena A e B melhor que o histórico do canal

Data: 2026-10-19 (decisão #5)
Hipótese: o GBM (Opção A: histórico do canal, horário, duração e texto) acerta mais pares de Shorts do mesmo canal,
  publicados a até 14 dias um do outro, que a mediana madura do canal.
Mecanismo: a mediana madura olha o canal como um todo; o modelo olha cada Short. Dentro do mesmo canal, o que muda
  de um Short para o outro mexe em quantas pessoas param para ver: no feed, o vídeo começa sozinho, e o assunto (que o
  título revela), a duração e o horário decidem quem fica até o fim e curte. O modelo aprende esses padrões nos Shorts
  anteriores do canal.
Métrica e limiar: manchete da #6 no teste temporal; aceito se o ganho for >= +3 pontos e o IC 95% do bootstrap
  pareado por canal não incluir zero, e se no "canal novo no futuro" o IC do ganho também não incluir zero.
  Os ~60% da Galaxies são referência, não meta.
Controle: a mediana madura (49,7% na manchete em 2025, IC 49,2% a 50,1%)
Tratamento: o GBM da M4
Resultado:
Decisão:

## H2: histórico do canal no GBM (M3)

Data: 2026-10-19
Hipótese: comparando dois Shorts do mesmo canal, o histórico ajuda pouco na manchete; onde ele ajuda é no número
  (o erro em curtidas), porque o modelo passa a saber o nível recente do canal.
Mecanismo: Shorts publicados a até 14 dias um do outro usam quase os mesmos posts maduros, então o histórico é
  praticamente o mesmo para os dois e não é ele que os diferencia. Na tabela do Acelerados (02 a 10/06/2025),
  históricos parecidos vieram com resultados de 392 a 2.649. O pouco que ele pode ajudar na ordem vem do que ainda
  muda entre os dois: posts novos que amadurecem e a cadência. No número, ele ajuda porque a mediana de tudo
  (1.070) ficou longe do nível recente do canal (os últimos 20 maduros, ~560).
Métrica e limiar: a régua dos blocos (#27), nos folds de walk-forward de 2025: o histórico fica se melhorar a
  manchete ou o `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a outra.
Controle: o GBM sem as features de histórico (a âncora do alvo relativo, a mediana madura do canal,
  continua: a H2 mede o histórico além dela)
Tratamento: o GBM com as 9 features de histórico da 3.5 (`hist_*`, `tendencia`, `dias_desde_ultimo`, `posts_30d`)
Resultado:
Decisão:

## H3: lacuna do histórico (#11)

Data: 2026-10-20
Hipótese: com a lacuna de 7 dias, o modelo erra menos o número (`mae_log`), e a ordem (a manchete) quase não muda.
Mecanismo: na ordem, A e B da mesma semana ganham os mesmos posts recentes, então o histórico continua quase igual
  para os dois (o argumento da H2). No número, o nível do canal 7 dias antes do Short está mais perto do nível do
  dia que o de 30 dias antes (a série da Lumina em junho), e o modelo, que também vê a mediana de todo o passado e o
  desvio, aprende a descontar os picos que passam (a regressão à média da M2). Os posts de 7 a 30 dias já estão
  quase maduros: de 7 a 14 dias, crescem ~2% (+1,9% na medição de 20/10; +0,8% na de 19/10), contra +75% de 0 a 3.
Métrica e limiar: a régua da #27 no walk-forward de 2025, só entre 7 e 30 dias: a de 7 fica se melhorar a manchete
  ou o `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a outra; senão, ficam os 30. A de 0 só mostra
  o otimismo do backtest.
Controle: lacuna de 30 dias (princípio 2; `opcao_a_texto.yaml`)
Tratamento: lacuna de 7 dias, a única candidata (`h3_lacuna_7.yaml`). A de 0 dias (`h3_lacuna_0.yaml`) entra na
  tabela só para mostrar o otimismo do backtest, nunca como candidata: o backtest volta o relógio, mas não as
  curtidas (a foto é de outubro de 2026), e um post com 0 a 3 dias ainda cresce +69% (decisão de 20/10)
Resultado:
Decisão:

## H4: texto (#9)

Data: 2026-10-19
Hipótese: o texto ajuda na ordem e pouco no número; os embeddings (b) ajudam mais que as manuais (a).
Mecanismo: o título não diz nada sobre o nível do canal, só sobre o conteúdo de cada Short, e o que muda de um
  título para outro é o significado: "O AUDI RS6 MAIS EXTREMO QUE EXISTE!" quer dizer carro de sonho na tela, e "POV:
  SEU CARRO SUPERAQUECEU" quer dizer problema chato. Os embeddings captam o significado; as manuais, só a forma
  (pergunta, emoji, hashtags). No número, o grosso do erro vem do nível do canal e dos virais, não do título (3.6:
  o MAE quase não muda, e isso não é motivo para descartar o texto).
Métrica e limiar: a régua dos blocos (#27), nos folds de walk-forward de 2025, cada tratamento contra o GBM sem
  texto. (b) só entra se também ganhar de (a) pela mesma régua; se empatar, fica (a), que é mais simples.
Controle: o GBM sem texto
Tratamento: (a) as features manuais do título e da descrição (CTA, pergunta, hashtags, números, termos do setor);
  (b) as manuais + embeddings multilíngues com PCA de 16 ajustado só no treino
Resultado:
Decisão:

## H5: idade do rótulo (#21)

Data: 2026-10-20
Hipótese: a época (a idade do rótulo) não ajuda na ordem e ajuda no número.
Mecanismo: na ordem, A e B da mesma semana estão na mesma fase da plataforma, então a época é igual para os dois
  (o argumento da H2). No número, de 2021 a 2026 houve mudanças da plataforma inteira (o algoritmo e o público dos
  Shorts) que mexeram nas curtidas de todos os canais ao mesmo tempo. A fase de cada canal o histórico já vê; a
  época vê o que é comum a todos, e, fixada na mais recente do treino, leva para a previsão o nível atual da
  plataforma.
Métrica e limiar: a régua da #27 no walk-forward de 2025: a idade fica se melhorar a manchete ou o `mae_log`, com
  IC 95% pareado por canal sem o zero, sem piorar a outra.
Controle: o GBM sem `target_age_days` (`opcao_a_texto.yaml`)
Tratamento: `target_age_days` no treino, fixada em 30 na previsão (`h5_idade_controle.yaml`). Como as curtidas
  foram todas coletadas em outubro de 2026, a idade é a data de publicação disfarçada: mede a época (do canal e da
  plataforma), não a maturação, que a curva já mediu (0% de crescimento a partir de ~21 dias). O 30 cai no galho
  da idade mais nova do treino (a árvore não extrapola): na previsão, vale "a época mais recente". A opção (b) da
  #21 (só rótulos de até 90 dias) sai: o rótulo mais novo do desenvolvimento tem 188 dias (decisão de 20/10)
Resultado:
Decisão:

## H6: teto por canal nas linhas de treino (#8)

Data: 2026-10-20
Hipótese: com o teto de 400, o modelo melhora o número no canal novo, sem mudar a ordem; no walk-forward, piora
  um pouco.
Mecanismo: vale ter muitos dados, mas concentrados em poucos canais eles fazem o modelo se especializar nos grandes:
  13 canais têm 62% das 20.579 linhas de treino (a Salon Line, 2.082; o canal mediano, 166), e o modelo pesa cada
  linha igual, então aprende ajustes da Salon Line que não necessariamente se repetem em outros canais, como a
  Lumina. Na ordem, o teto não muda: o que decide entre A e B é quem para e fica no feed (H1), um comportamento do
  público parecido em todos os canais, que os grandes também ensinam. No número, os ajustes que valem em muitos
  canais erram menos num canal nunca visto. No walk-forward, os canais são os do treino e os grandes
  dominam a manchete; os Shorts antigos ainda ensinavam coisas úteis sobre eles, então o modelo os conhece um pouco
  menos.
Métrica e limiar: nos dois cenários do run, com papéis diferentes (decisão de 20/10). No "canal novo no futuro",
  o teto tem de melhorar a manchete ou o `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a outra;
  no walk-forward de 2025, não pode piorar (o IC não fica inteiro do lado ruim) em nenhuma das duas.
Controle: sem teto (a manchete, #8)
Tratamento: até 400 Shorts por canal nas linhas de treino (o histórico continua com todos os posts)
Resultado:
Decisão:

## H7: thumbnail (#12)

Data: 2026-10-20
Hipótese: a capa ajuda na ordem e no número, mas pouco.
Mecanismo: a capa decide o clique quando aparece: na prateleira de Shorts da página inicial (mesmo na aba dos
  vídeos longos), na página do canal e na busca, junto com a prévia que o YouTube roda. Mas isso vale para o
  primeiro Short que a pessoa abre; depois ela rola, o vídeo começa sozinho e ninguém vê a capa, e a rolagem é a
  maioria das visualizações. A capa também é um retrato do conteúdo (rosto, produto, carro, texto na tela), uma pista
  do assunto além do título.
Métrica e limiar: a régua da #27 no walk-forward de 2025, na base toda: o bloco da capa fica se melhorar a manchete
  ou o `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a outra.
Controle: o melhor GBM da Opção A
Tratamento: + bloco da thumbnail (SigLIP 2 com PCA ajustado no treino, sondas e cor/brilho do OpenCV), base toda
Resultado:
Decisão:

## H8: rubrica do VLM (#13)

Data: 2026-10-20
Hipótese: a rubrica ajuda muito no número.
Mecanismo: os campos da rubrica (o gancho dos 3 primeiros segundos, energia e ritmo, rosto, produto em close,
  antes/depois e texto na tela) são a informação que temos sobre o conteúdo do vídeo, e o conteúdo é o que mais pesa
  para a pessoa ficar no Short em vez de passar e para ela gostar dele e curtir. É o mecanismo da H1, mas a H1 só vê
  o conteúdo pelo título; a rubrica vê o vídeo.
Métrica e limiar: a régua da #27 no walk-forward de 2025, na amostra: o bloco fica se melhorar a manchete ou o
  `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a outra. Na amostra, a manchete tem só 201 pares
  (391 Shorts de 62 canais; na base toda, 198.159): o IC do ganho fica em ~±6 a ±9 pontos, e um ganho real de +3
  apareceria ~1 vez em 10. Ela é inconclusiva por tamanho, nem confirma nem barra; quem decide, na prática, é o
  `mae_log` (decisão de 20/10)
Controle: o melhor GBM da Opção A, na amostra de 1.400 Shorts
Tratamento: + os campos da rubrica com kappa > 0,6 no teste-reteste
Resultado:
Decisão:

## H9: sinais do vídeo e do áudio (#12)

Data: 2026-10-20
Hipótese: os sinais ajudam no número, como a rubrica. Se acrescentam algo além dela, não tenho aposta: a ablação
  da amostra decide (o completo contra o completo sem os sinais).
Mecanismo: o mesmo da H8: cortes, movimento, cor, fala e energia do áudio medem o conteúdo que faz a pessoa ficar e
  curtir. Os dois lados são plausíveis: a rubrica resume o vídeo numa nota de 0 a 10, e o código mede com precisão
  (acrescentaria); mas o Gemini já assistiu ao vídeo, e a nota de energia já reflete os cortes e a música
  (repetiria).
Métrica e limiar: a régua da #27 no walk-forward de 2025, na amostra: o bloco fica se melhorar a manchete ou o
  `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a outra. Na amostra, a manchete tem só 201 pares
  (391 Shorts de 62 canais; na base toda, 198.159): o IC do ganho fica em ~±6 a ±9 pontos, e um ganho real de +3
  apareceria ~1 vez em 10. Ela é inconclusiva por tamanho, nem confirma nem barra; quem decide, na prática, é o
  `mae_log` (decisão de 20/10)
Controle: o melhor GBM da Opção A, na amostra de 1.400 Shorts
Tratamento: + cortes, movimento e aparência dos frames (4.3) e fala e energia do áudio (4.5)
Resultado:
Decisão:

## H10: rosto e texto na capa, por detectores (#31)

Data: 2026-10-21 (escrita antes de rodar a extração e o experimento)
Hipótese: o rosto e o texto medidos na capa ajudam pouquíssimo, na ordem e no número; talvez nada.
Mecanismo: três motivos, que se somam. (1) A capa só age onde é vista: na prateleira de Shorts, na página do canal e
  na busca; no feed, o vídeo começa sozinho e ninguém vê a capa, e a rolagem é a maioria das visualizações (o
  mecanismo da H7). (2) O SigLIP já vê boa parte do mesmo: contra o VLM, a sonda do rosto tem AUC 0,787 e a do texto,
  0,703 (F26); o detector troca a semelhança pela quantidade (quanto da capa o rosto ou o texto ocupa), mas a
  informação é quase a mesma. (3) O alvo é relativo ao canal: se um canal usa sempre o mesmo modelo de capa, a coluna
  fica constante dentro dele e não explica por que um Short rende mais que o normal do canal; o detector só pode
  ajudar nos canais que variam a capa.
Métrica e limiar: a régua da #27 no walk-forward de 2025, na base toda: o bloco fica se melhorar a manchete ou o
  `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a outra. Pela #36, só vira argumento da explicação
  se também aparecer na remoção (o completo sem os detectores), acima do controle de ruído.
Controle: o completo da H7 (`configs/experimentos/h7_thumb.yaml`, run `f405ee37`)
Tratamento: + as 6 colunas dos detectores nas 30.981 capas: rosto (tem, quantos, área do maior) com o YuNet e texto
  (tem, quantas caixas, área coberta) com o detector PP-OCRv3 (`configs/experimentos/h10_detectores.yaml`)
Resultado:
Decisão:

## H11: perda de Huber no lugar da L2 (#32)

Data: 2026-10-21 (escrita antes de rodar; a H10, dos detectores da capa, vem depois)
Hipótese: trocar a L2 pela Huber (δ = 1 em log) diminui o erro no número (`mae_log`), mas não mexe tanto na ordem
  (manchete), porque os Shorts parecidos continuam recebendo a mesma avaliação entre si.
Mecanismo: quando há Shorts virais, a Huber os inclui na conta, mas com um limite. Na L2, o erro entra ao quadrado, e
  um viral puxa o número de todos os Shorts parecidos (na folha da Lumina, com 9 Shorts em 0 e 1 em +3,0: 0,30 com L2
  e 0,11 com Huber; o `mae_log` dos 10 cai de 0,54 para 0,39).
Métrica e limiar: a régua da #27 no walk-forward de 2025 e no "canal novo no futuro", na base toda: a Huber fica se
  melhorar o `mae_log`, com IC 95% pareado por canal sem o zero, sem piorar a manchete (o IC da manchete não fica
  inteiro do lado ruim). Senão, a L2 fica, agora escolhida pela medição.
Controle: o completo da H7 com L2 (`configs/experimentos/h7_thumb.yaml`, run `f405ee37`)
Tratamento: a mesma config com `objective: huber` e `alpha: 1.0` (`configs/experimentos/h11_huber.yaml`)
Resultado:
Decisão:
