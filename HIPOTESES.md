# Hipóteses do Prévia

Escritas **antes** de rodar (aula 2.5: o "jardim dos caminhos que se bifurcam"). Cada hipótese diz o que se espera,
por quê, e como se decide. Depois de rodar, entram o resultado (com o run id do MLflow) e a decisão; o que estava
escrito antes não muda. Os resultados das H2 a H11 saem da abertura de 21/10 (`previa.abertura`), com os runs de
controle e de tratamento; os Δ são do tratamento sobre o controle, em pontos na manchete e em log no `mae_log`.

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
Resultado (21/10, teste aberto uma vez, run `6b1f7e77`): temporal +5,6 pontos [IC +3,6; +7,1] (55,3% × 49,7%); canal novo +3,5 [+1,9; +4,9]. Modelo final da #39 (sem histórico e texto; capa, idade como controle, Huber).
Decisão: H1 confirmada nos dois cenários (critério da #5 cumprido).

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward): `29c8914e` → `7b230752`: manchete −0,3 [−1,6; +0,6]; `mae_log` −0,011 [−0,032; +0,008].
Decisão: não fica (nenhuma das duas réguas com IC sem o zero, #27); o modelo final não tem o histórico (#39).

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward): lacuna de 7 dias, `7b230752` → `b6f87e18`: manchete −1,7 [−2,6; −0,6]; `mae_log` −0,018 [−0,032; −0,008]. A de 0 dias, só ilustração (`739112f6`): manchete +0,4 [−1,2; +1,8]; `mae_log` −0,062 [−0,082; −0,045], o ganho que o alvo imaturo dá de graça.
Decisão: não fica (o `mae_log` melhora, mas a manchete piora com o IC inteiro do lado ruim); fica a lacuna de 30 dias.

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward): (a) manuais, `192a4f36` → `caba0e64`: manchete +0,1 [−0,6; +0,9]; `mae_log` +0,003 [−0,005; +0,011]. (b) embeddings, `192a4f36` → `7b230752`: manchete +0,3 [−0,4; +1,0]; `mae_log` +0,004 [−0,006; +0,015] (sobre a, `caba0e64` → `7b230752`: +0,2 [−0,6; +0,9]; +0,001 [−0,010; +0,013]).
Decisão: não fica, nem (a) nem (b); o modelo final não tem o texto (#39).

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward): `7b230752` → `889dbae6`: manchete +0,3 [−0,4; +1,2]; `mae_log` −0,018 [−0,028; −0,007].
Decisão: fica (o `mae_log` melhora com IC sem o zero, sem piorar a manchete); a idade entra no treino e fica fixada em 30 na previsão.

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento): `7b230752` → `09275208`. Canal novo: manchete −0,1 [−0,9; +0,8]; `mae_log` −0,005 [−0,017; +0,005]. Walk-forward (só a guarda de não piorar): manchete −0,1 [−0,9; +0,8]; `mae_log` +0,006 [−0,007; +0,021].
Decisão: não fica (no canal novo, nenhuma régua com IC sem o zero); o treino segue sem teto.

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward): `7b230752` → `f405ee37`: manchete +2,3 [+1,2; +3,3]; `mae_log` −0,012 [−0,023; −0,003].
Decisão: fica nas duas réguas e é sólida nas três leituras da ablação (escada, remoção e acima do ruído, #36): é o único bloco de conteúdo que a explicação usa como argumento.

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward, na amostra; régua pelo `mae_log`, com a manchete como guarda): `e97356c6` → `59961a2f`: manchete −5,5 [−14,9; +3,4]; `mae_log` −0,005 [−0,021; +0,010].
Decisão: não fica; inconclusivo por tamanho (cerca de 200 pares na amostra).

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward, na amostra; régua pelo `mae_log`): `e97356c6` → `601b2217`: manchete −3,9 [−12,7; +4,8]; `mae_log` +0,000 [−0,014; +0,015].
Decisão: não fica; inconclusivo por tamanho.

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento, walk-forward): `f405ee37` → `dc4f5984`: manchete −0,5 [−1,1; +0,3]; `mae_log` −0,002 [−0,013; +0,006].
Decisão: não fica.

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
Resultado (abertura de 21/10, `uv run python -m previa.abertura`; controle → tratamento): `f405ee37` → `bfba1020`. Walk-forward: manchete +2,0 [+1,2; +3,0]; `mae_log` −0,043 [−0,062; −0,026]. Canal novo: manchete +0,4 [−0,4; +1,3]; `mae_log` −0,077 [−0,112; −0,047].
Decisão: fica nos dois cenários; o modelo final usa a Huber (#39, #40).

## Robustez (22/10, depois do teste; só o desenvolvimento)

Não é hipótese nova nem troca o modelo: confere se as duas escolhas que decidiram o modelo final se sustentam em
outras seeds (aula 2.6, erro comum 5) e se o boosting vence um random forest com os defaults (aula 3.2). Tudo no
walk-forward de 2025 e no "canal novo no futuro", dentro do treino; as configs estão em
`configs/experimentos/robustez/`, e cada comparação sai de `uv run python -m previa.experimento comparar A B`.

**Seeds.** O modelo final contra a mesma config sem a capa, e contra a mesma config com a L2, nas seeds 0, 1 e 2 (a
seed 0 do final é o run `e684ac8c`). Δ do modelo final sobre a variante:

| Escolha | Seed | Walk-forward: manchete | Walk-forward: `mae_log` | Canal novo: manchete | Canal novo: `mae_log` |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Capa | 0 (`61fd784a` → `e684ac8c`) | +3,2 [+1,6; +4,9] | −0,024 [−0,045; −0,003] | +4,0 [+2,4; +5,6] | −0,014 [−0,030; +0,006] |
| Capa | 1 (`91fd1af7` → `7dc467e4`) | +3,2 [+1,6; +4,7] | −0,023 [−0,039; −0,007] | +3,5 [+1,9; +4,9] | −0,011 [−0,027; +0,009] |
| Capa | 2 (`bb37bb0e` → `2d7e2650`) | +3,4 [+1,7; +5,0] | −0,025 [−0,044; −0,006] | +3,2 [+1,3; +5,4] | −0,002 [−0,014; +0,014] |
| Huber | 0 (`85b36d3d` → `e684ac8c`) | +1,6 [+0,7; +2,2] | −0,026 [−0,043; −0,011] | +0,8 [−0,3; +1,7] | −0,056 [−0,083; −0,034] |
| Huber | 1 (`2aa1f708` → `7dc467e4`) | +1,4 [+0,8; +1,9] | −0,030 [−0,050; −0,015] | +0,8 [−0,2; +1,5] | −0,066 [−0,104; −0,035] |
| Huber | 2 (`e0374960` → `2d7e2650`) | +1,3 [+0,5; +2,0] | −0,030 [−0,048; −0,014] | +0,8 [0,0; +1,9] | −0,060 [−0,087; −0,035] |

As duas escolhas passam a régua da #27 nas três seeds. A capa melhora a ordem nos dois cenários e o número no
walk-forward; a Huber melhora o número nos dois cenários e a ordem no walk-forward. O ganho da capa aqui (+3,2 a +3,4)
é maior que o da H7 (+2,3) porque o controle é outro: o modelo final sem a capa, e não a Opção A.

**Random forest (aula 3.2).** As mesmas linhas e features do modelo final num random forest com os defaults do
scikit-learn (run `938cb82e`): manchete de 54,2% no walk-forward e 53,5% no canal novo, contra 49,7% da mediana
madura. O modelo final sobre ele: walk-forward, manchete +1,2 [+0,2; +2,2] e `mae_log` −0,064 [−0,100; −0,039];
canal novo, manchete −0,2 [−1,2; +0,7] e `mae_log` −0,065 [−0,119; −0,028]. O boosting erra bem menos o número nos dois
cenários e ordena melhor no futuro; nos canais nunca vistos, a ordem empata. A maior parte do ganho de ordem vem das
features (o random forest já fica bem acima da mediana), não do algoritmo.
