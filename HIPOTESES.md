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
- **Onde se decide:** as escolhas de modelo (H2 a H8) nos folds de walk-forward de 2025, dentro do treino; o teste
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

Data: 2026-10-19
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
Controle: lacuna de 30 dias (princípio 2)
Tratamento: lacunas de 0 e 7 dias (o exercício da 3.5)
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

Data: 2026-10-19
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
Controle: o GBM sem `target_age_days`
Tratamento: (a) `target_age_days` como variável de controle, fixada em 30 na previsão; (b) treinar só com rótulos
  de idade parecida
Resultado:
Decisão:

## H6: teto por canal nas linhas de treino (#8)

Data: 2026-10-19
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
Controle: sem teto (a manchete, #8)
Tratamento: até 400 Shorts por canal nas linhas de treino (o histórico continua com todos os posts)
Resultado:
Decisão:

## H7: thumbnail (#12)

Data: 2026-10-19
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
Controle: o melhor GBM da Opção A
Tratamento: + bloco da thumbnail (SigLIP 2 com PCA ajustado no treino, sondas e cor/brilho do OpenCV), base toda
Resultado:
Decisão:

## H8: rubrica do VLM (#13)

Data: 2026-10-19
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
Controle: o melhor GBM da Opção A, na amostra de 1.400 Shorts
Tratamento: + os campos da rubrica com kappa > 0,6 no teste-reteste
Resultado:
Decisão:

## H9: sinais do vídeo e do áudio (#12)

Data: 2026-10-19
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
Controle: o melhor GBM da Opção A, na amostra de 1.400 Shorts
Tratamento: + cortes, movimento e aparência dos frames (4.3) e fala e energia do áudio (4.5)
Resultado:
Decisão:
