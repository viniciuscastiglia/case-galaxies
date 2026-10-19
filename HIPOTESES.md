# Hipóteses do Prévia

Escritas **antes** de rodar (aula 2.5: o "jardim dos caminhos que se bifurcam"). Cada hipótese diz o que se espera,
por quê, e como se decide. Depois de rodar, entram o resultado (com o run id do MLflow) e a decisão; o que estava
escrito antes não muda.

Regras de todas:

- **Métrica:** a manchete da #6, o par-a-par na janela de 14 dias com cada canal pesando pelo nº de Shorts dele,
  com IC 95% do bootstrap pareado por canal (aula 2.4).
- **Onde se decide:** as escolhas de modelo (H2 a H8) nos folds de walk-forward de 2025, dentro do treino; o teste
  (abr a set/2026) abre uma vez, no fim, só para a H1 (aula 2.5, erro comum 4).
- **Controle e tratamento** usam a mesma base, os mesmos cortes, a mesma seed e o mesmo modelo, mudando uma coisa.

## H1: o Prévia ordena A e B melhor que o histórico do canal

Data: 2026-10-19 (decisão #5)
Hipótese: o GBM (Opção A: histórico do canal, horário, duração e texto) acerta mais pares de Shorts do mesmo canal,
  publicados a até 14 dias um do outro, que a mediana madura do canal.
Mecanismo: (Vinicius)
Métrica e limiar: manchete da #6 no teste temporal; aceito se o ganho for >= +3 pontos e o IC 95% do bootstrap
  pareado por canal não incluir zero, e se no "canal novo no futuro" o IC do ganho também não incluir zero.
  Os ~60% da Galaxies são referência, não meta.
Controle: a mediana madura (49,7% na manchete em 2025, IC 49,2% a 50,1%)
Tratamento: o GBM da M4
Resultado:
Decisão:

## H2: histórico do canal no GBM (M3)

Data: 2026-10-19
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
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
Hipótese: (Vinicius)
Mecanismo: (Vinicius)
Métrica e limiar: (Vinicius)
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
