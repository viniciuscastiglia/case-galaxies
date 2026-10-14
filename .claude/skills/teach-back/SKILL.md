---
name: teach-back
description: Conduz o teach-back do Vinicius sobre uma peça ou aula do Prévia. Ele explica em 3 a 5 linhas o que a peça faz e por quê; o Claude compara com a aula e responde no formato combinado. Use quando terminar uma peça "Vinicius escreve" do ROTEIRO, antes de construir algo em cima dela, ou quando ele pedir.
argument-hint: "<aula ou peça> (ex.: 3.5, M3, A2)"
---

# /teach-back $ARGUMENTS

1. **Identifique a aula e a peça** na seção "Peças que o Vinicius escreve" do `docs/ROTEIRO.md`. Leia o cartão em
   `docs/material/` e, se precisar, a aula no clone (`~/autoestudos`).
2. **Peça a explicação** numa mensagem curta, em texto: "Em 3 a 5 linhas: o que essa peça faz, por que ela é assim
   (qual erro ela evita) e como você explicaria para a banca". Se ajudar, ancore no caso Lumina A vs. B.
3. **Espere a resposta.** Não antecipe a explicação.
4. **Responda no formato combinado em 12/10:**
   - Na primeira linha, o que está certo e o que está errado. Não chame de "errado" o que é só refinamento.
   - Corrija **uma coisa só**, com um exemplo concreto: a Lumina, números da base, um caso de 2 canais. O resto fica
     para depois.
   - Se fizer sentido, termine com uma pergunta de raciocínio, não de memória.
5. **Registre em `estudos/TEACHBACKS.md`:** data, aula ou peça, as linhas dele exatamente como ele escreveu, e a
   correção em uma linha. É o banco de respostas para a demo e para a banca.
