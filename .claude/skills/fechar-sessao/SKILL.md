---
name: fechar-sessao
description: Fecha a sessão de trabalho do Prévia. Atualiza o ROTEIRO, o caderno de decisões, o diário e a memória, roda os testes e propõe o commit. Use no fim de cada sessão ou bloco, ou quando o Vinicius disser que vai parar.
---

# /fechar-sessao

1. **Verifique** (superpowers:verification-before-completion): rode `uv run pytest -q` e `uv run ruff check .` e
   reporte a saída real. Nada de "deve passar".
2. **Atualize o `docs/ROTEIRO.md`:**
   - marque as caixas feitas (leituras, peças, tarefas do Claude);
   - reescreva o "Agora" em 3 linhas (fase, próximo passo, bloqueios);
   - atualize o status das decisões pendentes;
   - compare o feito com a fase. Se ela estourou em mais de meio dia, proponha o próximo corte de "Se atrasar"; a
     decisão é do Vinicius.
3. **Atualize o `docs/DECISOES.md`:** uma linha por decisão tomada na sessão (data, decisão, quem, aula, por quê,
   onde está), inclusive as do Claude.
4. **Atualize o `docs/DIARIO.md`:** uma entrada no topo, de 3 a 5 linhas, com o que foi feito, os números
   importantes (com run id) e o que ficou para a próxima sessão.
5. **Atualize o `CLAUDE.md`** só se algo nele envelheceu: o "Agora" (por exemplo, quando o modelo ou o agente passam
   a existir), os "Comandos" (cada comando novo de treino, avaliação, MCP ou demo) e as armadilhas novas. Mantenha-o
   curto; o detalhe vai para o ROTEIRO.
6. **Memória:** só o que vale entre sessões e não está nos documentos, como preferências e feedback do Vinicius. Não
   duplique o ROTEIRO.
7. **Commit:** mostre `git status --short` e proponha a mensagem, em Conventional Commits, em inglês e citando a aula.
   **Não commite** sem o Vinicius pedir.
8. **Termine** com o próximo passo da próxima sessão, numa linha.
