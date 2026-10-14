---
name: aula
description: Consulta uma aula do material da liga (trilha ML Preditivo e Agentes, RAG e Rerankers, Eval-Driven Development ou o hub do case Galaxies) antes de planejar ou implementar uma peça do Prévia. Use sempre antes de mexer numa peça que uma aula cobre, e quando o Vinicius pedir para estudar ou revisar uma aula.
argument-hint: "<aula> (ex.: 3.5, rag-4, edd-6, hub-roadmap)"
---

# /aula $ARGUMENTS

Traz para a sessão o que a aula pede ao Prévia, antes de planejar ou implementar. O material vence qualquer resumo,
inclusive os cartões deste repositório.

1. **Atualize o clone:** `git -C ~/autoestudos pull --ff-only`. Se ele não existir:
   `gh repo clone inteliacademyclub/autoestudos ~/autoestudos -- --depth 1`.
2. **Localize a aula** em `~/autoestudos/autoestudos/docs/`:
   - `X.Y` da trilha: `aulas/ml-preditivo-e-agentes/<X>-*/<Y>-*/index.mdx`. Por exemplo,
     `ls -d ~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/3-*/5-*`.
   - `rag-N`: `aulas/rag-e-rerankers/<N>-*/index.md`.
   - `edd-N`: `aulas/eval-driven-development/<N>-*/index.md`.
   - `hub-<página>`: `projetos/galaxies/<página>.mdx`. As páginas são `index`, `roadmap`, `monte-sua-base`,
     `stack-2026`, `leituras`, `templates`, `autoavaliacao`, `troubleshooting`, `laboratorio` e `glossario`.
   - `llm-N` (Conceitos básicos de LLM, por exemplo `llm-8` para Function calling):
     `aulas/conceitos-basicos-de-llm/<N>-*/index.md`.
3. **Leia a aula inteira** com Read, usando offset se vier truncada. Não substitua a leitura por grep.
4. **Leia o contexto do repositório:**
   - o cartão em `docs/material/` (`modulo-X.md`, `rag.md`, `edd.md` ou `hub.md`);
   - o que o `docs/ROTEIRO.md` diz da aula (peça M#/A#, decisões pendentes);
   - `docs/material/conflitos.md`, se a aula aparecer lá.
5. **Confira o repositório:** o que já existe para esta aula (código, testes, documentos) e o que falta.
6. **Devolva, curto e nesta ordem:**
   - o que a aula pede ao projeto ("No projeto Galaxies"), em 2 a 4 linhas;
   - os números, versões e limiares recomendados, que viram restrições do plano;
   - todos os "Erros comuns", como checklist `- [ ]`, para virar o Review Focus do plano;
   - o estado no repositório: o que está feito e o que falta;
   - a prática, segundo o ROTEIRO: Vinicius escreve (M#/A#), Vinicius decide, Claude ou leitura;
   - as decisões que a aula exige do Vinicius, cada uma com a recomendação e a seção da aula que a sustenta;
   - os pré-requisitos de outras trilhas (RAG, EDD, Conceitos de LLM), se houver.
7. **Não implemente nada** nesta skill. Se o cartão divergir da aula, a aula vence: corrija o cartão e avise.
