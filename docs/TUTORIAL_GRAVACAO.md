# Tutorial de gravação do vídeo (passo a passo)

O que você vai fazer: gravar a tela do Mac com a sua voz, em **7 pedaços curtos** (um por bloco do roteiro), juntar
os 7 num vídeo só, subir no YouTube como "Não listado" e mandar o link para a liga.

- **As falas** estão em `docs/ROTEIRO_VIDEO.md`. Este arquivo diz **como** gravar.
- **Tempo total:** ~60 min (15 de preparo, 30 de gravação com folga para regravar, 15 para juntar e subir).
- **Regra de ouro:** errou uma frase? Pare a gravação **daquele bloco** e grave o bloco de novo. Nunca recomece o
  vídeo inteiro.

---

## Parte 0 · Antes de tudo (2 min)

1. Confira que o push foi feito. No Claude Code, digite:
   ```
   ! git -C /Users/viniciustavarescastiglia/case-galaxies log --oneline -1 origin/main
   ```
   A linha que aparece tem que ser o último commit da noite. Se não for, rode:
   ```
   ! git -C /Users/viniciustavarescastiglia/case-galaxies branch -f main feat/lote-noite-2 && git -C /Users/viniciustavarescastiglia/case-galaxies push origin main feat/lote-noite-2
   ```
2. Abra no navegador: https://github.com/viniciuscastiglia/case-galaxies e confira que a página abre **sem login**
   (abra numa janela anônima: Cmd+Shift+N no Chrome). Se abrir, o repo está público.

## Parte 1 · Preparar o Mac (3 min)

1. **Silencie tudo:** clique no ícone da Central de Controle (os dois "botõezinhos" no canto superior direito) →
   **Foco** → **Não Perturbe**. Assim nenhuma notificação aparece no vídeo.
2. **Feche** WhatsApp, Slack, e-mail e tudo que possa apitar ou mostrar coisa pessoal.
3. **Carregador na tomada** (gravar tela gasta bateria).
4. Crie uma pasta para os vídeos: no Finder, na Mesa (Desktop), botão direito → **Nova Pasta** → nome `video-previa`.

## Parte 2 · Preparar as janelas (7 min)

Você vai usar **duas janelas**: o **navegador** (para o README no GitHub) e o **Terminal** (para a demo).

### 2.1 O navegador

1. Abra https://github.com/viniciuscastiglia/case-galaxies (o README aparece embaixo da lista de arquivos).
2. Aumente o zoom para ler bem no vídeo: **Cmd +** duas vezes (fica em ~125%).
3. Abra uma **segunda aba** com o data card:
   https://github.com/viniciuscastiglia/case-galaxies/blob/main/docs/DATA_CARD.md (também com Cmd + duas vezes).
4. Deixe o navegador em **tela cheia da janela** (clique duas vezes na barra de cima da janela para maximizar).

### 2.2 O Terminal

1. Abra o **Terminal** (Cmd+Espaço, digite `Terminal`, Enter).
2. Aumente a fonte: **Cmd +** umas 4 vezes (o texto tem que ficar grande; quem assiste no celular precisa ler).
3. Maximize a janela (clique duas vezes na barra de cima).
4. Entre na pasta do projeto:
   ```
   cd /Users/viniciustavarescastiglia/case-galaxies
   ```
5. **Aqueça a demo** (baixa as capas e carrega o modelo; leva ~30 s). Rode:
   ```
   uv run python -m previa.demo --mcp
   ```
   Espere terminar. Tem que aparecer no fim: `igual ao da função: True`. **Se der erro, pare e me mande o erro.**
6. Limpe a tela: digite `clear` e Enter.

### 2.3 O roteiro na mão

Abra o `docs/ROTEIRO_VIDEO.md` **no celular ou num papel impresso**, não na tela que está sendo gravada. Se só tiver
o computador: abra no VS Code numa janela pequena **fora da área gravada** (veja 3.3).

## Parte 3 · Testar o microfone (3 min)

1. Aperte **Cmd+Shift+5**. Aparece uma barra na parte de baixo da tela.
2. Na barra, clique em **Opções**:
   - em **Microfone**, marque **Microfone do MacBook** (ou o do seu fone, se ele tiver microfone);
   - em **Salvar em**, escolha **Outro local…** → a pasta `video-previa` da Mesa;
   - marque **Mostrar Cliques do Mouse** (ajuda quem assiste a seguir você).
3. Na barra, escolha **Gravar Tela Inteira** (o ícone de tela com um círculo pequeno) e clique em **Gravar**.
4. Fale por 10 segundos: "Teste, um, dois, três, este é o Prévia".
5. Pare: clique no **botão de parar** (quadrado dentro de um círculo) na **barra de menus lá em cima**, à direita.
   Atalho: **Cmd+Ctrl+Esc**.
6. Abra o arquivo que apareceu na pasta `video-previa` e ouça. A voz está clara? Sem chiado alto? Ótimo. Apague esse
   arquivo de teste.

### 3.3 Se o roteiro estiver no mesmo computador

Em vez de "Gravar Tela Inteira", use **Gravar Parte Selecionada** (o ícone de retângulo pontilhado) e desenhe o
retângulo cobrindo só o navegador/Terminal. O roteiro fica do lado, fora do retângulo. A seleção fica salva para os
próximos blocos.

---

## Parte 4 · Gravar os 7 blocos (~30 min com folga)

Para **cada bloco**, o ciclo é sempre o mesmo:

1. **Arrume a tela** como o bloco pede (abaixo), **antes** de gravar.
2. **Cmd+Shift+5** → **Gravar**.
3. Respire, espere 1 segundo, e **leia a fala** do bloco no `ROTEIRO_VIDEO.md`, com calma.
4. Espere 1 segundo depois da última frase e **pare** (Cmd+Ctrl+Esc).
5. Na pasta `video-previa`, **renomeie o arquivo** (clique no nome, Enter, digite): `1.mov`, `2.mov`… até `7.mov`.
   O número é o do bloco: é isso que garante a ordem na hora de juntar.
6. Errou? Apague o arquivo e grave o bloco de novo. Não tem problema regravar 3 ou 4 vezes.

### Bloco 1 · O problema (~40 s) → arquivo `1.mov`

- **Tela:** o navegador, na aba do **README no GitHub**, no topo (o título "Prévia" e a tabela "Destaques").
- **Durante a fala:** pode rolar devagar até a seção "1. Problema e recorte" com o trackpad.

### Bloco 2 · A base (~40 s) → `2.mov`

- **Tela:** a aba do **DATA_CARD**. Comece no topo.
- **Durante a fala:** role devagar até a seção "O que esta base NÃO representa" (use **Cmd+F** e digite `NÃO
  representa` antes de gravar, para saber onde fica; feche a busca com Esc antes de gravar).

### Bloco 3 · Baseline contra modelo (~50 s) → `3.mov`

- **Tela:** a aba do **README**, na seção **7. Resultados**, na tabela **"Teste final"** (a que tem 49,7% e 55,3%).
  Para achar: **Cmd+F** → digite `Teste final` → Esc. Deixe a tabela no meio da tela.
- **Durante a fala:** quando disser "49,7%" e "55,3%", passe o mouse em cima dos números.

### Bloco 4 · A camada multimodal (~45 s) → `4.mov`

- **Tela:** README, tabela **"Ablação por blocos"** (Cmd+F → `Ablação por blocos` → Esc).
- **Durante a fala:** passe o mouse na linha **"+ capa (H7)"** quando falar dos +2,3 pontos.

### Bloco 5 · Ao vivo: o de-para (~90 s) → `5.mov`

- **Tela:** o **Terminal**, limpo (`clear`), com o comando **já digitado, mas sem apertar Enter**:
  ```
  uv run python -m previa.demo --mcp
  ```
- **Comece a gravar, aperte Enter**, e enquanto o comando roda (~30 s), leia o **primeiro parágrafo** da fala do
  bloco 5 ("Agora ao vivo. Estes são seis canais…"). Quando a tabela aparecer, siga a fala apontando com o mouse: a
  1ª tabela (previsto, intervalo, real), depois as linhas "A ou B", depois a explicação.
- **Pare a gravação no fim do bloco 5, mas não feche nem limpe o Terminal**: o bloco 6 usa o final dessa mesma
  saída.

### Bloco 6 · Pelo MCP (~45 s) → `6.mov`

- **Tela:** o **mesmo Terminal**, rolado até o fim da saída, onde está "== O mesmo A ou B pelo servidor MCP", "tools
  do servidor" e "igual ao da função: True".
- **Durante a fala:** aponte "prever_criativo, comparar_criativos" e depois "igual ao da função: True".
- **Não precisa rodar nada de novo.**

### Bloco 7 · Limitações e próximos passos (~60 s) → `7.mov`

- **Tela:** o navegador, README, seção **8. Limitações** (Cmd+F → `8. Limitações` → Esc).
- **Durante a fala:** role devagar. No "Obrigado!", espere 2 segundos e pare.

---

## Parte 5 · Juntar os 7 pedaços (5 min)

1. Na pasta `video-previa`, abra o **`1.mov`** (clique duas vezes: abre no **QuickTime Player**).
2. No menu de cima: **Editar → Adicionar Clipe ao Final…** → escolha **`2.mov`** → **Escolher Mídia**.
3. Repita **Editar → Adicionar Clipe ao Final…** para **`3.mov`**, **`4.mov`**, **`5.mov`**, **`6.mov`** e **`7.mov`**,
   nessa ordem. Embaixo aparecem os 7 pedaços em fila.
4. Clique em **OK** (no canto, depois do último pedaço).
5. **Arquivo → Exportar Como → 1080p…** → nome `previa_video` → salve na pasta `video-previa` → **Salvar**. Espere a
   barra de exportação terminar (1 a 3 min).
6. **Confira a duração:** abra o `previa_video.mov` e veja o tempo total na barra de baixo. Tem que estar **entre 5:00
   e 7:00**.
   - Passou de 7:00? Regrave o bloco mais longo falando um pouco mais rápido, ou corte uma frase da fala.
   - Ficou abaixo de 5:00? Fale mais devagar no bloco 5 (o ao vivo) e no 7.
7. **Assista ao vídeo inteiro uma vez** com som, para ver se não ficou nenhum pedaço faltando ou repetido.

## Parte 6 · Subir no YouTube (10 min)

1. Abra https://studio.youtube.com e entre com a sua conta Google. Se ele pedir para **criar um canal**, aceite (é
   rápido: só confirmar o nome).
2. Clique em **Criar** (botão com uma câmera, no canto superior direito) → **Enviar vídeos**.
3. **Selecionar arquivos** → escolha `previa_video.mov` na pasta `video-previa`.
4. Enquanto envia, preencha:
   - **Título:** `Prévia: previsão de curtidas de YouTube Shorts antes da publicação (case Galaxies)`
   - **Descrição:** cole isto:
     ```
     Demonstração do Prévia, projeto do case Inteli Academy x Galaxies.
     Código, dados de cada decisão e resultados: https://github.com/viniciuscastiglia/case-galaxies
     ```
   - **Público:** marque **"Não, não é conteúdo para crianças"**.
5. Clique em **Avançar** até chegar em **Visibilidade** (são 3 telas: Detalhes, Elementos do vídeo, Verificações).
6. Em **Visibilidade**, marque **Não listado** (só quem tem o link vê). Se a liga pediu "público", marque **Público**.
7. **Antes de clicar em Salvar, copie o link** do vídeo (aparece à direita, "Link do vídeo", com um botão de copiar).
8. Clique em **Salvar**. O YouTube processa o vídeo por alguns minutos; o link já vale.
9. **Teste o link** numa janela anônima (Cmd+Shift+N): tem que abrir o vídeo.

## Parte 7 · Mandar para a liga (2 min, até 23h59)

Mande, pelo canal que a liga pediu, os dois links:

```
Repositório: https://github.com/viniciuscastiglia/case-galaxies
Vídeo: <o link do YouTube que você copiou>
```

---

## Se algo der errado

| Problema | O que fazer |
| :--- | :--- |
| A demo dá erro no Terminal | Não insista: me mande a mensagem de erro. Plano B: grave o bloco 5 mostrando o arquivo `data/demo/respostas.json` aberto no VS Code e diga "estas são as respostas salvas da demo". |
| O Cmd+Shift+5 não grava som | Opções → Microfone: escolha o microfone de novo. No 1º uso, o Mac pode pedir permissão: Ajustes do Sistema → Privacidade e Segurança → Gravação de Tela/Microfone → ligue para o "Captura de Tela"/QuickTime. |
| O "Adicionar Clipe ao Final" está cinza | Clique uma vez dentro do vídeo aberto no QuickTime e tente de novo; ou feche e abra o `1.mov` de novo. |
| O YouTube recusa o arquivo ou demora muito | Exporte de novo em **720p** (Arquivo → Exportar Como → 720p), que fica menor. |
| Passou das 23h40 e o vídeo não está pronto | Mande à liga o link do repositório já, e o do vídeo assim que subir, avisando o horário. |

## Horário sugerido

| Hora | O quê |
| :--- | :--- |
| agora | Partes 0, 1, 2 e 3 (15 min) |
| +15 min | Parte 4, gravar os 7 blocos (30 min) |
| +45 min | Parte 5, juntar e conferir (5 min) |
| +50 min | Parte 6, YouTube (10 min) |
| +60 min | Parte 7, mandar os links |
