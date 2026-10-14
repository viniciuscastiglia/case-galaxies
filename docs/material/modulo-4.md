# Módulo 4: Do vídeo às features (8 aulas, 400 min)

Fonte: `~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/4-do-video-as-features/`.

O encadeamento do módulo:

- **4.1:** o pipeline, o cache e as camadas;
- **4.2:** como obter a mídia e a política de retenção;
- **4.3 a 4.6:** a extração de colunas (cortes, luz e movimento; SigLIP 2; áudio e fala; rosto, texto e objetos);
- **4.7:** o VLM como formulário;
- **4.8:** a fusão e a prova por ablação.

No Prévia, o módulo inteiro é encanamento do Claude. O Vinicius lê a fundo 4.1, 4.4 e 4.8 e as outras na velocidade
rápida ("No projeto Galaxies", "Erros comuns" e "Resumo"); decide o recorte e escreve a M9 (ablação).

**O menor recorte válido** (4.1 e hub). A camada leve é obrigatória: "thumbnail ou frames, transcrição e atributos
extraídos por VLM".

- Só texto e metadados não substituem a camada multimodal.
- A autoavaliação do hub chama o VLM de "recomendado", o índice do hub o põe no núcleo (ver
  [conflitos](conflitos.md)).
- "A ordem importa mais do que a camada": um mínimo cacheado e medido vale mais que um ambicioso incompleto.
- Um resultado negativo bem medido também vale ("não detectamos ganho com N vídeos").

**Ambiente do módulo.** torch 2.14.0, transformers 5.17.0, pillow 12.3.0, opencv-python-headless 5.0.0.93, scenedetect
0.7.1, faster-whisper 1.2.1, librosa 1.0.0, google-genai 2.25.0, `yt-dlp[default]==2026.8.19` (com Deno ≥ 2.3),
httpx 0.28.1 e tenacity 9.1.4. Na 4.6: mediapipe 1.0.1 e paddleocr 3.7.0 + paddlepaddle 3.3.1. Também o ffmpeg no
sistema.

**Na máquina do Vinicius (Mac M4, 16 GB):**

- o MediaPipe aborta no Apple Silicon, então rosto só no Colab ou em Linux;
- o yt-dlp só roda do IP de casa;
- importe `lightgbm` antes de `torch`, ou extraia num processo e treine em outro.

## 4.1 Anatomia de um Short e o pipeline multimodal

45 min · encanamento + produto · leitura profunda · prática: Claude

- **Conceitos.**
  - 9:16 equivale a 0,5625. `ffprobe` lê os metadados sem decodificar; corrija a rotação.
  - O gancho (cerca de 3 s) decide; o loop conta como view desde 31/03/2025; o som é conteúdo.
  - Não treinar rede de vídeo: encoders congelados → colunas → GBM, como no MVP (SMP 2025).
  - O canal domina; o visual soma de +0,02 a +0,04. Entre A e B do mesmo canal, o canal se cancela e o conteúdo é
    tudo o que sobra.
- **Constrói.** `metadados_arquivo` (12 colunas) e o decorador `cache_por_video(etapa, versao)`: um JSON por vídeo,
  escrita atômica e falhas em `_erros/`. Também `processar_lote` e `consolidar`.
- **No projeto Galaxies.** A tabela qualitativa de A × B. Tudo o que o modelo puder dizer de diferente sobre os dois
  virá destas colunas.
- **Erros comuns.**
  1. Querer treinar uma rede de vídeo.
  2. Processar sem cache.
  3. Extrair de um jeito no treino e de outro na previsão.
  4. Ignorar a rotação e a proporção.
  5. Começar pela camada ambiciosa.
- **Exercício.** O esqueleto do pipeline em 20 vídeos. O Claude faz.

## 4.2 Obtendo a mídia

45 min · encanamento · leitura rápida · prática: Claude

- **Conceitos.**
  - "Baixe o mínimo, na menor resolução que serve, uma vez só, e apague."
  - Árvore de decisão: canal, tempo e texto não precisam de nada; a capa sai da thumbnail da API (sem cota); a
    rubrica do VLM sai da URL pública no Gemini; pixels e áudio exigem o yt-dlp em 360p.
  - Na previsão, a marca entrega o MP4, então o pipeline aceita arquivo local desde o início.
  - Termos: scraping proibido (III.E.6) e retenção de 30 dias. A postura é educacional: mínimo necessário, apagar
    depois de extrair.
- **Constrói.**
  - `baixar_thumbnail` (maxres → default, tenacity) e `remover_barras`, que tira o pillarbox.
  - `baixar_video` com `-S "res:360"`, pausas de 5 a 10 s e Deno. "Not a bot" para o lote.
  - `extrair_e_apagar`.
- **No projeto Galaxies.** A e B passam pelo mesmo caminho: converter para lado menor 360 com ffmpeg. A "thumbnail"
  de A/B é o frame escolhido pela marca; documente.
- **Erros comuns.**
  1. Baixar em resolução máxima.
  2. Usar `height<=360` em vídeo vertical, que baixa 144×256.
  3. Insistir depois do "not a bot".
  4. Usar cookies da conta pessoal.
  5. Guardar os vídeos "para depois".
  6. Tratar a thumbnail atual como a da publicação.
- **Notas.** Concorrência de 1 a 2. O free tier do Gemini não serve para o MP4 inédito.

## 4.3 Frames, cortes e sinais de baixo nível

50 min · features · leitura rápida · prática: Claude

- **Conceitos.**
  - Amostragem de 3 frames no gancho mais 5 uniformes.
  - O OpenCV lê em BGR.
  - Cortes com `AdaptiveDetector`; o `ContentDetector(27)` perde cortes entre planos parecidos.
  - Medidas: brilho, contraste, saturação, colorfulness e optical flow de Farnebäck, que nunca atravessa um corte.
  - "Nenhum número sem gabarito."
- **Constrói.** `features_de_corte` e `movimento`, mais a etapa `sinais_baixo_nivel`, com cerca de 19 colunas e
  ~0,2 s por vídeo.
- **Erros comuns.**
  1. Esquecer o BGR.
  2. Usar resolução ou fps diferentes entre treino e previsão.
  3. Deixar o flow atravessar um corte.
  4. Confiar no limiar padrão.
  5. Usar a API antiga do PySceneDetect.
  6. Processar frames com faixas pretas.
- **Notas.** O `decord` está abandonado.

## 4.4 Embeddings visuais: CLIP e SigLIP 2

50 min · ML + AI Eng · leitura profunda · prática: Claude; o Vinicius valida as sondas, se der

- **Conceitos.**
  - CLIP usa softmax no lote; SigLIP usa sigmoide por par. O SigLIP 2 é multilíngue e Apache-2.0.
  - Sonda = cosseno entre o frame e uma frase. Use a relativa (menos a média das sondas); a bruta engana.
  - Vetores separados para vídeo, gancho e thumbnail; PCA de 16 a 32 por família, só no treino.
  - UMAP e t-SNE só servem para olhar.
- **Constrói.** `google/siglip2-base-patch16-224` (~1,5 GB, 768 dimensões, roda em CPU). No transformers 5, o vetor
  sai em `.pooler_output`. Também `embeddings_do_video`, que devolve `emb_video`, `emb_gancho` e `emb_thumb`. Custa de
  6 a 9 s por vídeo em CPU.
- **No projeto Galaxies.** A deve ter a sonda `logo` alta; B, `rosto` e `texto`. Os vizinhos no PCA alimentam a 6.4
  (comparáveis).
- **Erros comuns.**
  1. Tratar o retorno como tensor.
  2. Esquecer `padding="max_length"` e as minúsculas.
  3. Ajustar o PCA na base inteira.
  4. Usar UMAP como feature.
  5. Confiar numa sonda sem validá-la.
  6. Misturar thumbnail, gancho e vídeo num vetor só.
- **No Prévia.** Thumbnails da base toda com SigLIP 2 (Fase 4). É o único bloco que passou no bootstrap da simulação
  da 4.8. Liga com RAG 4 (embeddings).

## 4.5 Áudio e fala

50 min · features · leitura rápida · prática: Claude (na amostra)

- **Conceitos.**
  - Duas perguntas: o que é dito (faster-whisper com VAD e timestamps) e como soa (librosa: RMS, onsets, pulso).
  - O idioma só vale com probabilidade ≥ 0,7; o VAD evita a alucinação do Whisper.
  - O BPM sozinho engana.
  - Prefira `aud_inicio_fala_s` contínua.
- **Constrói.** `extrair_audio` (WAV 16 kHz mono) e o Whisper (`tiny` para testar, `small` no projeto). Cerca de 15
  colunas `aud_*`, mais `aud_musica_provavel`.
- **Erros comuns.**
  1. Aceitar idioma de vídeo sem fala.
  2. Usar o BPM sozinho.
  3. Rodar sem VAD.
  4. Confiar na transcrição do `tiny`.
  5. Não guardar a transcrição.
- **Notas.** A librosa 1.0 não abre `.mp4`: extraia o WAV antes. No automotivo, o ronco do motor engana o detector de
  música.

## 4.6 Detectores leves: rosto, texto e objetos

50 min · features · leitura rápida · prática: Claude (OCR na amostra; rosto só se houver Colab)

- **Conceitos.**
  - O detector responde "existe e onde"; o VLM responde "o que significa".
  - MediaPipe Tasks (`FaceDetector`), PaddleOCR com confiança ≥ 0,8 e OWLv2 com score contínuo.
  - Licenças: o YOLO é AGPL; prefira OWLv2 ou RT-DETR.
- **Constrói.** `rostos_no_frame`, `texto_no_frame`, `objetos_no_frame` e `agrega_detectores`, que gera as colunas
  `det_*`.
- **Erros comuns.**
  1. Copiar tutorial antigo do MediaPipe.
  2. Passar RGB onde se espera BGR.
  3. Multiplicar caixas em int16.
  4. Usar limiar global no OWLv2.
  5. Ignorar a licença.
  6. Rodar o detector caro em todos os frames.
- **Notas.** O `FaceDetector` aborta no Apple Silicon. A sonda "rosto" do SigLIP é a alternativa local.

## 4.7 VLM como extrator estruturado

55 min · AI Eng · leitura rápida · prática: o Vinicius decide a rubrica (5 a 8 campos), se o VLM entrar

- **Conceitos.**
  - O VLM extrai, não prevê: preenche um formulário com âncoras 0/5/10.
  - Valide com Pydantic: `resp.parsed` vira `None` em silêncio.
  - Backoff só para 429 e 5xx.
  - `temperature=0` não garante determinismo.
  - Confiabilidade medida em teste-reteste: kappa para os campos booleanos, Spearman para as notas.
- **Constrói.**
  - A classe `Rubrica`, com 18 campos.
  - `extrair_rubrica` (Gemini, `fps=1`, resolução LOW), com teste via `httpx.MockTransport`.
  - `etapa_vlm` com cache e `teste_reteste`.
  - Free tier: 8 h de vídeo por dia.
- **No projeto Galaxies.** A e B não têm URL: vão pela Files API, e criativo inédito nunca vai ao free tier.
- **Erros comuns.**
  1. Pedir a previsão ao VLM.
  2. Dar notas sem âncora.
  3. Confiar em `resp.parsed`.
  4. Repetir qualquer erro.
  5. Deixar a extração para o fim.
  6. Mudar o prompt no meio da base.
- **Notas.** Mantenha os campos com kappa acima de ~0,6. `potencial_viral` é opinião e só entra se a ablação ajudar.
- **No Prévia.** A rubrica curta, via Gemini por URL, na amostra, só se a Fase 4 estiver no ritmo; senão, vira
  limitação documentada. Liga com EDD 4 e Function calling.

## 4.8 Fusão e ablação multimodal

55 min · ML · leitura profunda · prática: **Vinicius escreve M9**

- **Conceitos.**
  - Early fusion em colunas.
  - Prefixos: `meta_`, `canal_`, `txt_`, `vis_`, `aud_`, `det_` e `vlm_`.
  - Mídia ausente fica NaN; a flag `midia_ok` é só diagnóstico.
  - Escada cumulativa mais "todos menos um".
  - Bootstrap pareado por canal.
- **Constrói.** O join `validate="one_to_one"`; um bool com NaN vira `object` no pandas 3, então converta para float.
  Também `pca_no_treino`, K-Means nas frases do VLM (ajustado no treino) e a escada abaixo.

  | Degrau | Spearman | Δ [IC 95%] |
  | :--- | :--- | :--- |
  | A (meta + canal) | 0,564 | — |
  | + texto | 0,543 | −0,020 [−0,050; +0,009] |
  | + visual | 0,587 | +0,046 [+0,008; +0,084] |
  | + áudio | 0,591 | +0,004 |
  | + detectores | 0,600 | +0,009 [−0,003; +0,021] |
  | + VLM | 0,597 | −0,004 |

  Só o visual passou no bootstrap: +0,024 sobre A, dentro da faixa de +0,02 a +0,04 do briefing.
- **No projeto Galaxies.** A ablação decide quais blocos a tool calcula para A e B. Uma explicação apoiada num bloco
  que não passou é frágil.
- **Erros comuns.**
  1. Ajustar PCA ou K-Means na base inteira.
  2. Comparar degraus com splits ou seeds diferentes.
  3. Ler a diferença sem intervalo.
  4. Perder colunas no join.
  5. Usar a flag de falha como feature.
  6. Concluir pela escada cumulativa.
- **Notas.** Registre cada degrau no MLflow. A tabela do README tem degrau, número de colunas, Spearman, Δ, IC e
  decisão.
- **No Prévia.** M9: a escada, o bootstrap e os 20 piores erros. Se o multimodal atrasar, rodar com meta, canal e
  texto; a metodologia vale sem vídeo.
