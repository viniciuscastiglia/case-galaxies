# F2 (camada 1 nas capas) e rascunhos F3, F29, F42 — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Emenda de 21/10 (aprovada pelo Vinicius; `docs/DECISOES.md`, "Texto na capa sem o PaddleOCR"):** o texto na capa é o detector PP-OCRv3 do `opencv_zoo` pelo `cv2.dnn.TextDetectionModel_DB` (só detecta, não lê), no lugar do PaddleOCR (que não instala na Stack 2026). São 6 colunas `det_*_thumb`, com `det_n_caixas_texto_thumb` no lugar de `det_n_linhas_thumb`, e sem `det_texto_tela_thumb`. Onde o texto abaixo diz PaddleOCR, vale esta emenda.

**Goal:** rosto (YuNet) e texto na tela (PaddleOCR) nas capas, como colunas `det_*_thumb` prontas para a H10 e
para a tool (F17); e os rascunhos do model card (F3), da comparação com a literatura (F29) e das limitações (F42).

**Architecture:** um módulo novo, `previa.detectores`, sem torch e fora do processo do LightGBM, no molde do
`previa.visual`: uma função por capa (`features_da_capa`, a mesma no lote e na tool, princípio 4), um lote com cache
em parquet que retoma de onde parou, e a concordância com a rubrica do VLM. O runner lê o bloco por `ler_extra`.
Os documentos são rascunhos do Claude com `PREENCHER:` onde o número depende de run lacrado ou do teste.

> Emenda: o OCR acima é o detector PP-OCRv3 do opencv_zoo via OpenCV (sem PaddleOCR; ver a nota do topo).

**Tech Stack:** OpenCV 5.0.0.93 (`cv2.FaceDetectorYN`, YuNet 2023mar, MIT), PaddleOCR 3.7.0 + paddlepaddle 3.3.1
(Stack 2026, Apache-2.0), pandas 3, scikit-learn (`cohen_kappa_score`), pytest.

**Spec:** `docs/AUDITORIA.md` (F2, F3, F29, F42 e a seção 3), `docs/DECISOES.md` (#31, 21/10), aula 4.6
(`~/autoestudos/autoestudos/docs/aulas/ml-preditivo-e-agentes/4-do-video-as-features/6-detectores-leves/index.mdx`).

## Global Constraints

> Emenda: as restrições do PaddleOCR abaixo (BGR e caixas int16 valem; `lang="pt"`, `rec_scores` e `det_texto_tela_thumb` não) foram trocadas pelo detector PP-OCRv3 do OpenCV, 6 colunas, `det_n_caixas_texto_thumb` (nota do topo).

- **Nada roda na base antes da H10 do Vinicius.** Os testes usam fakes; só a medição de tempo em 50 capas (Task 3,
  passo de custo) toca dados reais, e ela não lê curtidas nem métricas.
- **H2 a H9, H11 e as ablações estão lacradas:** nenhum subagente abre métricas de runs do MLflow nem o período de
  teste (`carregar_base(final=True)` proibido). Os documentos usam `PREENCHER:` para esses números.
- Sem torch e sem LightGBM no `previa.detectores` (OpenMP no macOS; CLAUDE.md, Armadilhas).
- A imagem de entrada é a mesma que o SigLIP vê: `recortar_9_16` de `previa.thumbs` sobre a capa em RGB.
- Os dois detectores esperam **BGR**: converta com `rgb[:, :, ::-1]` e `np.ascontiguousarray` (4.6, erro 2).
- Caixas do PaddleOCR vêm em `int16`: converta para `float` antes de calcular área (4.6, erro 3).
- OCR: `PaddleOCR(lang="pt", use_doc_orientation_classify=False, use_doc_unwarping=False,
  use_textline_orientation=False)`; leitura vale com `rec_scores >= 0.8` **e** ao menos 2 caracteres alfanuméricos
  (4.6, código de `texto_no_frame`).
- YuNet: `face_detection_yunet_2023mar.onnx` do `opencv_zoo`, `score_threshold` 0.9 (padrão da demo do zoo; o 0,5 da
  aula é do BlazeFace), `nms_threshold` 0.3, `top_k` 5000; `setInputSize((w, h))` a cada imagem. O arquivo fica em
  `DADOS / "modelos"` e é conferido por sha256 da config.
- Parâmetros em `configs/multimodal.yaml`, seção `detectores:` (versão, URL, sha256, limiares, lote); nada no código.
- Colunas de saída (todas por `video_id`): `det_rosto_thumb` (bool), `det_n_rostos_thumb` (int),
  `det_area_rosto_thumb` (float, fração do quadro), `det_texto_thumb` (bool), `det_n_linhas_thumb` (int),
  `det_area_texto_thumb` (float), `det_texto_tela_thumb` (str, fora do modelo).
- Caminhos por `previa.caminhos` (`DADOS`, `CONFIGS`), nunca relativos (F18).
- Código, nomes e comentários em português, no estilo do `previa.visual`; docstring de módulo citando a aula.
- **Subagentes não fazem commit** (vários trabalham no mesmo checkout ao mesmo tempo; o índice do git é um só). O
  controlador revisa e faz o commit de cada tarefa.
- Verificação: `uv run pytest` e `uv run ruff check .` verdes, mais `uv run ruff format src tests`.

## Review Focus

- Capa ausente ou corrompida (`Image.open` falha): o lote registra o erro e segue; o vídeo fica sem linha, e o resumo
  conta `erros`. Teste na Task 3.
- Capa toda preta ou de um só tom (as barras do letterbox): zero rostos, zero texto, sem exceção. Teste na Task 1.
- `faces` do YuNet vem `None` quando não há rosto (não um array vazio). Teste na Task 1.
- O PaddleOCR devolve listas vazias (`rec_texts == []`) ou um "C" gigante de baixa qualidade: área 0 e texto vazio.
  Teste na Task 2.
- Lote interrompido no meio: a segunda chamada não refaz o que está salvo e nunca deixa um parquet pela metade
  (`os.replace`). Teste na Task 3.

---

### Task 1: rosto na capa com YuNet

**Files:**
- Create: `src/previa/detectores.py`
- Modify: `configs/multimodal.yaml` (seção `detectores:`)
- Test: `tests/test_detectores.py`

**Interfaces:**
- Produces:
  - `rostos_na_capa(rgb: np.ndarray, detector) -> dict` com `{"n_rostos": int, "area_maior_rosto": float}`; o
    `detector` tem `setInputSize((w, h))` e `detect(bgr) -> (retval, faces | None)`, como o `cv2.FaceDetectorYN`.
  - `criar_yunet(cfg: dict) -> cv2.FaceDetectorYN` (baixa o `.onnx` se faltar, confere o sha256, levanta
    `ValueError` se não bater).
  - `ler_config_detectores() -> dict` (a seção `detectores` de `configs/multimodal.yaml`).

- [ ] **Step 1: leia a aula 4.6 inteira** (caminho no Spec) e o `previa.visual` (molde do módulo).
- [ ] **Step 2: escreva os testes que falham**, com um detector falso que guarda o que recebeu:
  - `test_rosto_area_e_contagem`: capa 100×200; o falso devolve 2 rostos de 10×10 e 20×30 → `n_rostos == 2`,
    `area_maior_rosto == pytest.approx(600 / 20000)`.
  - `test_rosto_recebe_bgr_e_tamanho_certo`: com uma imagem RGB de pixel `(255, 0, 0)`, o falso recebe pixel
    `(0, 0, 255)` e `setInputSize((100, 200))` (largura, altura).
  - `test_sem_rosto_faces_none`: o falso devolve `(1, None)` → `{"n_rostos": 0, "area_maior_rosto": 0.0}`.
  - `test_capa_preta_sem_rosto`: o mesmo resultado numa imagem `np.zeros((200, 100, 3), np.uint8)`.
  - `test_sha256_errado_levanta`: arquivo local com conteúdo qualquer e sha256 da config diferente → `ValueError`.
- [ ] **Step 3: rode e veja falhar** — `uv run pytest tests/test_detectores.py -v` → FAIL (módulo não existe).
- [ ] **Step 4: implemente** as três funções. A área usa as colunas 2 e 3 (largura, altura) de cada linha de
  `faces` em `float`, dividida por `w * h`. Baixe o modelo com `httpx` só quando o arquivo não existir.
- [ ] **Step 5: rode e veja passar**; depois `uv run ruff check . && uv run ruff format src tests`.
- [ ] **Step 6: confira com o modelo real**, sem a base: baixe o `.onnx`, grave o sha256 na config e rode
  `rostos_na_capa` numa imagem de rosto de domínio público (`skimage.data.astronaut()` se o scikit-image existir; se
  não, qualquer foto sem licença restritiva em `/tmp` do scratchpad). Anote a saída no relatório da tarefa.

### Task 2: texto na capa com PaddleOCR

> **Emenda de 21/10 (aprovada pelo Vinicius; `docs/DECISOES.md`):** o PaddleOCR não instala na Stack 2026. A Task 2
> passa a usar o detector PP-OCRv3 do `opencv_zoo` pelo `cv2.dnn.TextDetectionModel_DB`, sem reconhecimento:
> `.superpowers/sdd/2026-10-21-f2-e-documentos/task-2-brief-v2.md`. Sem `det_texto_tela_thumb` na Task 3.

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (`paddleocr==3.7.0`, `paddlepaddle==3.3.1`), `src/previa/detectores.py`,
  `configs/multimodal.yaml`
- Test: `tests/test_detectores.py`

**Interfaces:**
- Consumes: `ler_config_detectores()` (Task 1).
- Produces:
  - `texto_na_capa(rgb: np.ndarray, ocr, conf_min: float) -> dict` com `{"texto": str, "n_linhas": int,
    "area_texto": float}`; o `ocr` tem `predict(bgr) -> [{"rec_texts", "rec_scores", "rec_boxes"}]`.
  - `criar_ocr(cfg: dict)` → o `PaddleOCR` configurado como nas Global Constraints.

- [ ] **Step 1: instale** `uv add paddleocr==3.7.0 paddlepaddle==3.3.1`. Se o uv não resolver nas duas plataformas
  do `[tool.uv] environments`, ou o import falhar no macOS arm64, **pare e relate BLOCKED** com o erro: pela #31, o
  texto vira limitação e fica só o rosto.
- [ ] **Step 2: escreva os testes que falham**, com um OCR falso:
  - `test_texto_filtra_confianca_e_um_caractere`: leituras `["1 PRODUTO", "C", "base"]`, scores
    `[0.95, 0.99, 0.5]` → `texto == "1 PRODUTO"`, `n_linhas == 1`.
  - `test_area_texto_com_int16`: caixa `np.array([[0, 0, 300, 200]], dtype=np.int16)` numa capa 400×600 →
    `area_texto == pytest.approx(60000 / 240000)` (sem estouro de `int16`).
  - `test_ocr_recebe_bgr`: como no rosto.
  - `test_ocr_vazio`: `rec_texts == []` → `{"texto": "", "n_linhas": 0, "area_texto": 0.0}`.
- [ ] **Step 3: rode e veja falhar.**
- [ ] **Step 4: implemente** `texto_na_capa` e `criar_ocr`, com `conf_min` vindo da config (0.8).
- [ ] **Step 5: rode e veja passar**; ruff.
- [ ] **Step 6: confira com o PaddleOCR real** numa imagem sintética com "1 PRODUTO, 20 SEGUNDOS" desenhada com PIL
  (como o `frame_texto.png` da 4.6); anote a saída e o tempo por imagem.

### Task 3: lote das capas, bloco do runner e concordância com o VLM

> **Aviso (emenda de 21/10):** esta Task 3 foi reescrita em `.superpowers/sdd/2026-10-21-f2-e-documentos/task-3-brief-v2.md` (6 colunas, sem `det_texto_tela_thumb`, `det_n_caixas_texto_thumb`). Siga o brief v2.

**Files:**
- Modify: `src/previa/detectores.py`, `src/previa/experimento.py` (`ler_extra`, `FORA_DO_MODELO`)
- Test: `tests/test_detectores.py`, `tests/test_experimento.py`

**Interfaces:**
- Consumes: `rostos_na_capa`, `texto_na_capa`, `criar_yunet`, `criar_ocr` (Tasks 1 e 2); `recortar_9_16` de
  `previa.thumbs`; `PASTA` (thumbs) e o JSONL da rubrica (`DADOS / "vlm" / "rubrica_v1.jsonl"`, como no
  `ler_extra("vlm")`).
- Produces:
  - `features_da_capa(rgb: np.ndarray, yunet, ocr, conf_min: float) -> dict` com as 7 colunas `det_*_thumb` das
    Global Constraints. **É a função que a tool chama na capa do criativo (F17).**
  - `detectar_capas(ids, pasta_thumbs: Path, destino: Path, yunet, ocr, conf_min: float, lote: int,
    log=print) -> dict` com `{"novos", "pulados", "sem_thumb", "erros"}`; lotes `lote_*.parquet` com `video_id`.
  - `carregar_deteccoes(pasta: Path) -> pd.DataFrame` (índice `video_id`).
  - `concordancia(det: pd.DataFrame, rotulos: pd.DataFrame) -> pd.DataFrame`: uma linha por par
    (`det_rosto_thumb`×`tem_rosto`, `det_texto_thumb`×`texto_na_tela`) com `n`, `concordancia`, `kappa`
    (`cohen_kappa_score`; NaN se um lado tiver uma classe só) e `divergentes` (quantos).
  - CLI: `uv run python -m previa.detectores capas` (lote + `DADOS / "det_thumb_v<versao>.parquet"`) e
    `uv run python -m previa.detectores concordancia` (imprime a tabela e salva os divergentes em
    `DADOS / "det_thumb_divergentes.csv"` para ele assistir).
  - `ler_extra("det_thumb")` no runner.

- [ ] **Step 1: escreva os testes que falham:**
  - `test_features_da_capa_colunas_e_tipos`: com os falsos das Tasks 1 e 2 → exatamente as 7 chaves; `det_rosto_thumb`
    é `n_rostos > 0`, `det_texto_thumb` é `n_linhas > 0`.
  - `test_lote_retoma_sem_refazer`: 3 capas em `tmp_path`, `lote=2`; a 1ª chamada dá `novos == 3`; a 2ª,
    `novos == 0, pulados == 3`, e o falso não é chamado de novo.
  - `test_lote_capa_corrompida_conta_erro_e_segue`: um `.jpg` com bytes inválidos → `erros == 1`, as outras salvas.
  - `test_concordancia_kappa_e_divergentes`: 4 vídeos, rosto det `[T, F, T, F]` × VLM `[T, F, F, F]` →
    `concordancia == 0.75`, `divergentes == 1`; texto com uma classe só → `kappa` NaN.
  - `test_ler_extra_det_thumb_sem_texto` (em `test_experimento.py`, com a fixture de `DADOS` temporário): um parquet
    com as 7 colunas → o bloco devolve 6 colunas `float`, índice `post_id`, sem `det_texto_tela_thumb`.
- [ ] **Step 2: rode e veja falhar.**
- [ ] **Step 3: implemente.** O lote segue o `embedar_thumbs` (escrita em `.tmp` e `os.replace`); uma capa com erro
  vai para o `log` com o motivo e entra no resumo. A concordância casa `video_id` com o `post_id` da rubrica e usa só
  a interseção. No runner, `"det_texto_tela_thumb"` entra no `FORA_DO_MODELO` ("texto lido na capa; vai para a
  explicação, não para o modelo (4.6)").
- [ ] **Step 4: rode e veja passar**; `uv run pytest` inteiro e ruff.
- [ ] **Step 5: custo e convivência, sem resultado nenhum:** (a) rode `features_da_capa` em 50 capas reais
  (`sorted(PASTA.glob("*.jpg"))[:50]`), sem gravar nada em `DADOS`, e relate segundos por capa e a estimativa para
  30.981; (b) num subprocesso, `previa.openmp.um_runtime()`, depois `import lightgbm`, `import cv2`, `import
  paddleocr`, crie os dois detectores, rode 1 capa e treine um `lightgbm.LGBMRegressor(n_estimators=5)` em dados
  aleatórios. Relate se houve segfault: a F17 decide por aí se a tool chama os detectores no próprio processo.
- [ ] **Step 6: não rode `capas` nem `concordancia` na base** (esperam a H10).

### Task 4: rascunho do model card (F3)

**Files:**
- Create: `docs/MODEL_CARD.md`

- [ ] **Step 1: leia** o hub `roadmap.mdx` (o que o model card cobre), a 6.5 (o recurso `model://card`), a 6.7 (o
  checklist) e a 7.4, em `~/autoestudos/autoestudos/docs/`; o `docs/DATA_CARD.md` (molde e tom); o `README.md`
  (seções 5 a 8); o `docs/DECISOES.md`; o `docs/HIPOTESES.md`; e as configs em `configs/`.
- [ ] **Step 2: escreva** `docs/MODEL_CARD.md` em PT-BR, nas seções que as aulas pedem (no mínimo: uso pretendido e
  fora de escopo; modelo e versão; dados de treino e o corte por data; alvo; métricas do teste temporal e do canal
  novo com IC; cobertura do intervalo; o `confidence_flag`; limitações e vieses; licenças dos checkpoints: SigLIP 2,
  Whisper/faster-whisper, o encoder de texto da config de texto, Gemini (termos da API), YuNet e PaddleOCR; contato e
  versão). Cada número vem com a fonte (arquivo, decisão ou run id); o que depende de run lacrado, do teste ou de uma
  peça a fazer fica `PREENCHER:` com o que vai ali. Cada licença vem do model card ou do repositório oficial do
  checkpoint (cite o link).
- [ ] **Step 3: confira** que nenhum número foi inventado: liste no relatório cada número e a fonte.

### Task 5: comparação com a literatura (F29) e limitações (F42)

**Files:**
- Modify: `README.md` (seções 7 e 8), `docs/DATA_CARD.md` ("Vieses e limitações conhecidos")

- [ ] **Step 1: leia** o hub `leituras.mdx` inteiro, a 6.7 (README), a 7.3 (a "lembrança" do pré-treino), a 1.5 (a
  varredura semanal) e a RAG 6 (recall dos comparáveis), em `~/autoestudos/autoestudos/docs/`.
- [ ] **Step 2: F29.** Na seção 7 do README, uma subseção "Comparação com a literatura": a tabela com o que a
  literatura do hub reporta (Spearman de 0,55 a 0,85; par-a-par de 76% nos 11 mil Shorts; +0,02 a +0,04 do visual,
  conferidos no `leituras.mdx`, com a referência de cada um), a coluna do Prévia com `PREENCHER:` e uma nota de por
  que os números não são diretamente comparáveis (alvo, base, régua intra-canal, período). Marque com `V:` o
  parágrafo que o Vinicius reescreve.
- [ ] **Step 3: F42.** Na seção 8 do README e no data card, as três limitações, cada uma com a aula: o VLM recebe a
  URL pública (risco pequeno de "lembrança" do pré-treino, 7.3); não há varredura semanal completa do coletor (1.5);
  o recall dos comparáveis não foi medido (RAG 6). Mantenha o estilo das limitações que já existem; não apague nada
  delas.
- [ ] **Step 4: confira** que os números da literatura batem com o `leituras.mdx` (cite a linha no relatório).
