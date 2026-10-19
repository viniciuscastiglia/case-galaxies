# Amostra multimodal e rubrica do VLM — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** sortear a amostra da #12 e extrair a rubrica da #13 com o Gemini, em background, a partir de 18/10.

**Architecture:** `previa.amostra` sorteia os Shorts pela regra escrita (sem curtidas, sem abrir o teste).
`previa.vlm` tem a `Rubrica` (Pydantic), o pedido ao Gemini por URL e o laço cacheado que respeita a cota diária.

**Tech Stack:** google-genai 2.25.0 (Stack 2026), tenacity 9.1.4, pydantic 2.13.5, httpx 0.28.1, duckdb.

**Spec:** desenho aprovado no chat em 18/10; decisões #12 e #13 (`docs/DECISOES.md`); aula 4.7 inteira.

## Global Constraints

- Amostra: 20 por canal, seed 0, só primeiras cópias (`duplicate_group == post_id`), alocação proporcional aos
  blocos treino, calibração e teste do canal (`split_treino_cal_teste`); gap e pré-01/07/2021 ficam fora. Lê só
  `post_id`, `channel_id`, `published_at`, `duplicate_group` e `sector`: nunca `metric_target` (o teste não é aberto,
  nada vai para `teste_aberto.log`). Duração de `videos.duration_s` no DuckDB.
- Rubrica de 6 campos com âncoras 0/5/10 na `description` (4.7): `forca_hook_0a10` (0:00 a 0:03, como na aula),
  `energia_ritmo_0a10`, `tem_rosto`, `close_up_produto`, `before_after_ou_transformacao`, `texto_na_tela`.
- Pedido (4.7): URL `https://www.youtube.com/watch?v=<id>`, `fps=1`, `MEDIA_RESOLUTION_LOW`, `temperature=0`,
  `response_mime_type="application/json"`, `response_schema=Rubrica`; prompt `BASE` + regra do setor (beleza,
  automotivo). Modelo `gemini-3.5-flash-lite`, no config.
- Validação sempre por `Rubrica.model_validate_json(resp.text)`; nunca `resp.parsed` (erro comum 3).
- Retry (tenacity) só em 429 e 5xx, `wait_random_exponential(multiplier=2, max=120)`, 6 tentativas (erro comum 4).
- Cota: no máximo 8 h (28.800 s) de vídeo por dia do Pacífico (zera às 4h de Brasília); 429 depois das tentativas =
  cota esgotada, o laço para e avisa (princípio 7). Intervalo mínimo entre pedidos no config.
- Cache por versão (`data/vlm/rubrica_v<versao>.jsonl`, uma linha por vídeo, com `_tokens`, `_modelo`, `_em`);
  erros de validação e 400 em `erros_v<versao>.jsonl`, refeitos na rodada seguinte. Prompt ou schema mudou: sobe a
  versão (erro comum 6). Teste-reteste: 50 Shorts com `versao=2`.
- Chave: `GOOGLE_API_KEY` do `.env`, nunca impressa; mensagens de erro passam por `sem_chave`.
- Só Shorts publicados vão ao free tier (princípio 5). Nada de `data/` no git. Commit só quando o Vinicius pedir.

## Review Focus

- Short removido ou privado (o Gemini responde 400 ou 403): vai para os erros e o laço segue.
- Resposta com nota fora da escala: `ValidationError`, registrada como erro, sem retry.
- Rodar o laço duas vezes: nenhum vídeo é pedido duas vezes.
- Cota esgotada no meio: para limpo, com o que já foi feito salvo.
- A ordem de processamento é sorteada: se parar no meio, o que existe é uma subamostra aleatória.

---

### Task 1: `previa.amostra`

**Files:** Create `src/previa/amostra.py`, `configs/multimodal.yaml`; Test `tests/test_amostra.py`.

**Produces:** `alocar(contagens: dict[str, int], n: int) -> dict[str, int]` (maiores restos; nunca mais que o
disponível); `sortear_amostra(df, cfg_avaliacao, por_canal=20, seed=0) -> DataFrame` com `post_id`, `channel_id`,
`published_at`, `sector`, `bloco` e `ordem` (permutação com a seed); CLI `uv run python -m previa.amostra` grava
`data/amostra_multimodal.csv` com `duration_s`.

- [ ] Testes: `test_alocar_proporcional` (100/50/50 com n=20 → 10/5/5), `test_alocar_canal_pequeno` (total 7 → 7),
  `test_amostra_so_primeiras_copias`, `test_amostra_fora_do_gap_e_antes_de_2021`, `test_amostra_reprodutivel`
  (mesma seed, mesma amostra), `test_amostra_nao_le_curtidas` (o DataFrame de entrada sem `metric_target` funciona).
- [ ] Falham no import; implementar; passam.

### Task 2: `previa.vlm` — rubrica e pedido

**Files:** Create `src/previa/vlm.py`; Test `tests/test_vlm.py`; Modify `pyproject.toml` (`google-genai==2.25.0`).

**Produces:** `Rubrica`; `montar_prompt(setor) -> str`; `montar_pedido(url, setor) -> (contents, config)`;
`extrair_rubrica(cliente, url, setor, modelo) -> (Rubrica, int)`.

- [ ] Testes com `httpx.MockTransport` (como na 4.7): `test_pedido_vai_com_url_fps_e_resolucao_baixa`,
  `test_resposta_valida_vira_rubrica`, `test_nota_fora_da_escala_levanta`, `test_400_nao_repete` (1 pedido),
  `test_429_repete_e_depois_passa` (espera zerada no teste), `test_rubrica_tem_6_campos_com_ancoras`.
- [ ] Falham; implementar; passam.

### Task 3: laço cacheado e CLI

**Files:** Modify `src/previa/vlm.py`; Test `tests/test_vlm.py`.

**Produces:** `processar(amostra, extrair, pasta, versao, limite_s, agora) -> dict` (contagens: ok, erros, pulados,
segundos_hoje, parou_por); CLI `uv run python -m previa.vlm rodar [--limite N] [--versao V]` e
`uv run python -m previa.vlm reteste` (os 50 primeiros da ordem, `versao=2`).

- [ ] Testes com um `extrair` falso: `test_pula_o_que_ja_esta_no_cache`, `test_erro_vai_para_o_arquivo_e_segue`,
  `test_para_no_limite_de_segundos_do_dia`, `test_para_quando_a_cota_acaba`, `test_processa_na_ordem_sorteada`.
- [ ] Falham; implementar; passam. `uv run pytest` e `uv run ruff check .` (só os 5 da M2 falhando).

### Task 4: rodar

- [ ] Sortear a amostra; conferir contagens por bloco e setor.
- [ ] `rodar --limite 20`; conferir as 20 respostas (campos, tokens, erros).
- [ ] O resto em background até a cota; log em `data/logs/vlm.log`.
- [ ] Registrar no `docs/DECISOES.md` o que for encanamento.
