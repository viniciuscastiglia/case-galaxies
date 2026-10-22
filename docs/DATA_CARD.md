# Data card: base de Shorts do Prévia (v1, 20/10/2026)

> Exercício da [aula 1.8](https://inteliacademyclub.github.io/autoestudos/aulas/ml-preditivo-e-agentes/formulacao-e-dados/documentacao-etica-e-termos/).
> A v1 descreve a base congelada dos experimentos (`data/base_v1.parquet`, exportação de 19/10/2026, 06h46, hash de
> desenvolvimento `09f6035c11c9`). Os números do período de teste (01/04 a 14/09/2026) são contagens de linhas,
> sem ler curtidas: o teste abre uma vez, no fim. A v0.1 (08/10) está no histórico do git.
>
> **v2 (decisão #33):** `data/base_v2.parquet`, exportação de 21/10/2026, 06h25, hash `96fd13a28156` (do arquivo inteiro; o hash da base cortada, a que o modelo de uso de fato
> treina, vai na tag `hash_base` do run do modelo de uso), congelada por
> `python -m previa.dados congelar`. É a base do modelo de uso, retreinado com a mesma receita depois do teste final:
> ela gasta de propósito os meses do teste da v1 no treino e na calibração (`configs/avaliacao_v2.yaml`, datas a
> confirmar pelo Vinicius), e o `carregar_base` a recusa antes da primeira abertura do teste. Os números dela não se
> comparam com os do teste da v1. É apagada em 30/10 com a v1.

## Origem

- **Fonte principal:** YouTube Data API v3, com a chave de um projeto Google Cloud próprio. São dados públicos lidos
  sem autorização dos donos dos canais ("Non-Authorized Data" nas Developer Policies).
- **Endpoints:** `channels.list` (metadados e inscritos), `playlistItems.list` (playlist de uploads), `videos.list`
  (detalhes e estatísticas), `channelSections.list` (canais em destaque) e `search.list` (só para descobrir canais).
- **Fontes de apoio:** nenhuma em uso. SMTPD e MicroLens-100K, se entrarem, servem só para ensaio, nunca para teste
  (aula 1.3).
- **Mídia (módulo 4):** a thumbnail pública de cada Short (`hqdefault`, 480 × 360) para a base toda, e os vídeos de
  uma amostra de 1.400 Shorts, baixados em 360p com o yt-dlp, só para extrair features, e apagados em seguida.

## Período

- **Publicação dos Shorts rotulados:** de 14/05/2014 a 19/09/2026. Antes de 01/07/2021 há 336 Shorts (1,1%),
  vídeos verticais curtos de antes do formato no Brasil, que ficam só no histórico do canal (#2).
- **Coletas:** de 05/10 a 19/10/2026 para a v1, diárias às 05h15 (launchd), até 30/10. Na primeira coleta, de
  05/10, entrou o histórico inteiro de cada canal.
- **Blocos do split (#3):** treino de 01/07/2021 a 01/01/2026 (20.579 Shorts), gap em jan/2026 (1.025),
  calibração de 01/02 a 31/03/2026 (1.882) e teste de 01/04 a 14/09/2026 (7.265, dos 70 canais).

## Volume

31.303 Shorts rotulados de 70 canais (35 por setor), todos validados no esquema mínimo. São 30.983 criativos e 320
cópias (duplicatas, #1). A tabela de vídeos do coletor tem 95.998 vídeos, 32.813 deles Shorts; vira rótulo o Short
com 30 dias ou mais, sem teto por canal (o teto é parâmetro do experimento, #8).

Shorts rotulados por setor e faixa de inscritos (a faixa é a da curadoria, com os inscritos de outubro):

| Setor | até 10 mil | 10–100 mil | 100 mil–1 mi | mais de 1 mi | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Automotivo | 126 | 2.076 | 6.988 | 5.100 | 14.290 |
| Beleza | 311 | 5.853 | 7.644 | 3.205 | 17.013 |
| **Total** | **437** | **7.929** | **14.632** | **8.305** | **31.303** |

Canais incluídos:

| Setor e tipo | até 10 mil | 10–100 mil | 100 mil–1 mi | mais de 1 mi | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Automotivo, criador | 0 | 6 | 11 | 4 | 21 |
| Automotivo, marca | 2 | 2 | 9 | 1 | 14 |
| Beleza, criador | 0 | 7 | 8 | 5 | 20 |
| Beleza, marca | 2 | 5 | 5 | 3 | 15 |

Fora da base, o **golden set** (#4): 30 Shorts maduros de 01/04 a 14/09/2026, de 6 canais sorteados entre as
reservas revisadas (2 marcas e 1 criador por setor), com o histórico dos 6 canais (1.793 Shorts), em `data/golden/`.

## Licença e termos

- **Termos:** YouTube API Services Terms of Service e Developer Policies, na versão de 14/09/2026 lida na aula 1.8.
- **Uso:** educacional e interno, sem produto e sem clientes reais consumindo as previsões.
- **Sem redistribuição:** o repositório (privado) tem o código, as planilhas de curadoria e este data card. Base,
  parquets, thumbnails, vídeos, frames e áudio ficam fora do git (`.gitignore`).
- **Retenção:** a coleta diária renova as estatísticas de todo vídeo cujo dado mais recente passou de 25 dias e apaga
  os que sumiram da API. Em seguida, uma auditoria confere se algum vídeo passou de 30 dias sem renovação; se passou,
  a execução falha e o Mac avisa. A base congelada v1 é só leitura e é apagada com o resto.
- **Mídia:** o vídeo da amostra é apagado logo depois da extração, também quando uma etapa falha; ficam só as
  features (cortes, cor, fala, energia do áudio). As thumbnails ficam em `data/thumbs/` até 30/10.
- **Gemini (free tier):** recebeu só URLs de Shorts já publicados (a rubrica da amostra e o reteste de 50). Um
  criativo não publicado nunca vai ao free tier (princípio 5 do `CLAUDE.md`).
- **Apagamento:** os dados brutos serão apagados em **30/10/2026**. Desse dia em diante, o `scripts/coletar.sh` não
  coleta e avisa todo dia, até o passo a passo abaixo ser feito.
- **Tensões conhecidas com as políticas (aula 1.8):** a regra dos 30 dias (III.E.4.d) contra os snapshots antigos,
  que são o rótulo; métricas derivadas (III.E.4.h) contra o alvo relativo; agregação de canais de donos diferentes
  (III.E.2); o download de mídia (III.E.1 e III.E.6), limitado à amostra, em ritmo lento, com apagamento logo depois
  da extração.

Passo a passo do apagamento, em 30/10 (v1: inclui as pastas do módulo 4 e do golden set):

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.previa.coletor.plist
rm ~/Library/LaunchAgents/com.previa.coletor.plist
rm -f data/shorts.duckdb data/*.parquet data/cota*.json data/amostra_multimodal.csv data/conferencia_shorts*.csv
rm -rf data/logs data/thumbs data/emb_thumb_v1 data/cache data/vlm data/golden data/videos data/ablacao media/ artefatos/
```

<!-- V: decidir até 30/10 o que fazer com o MLflow (`mlflow.db` e `mlruns/`): ele guarda as métricas agregadas,
que o README cita pelo run id, e as previsões por `post_id` (`previsoes.json`), que são dado derivado da API. Uma
opção é apagar só os `previsoes.json` e manter as métricas. -->

## Coleta

- **Curadoria:** `data/canais.csv` lista incluídos e excluídos, com o motivo de cada um. Os critérios foram escritos
  antes de olhar resultados (`docs/PLANO_DA_BASE.md`). Origem dos 70 canais: 33 da lista manual, 19 da lista manual
  com o handle corrigido pela busca e 18 da busca por termo de 08/10 (15 pelo modo vídeo, 3 pelo modo canal).
- **Método:** playlist de uploads (`UU…`) → `playlistItems.list` → `videos.list`, de forma incremental. Snapshots
  diários dos Shorts com até 60 dias e renovação dos vencidos.
- **Regra de Short:** até 60 s se publicado antes de 15/10/2024, até 180 s depois (quando o limite mudou para 3
  minutos), e não horizontal: largura menor ou igual à altura no player, com formato desconhecido contando como
  Short. A conferência de 50 Shorts a olho (#19, F14, 21/10) deu **49 de 50 de acordo com a regra do coletor (98%)**. Por
  estrato: vertical recusado na borda 10/10; Short até 60 s antes de 15/10/2024 9/10; Short depois, de 61 a 180 s
  15/15; Short depois, até 60 s 15/15. A regra sem `isShort` erra em 1 de 50 nas bordas (`data/conferencia_shorts.csv`,
  `python -m previa.conferencia resultado`; DECISOES de 21/10).
- **Teto por canal:** a exportação leva todos os Shorts rotulados. O teto de 400 vale só para as linhas de treino,
  como parâmetro do experimento (H6): o histórico do canal precisa de todos os posts (paridade treino/uso).
- **Duplicatas (#1, 14/10, entre canais desde 17/10):** a coluna `duplicate_group` leva o ID da primeira cópia
  publicada do mesmo criativo: título normalizado e duração iguais **e** thumbnail com dHash de 64 bits a até 10
  bits da cópia mais antiga (`src/previa/duplicatas.py`, aula 2.2). Na v1: 320 cópias. Nada é apagado: só a
  primeira cópia vira linha de treino e de teste, e o histórico do canal usa todas. Uma thumbnail que falha vira
  aviso, e o post fica pela regra de título e duração.
- **Maturação (medida em 19/10, com os snapshots de 05 a 19/10):** o crescimento mediano das curtidas em ~12 dias é
  de +69% para quem foi visto com 0 a 3 dias, +4,1% com 3 a 7, +0,8% com 7 a 14, +0,1% com 14 a 21 e 0% a partir de
  21 dias (`uv run python -m previa.maturacao`). A regra dos 30 dias tem folga.
- **Versão:** `src/previa/coletor.py` no git. Mudanças relevantes: retry de rede (06/10), preferência por IPv4 e
  busca por termo (08/10), validação e auditoria a cada coleta (08/10).
- **Cota:** de 180 a 205 unidades por dia na coleta diária (outubro), de um limite de 10 mil. Picos: 3.805 na
  primeira coleta (05/10), 8.615 na avaliação de 614 candidatos (08/10) e 384 no golden set (19/10).

## Setores e estratificação

Beleza e automotivo, 35 canais cada. Nos Shorts rotulados, o automotivo tem 83% de criadores; a beleza, 50% de
criadores e 50% de marcas, depois dos 20 criadores de beleza incluídos em 08/10. Idioma e país: canais brasileiros,
em português. O país da API é opcional e vem vazio em muitos canais, por isso a conferência final de país e idioma é
humana (critério 2 do plano).

## Alvo e rótulo

- **Rótulo** (`metric_target`): o `likeCount` do primeiro snapshot com 30 dias ou mais de idade.
- **Alvo de treino** (README, aula 1.6): o log da razão entre as curtidas do Short e a mediana madura das curtidas
  dos posts anteriores do mesmo canal (publicados ao menos 30 dias antes), definido com 10+ posts maduros (#20).
- **Idade do rótulo:** na base de desenvolvimento (até 31/03/2026), mediana de 636 dias, 10% com até 261 e 10% com
  mais de 1.361; **85% foram medidos com mais de 300 dias**, porque a primeira coleta pegou o histórico inteiro de
  cada canal. No teste (publicados de abril a setembro de 2026, coletados em outubro), os rótulos têm de 1 a 6 meses.
- **Curtidas ocultas:** viram `None`, nunca 0, e atingem 3,7% dos Shorts da base de desenvolvimento. Canais com mais
  da metade das curtidas ocultas ficaram fora na curadoria, e o esquema falha se a base passar de 10%.
- **Distribuição (desenvolvimento):** mediana de 271 curtidas e percentil 90 de 5.922. É a cauda longa da aula 1.6.

## Features derivadas (módulo 4)

| Tabela | O que tem | Cobertura |
| :--- | :--- | :--- |
| `emb_titulo_v1.parquet` | embedding do título (MiniLM multilíngue, 384 dimensões; o PCA é ajustado no treino de cada fold) | os 31.303 |
| `emb_thumb_v1/` | embedding da capa (SigLIP 2, faixa central 9:16) | 30.981 |
| `sondas_thumb_v1.parquet` | 10 sondas relativas da capa (4.4) | 30.981 |
| `aparencia_thumb_v1.parquet` | brilho, contraste, saturação e colorfulness da capa (4.3) | 30.981 |
| `vlm/rubrica_v1.jsonl` | rubrica do Gemini: gancho, energia e ritmo, rosto, produto em close, antes/depois, texto na tela (kappa de 0,72 a 1,00 no reteste de 50) | 1.400 da amostra |
| `cache/` | cortes, movimento e aparência dos frames, fala e energia do áudio | a amostra, em andamento |

A amostra (#12) tem 20 Shorts por canal, sorteados com seed 0 entre as primeiras cópias, proporcional ao tamanho de
cada bloco de tempo, sem olhar curtidas: 930 no treino, 111 na calibração e 359 no teste (19,9 h de vídeo). A capa é
a da data da coleta, que o canal pode ter trocado depois da publicação (4.2, erro comum 6).

## O que esta base NÃO representa

Lista do Vinicius em 05/10 (exercício da aula 1.3), com os complementos da revisão:

- **Vídeos longos:** só Shorts, por escopo.
- **Canais com menos de 30 Shorts:** uma marca que acabou de começar a postar Shorts está fora do que o modelo viu.
  Previsões para canais com pouco histórico saem marcadas com baixa confiança (aula 5.4).
- **Retenção, CTR e impressões:** só o dono do canal vê. Não podem ser features nem aparecer na explicação.
- **O período depois de ~19/09/2026:** mudanças no algoritmo ou nas tendências depois do corte de rótulos (drift,
  aula 1.2).
- **Outros setores, países e idiomas:** só beleza e automotivo, canais brasileiros, em português.
- **Canais pequenos:** a lista inicial foi feita de memória e favorece canais grandes. A busca de 08/10 trouxe
  criadores entre 10 e 100 mil inscritos, mas ainda não há nenhum criador com menos de 10 mil.
- **Mídia paga:** a API não diz se um Short foi impulsionado, então parte das curtidas pode vir de anúncio.

Complementos do Claude, a partir do modelo da aula (revisar):

- **Canais inativos:** quem não postou Short nos 90 dias antes da curadoria ficou fora.
- **Canais que ocultam curtidas ou inscritos:** ficaram fora pelos critérios 5 e 6.
- **Motos:** fora do setor automotivo, por decisão do plano.

## Vieses e limitações conhecidos

Escritos pelo Vinicius em 08/10 (exercício da aula 1.8).

1. **Mix de setor e tipo.** No setor beleza, 74% dos Shorts são de marcas, enquanto no automotivo 74% são de
   criadores. O que isso muda é que o modelo não vai aprender tanto sobre criadores de beleza e marcas de automotivo,
   podendo dar uma resposta mais fraca nesses casos. Além disso, se a métrica vier muito diferente entre beleza e
   automotivo, não será possível saber se o problema está no setor ou no tipo de canal.
2. **Inscritos como foto da coleta.** O número de inscritos é uma foto do dia da coleta, e não do dia da publicação:
   a API do YouTube não dá acesso ao número de inscritos de um canal no dia em que um vídeo foi postado, apenas ao
   número atual. Com isso, o número de inscritos não entra como feature para o modelo, porque viraria vazamento de
   dados. Ao usarmos o número de inscritos para classificar a faixa de tamanho do canal, não sabemos em que faixa o
   canal de fato estava quando o vídeo foi postado, e metade dos rótulos tem quase 1 ano de vida, tempo suficiente
   para um canal mudar de faixa. Como consequência, a diferença de erro entre as faixas parece menor do que é. É
   possível reduzir isso fazendo a análise por faixa apenas com vídeos dos últimos 90 dias, porque é difícil a faixa
   mudar nesse período.
3. **Idade do rótulo.** Os Shorts da base deveriam ser todos medidos com cerca de 30 dias de vida, porque é quando as
   curtidas estabilizam (aula 1.7, confirmado nos nossos dados). Na nossa base, 53% dos Shorts foram medidos com mais
   de 300 dias (mediana de 324 dias). Mesmo que o Short típico pare de crescer com 30 dias, existem exceções que
   continuam crescendo e ficam com meses de curtidas extras na base. Além disso, Shorts muito antigos vêm de outro
   algoritmo e de um canal menor do que o atual, então não dá para separar o efeito da época do efeito da maturação.
   Quando usarmos o split temporal para treino e teste, o alvo vai acabar sendo medido de jeitos diferentes (num
   teste com os 15% mais recentes, a idade mediana do rótulo é de 52 dias; no treino, de 396), e o modelo pode errar
   para cima no teste de forma sistemática. E o cliente quer saber a performance do Short dele no próximo mês, não
   nos próximos anos. O lado bom: como o teste fica com os rótulos mais jovens, ele mede o que o cliente quer e vai
   mostrar se o viés atrapalha. O exemplo da aula usa a idade do rótulo como variável de controle: ela entra como
   feature no treino, e na hora da previsão a gente fixa o valor em 30. Outra saída é treinar só com rótulos de idade
   parecida, mas isso joga fora muitos dados. A escolha sai de um experimento no módulo 3.

<!-- V: os números dos itens 1 e 3 são de 08/10. Na v1: no item 1, a beleza tem 50% de marcas (eram 74%) e o
automotivo, 83% de criadores; no item 3, a base de desenvolvimento tem mediana de 636 dias e 85% com mais de 300.
Atualizar com as suas palavras, e o item 3 com o resultado da H5 -->

4. **Capa da data da coleta.** (complemento do Claude, a revisar) A thumbnail é a de outubro de 2026, e o canal
   pode tê-la trocado depois da publicação. Na previsão, a capa de A e B é a que a marca escolher.

5. **VLM recebe a URL pública.** (complemento do Claude, a revisar) A rubrica é extraída pelo Gemini a partir da URL
   pública do Short, então há um risco pequeno de o modelo "lembrar" do vídeo do pré-treino em vez de só descrevê-lo
   (aula 7.3). Não pedimos nem damos curtidas ao VLM; o risco não foi medido.
6. **Sem varredura semanal completa do coletor.** (complemento do Claude, a revisar) A coleta diária é incremental e
   para quando uma página inteira de IDs já é conhecida; a aula 1.5 pede uma varredura completa por semana, porque a ordem da API não é
   garantida. Não a fizemos, e um Short fora de ordem pode ter ficado de fora.
7. **Recall dos comparáveis não medido.** (complemento do Claude, a revisar) Não medimos quantos criativos parecidos
   a busca de comparáveis recupera (RAG 6, recall@K).

## Esquema

`src/previa/esquema.py`: `Post` (Pydantic, um registro) e `BaseShorts` (pandera, a tabela de treino). Campos:
`post_id`, `channel_id`, `published_at`, `caption` (título + descrição), `media_ref` (link do Short, nunca arquivo),
`metric_target`, `target_age_days` (≥ 30), `subscribers_at_collection` (fora das features de treino, aula 1.7),
`collected_at`, `sector` e `duplicate_group`. Checagens: ID único, coleta depois da publicação e no máximo 10% sem
curtidas. O modo estrito trava coluna faltando ou sobrando. A validação roda a cada coleta diária e grava
`data/base_minima.parquet`; a base congelada é uma cópia dela.

## Conformidade (checklist da aula 1.8, revisto em 20/10)

| Item | Situação | O que falta e até quando |
| :--- | :--- | :--- |
| `collected_at` e `age_days` em toda linha | ✅ | — |
| Refresh ou exclusão em até 30 dias, agendado | ✅ renovação diária (25 dias) + auditoria diária que falha alto | — |
| Base, vídeos, frames e áudios fora de repositório ou pasta pública | ⚠️ base fora do git, mas `data/canais.csv`, versionado, guarda inscritos e contagens da API | Decidir até 30/10: tirar os números da API do arquivo versionado (e do histórico do git) ou registrar a exceção |
| Nenhum vídeo baixado guardado | ✅ o vídeo da amostra é apagado depois da extração, também com falha de etapa; as thumbnails ficam até 30/10 | Apagar em 30/10 |
| Nada confidencial ou pessoal no Gemini free tier | ✅ só URLs de Shorts publicados | Manter: criativo não publicado nunca vai ao free tier |
| Comentários não coletados | ✅ só o `commentCount`; nenhum texto, nenhum autor | — |
| Sem reconhecimento facial | ✅ só a presença de rosto (um campo da rubrica), sem identificar ninguém | — |
| Data card com "o que NÃO representa" e vieses | ✅ | — |
| Esquema validado com pandera a cada exportação | ✅ desde 08/10 | — |
| Previsão apresentada como estimativa do modelo | ❌ a tool ainda não existe | No contrato da tool (A1) e no README, até 21/10 |
| Plano de exclusão dos dados brutos, com data | ✅ 30/10/2026; a coleta desliga sozinha | Seguir o passo a passo acima; decidir o MLflow |

## Contato e versão

- **Responsável:** Vinicius Castiglia. **Repositório:** `case-galaxies` (privado).
- **v0.1 (08/10/2026):** primeira versão, exercício da aula 1.8.
- **v1 (20/10/2026):** a base congelada dos experimentos (70 canais, 31.303 Shorts), a mídia do módulo 4, o golden
  set, a maturação medida e o passo a passo do apagamento com as pastas novas. Rascunho do Claude, a revisar.
