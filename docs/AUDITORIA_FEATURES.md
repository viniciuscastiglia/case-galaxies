# Auditoria das features

Exercício da aula 2.2 (item 1): auditoria da linha do tempo, coluna a coluna. Para cada feature candidata, a tabela
registra de onde ela vem, **quando** o valor existe e se ela pode vazar. As candidatas saem do que a coleta já guarda,
do histórico do canal e do vídeo (módulo 4). A escolha final das features vem nos módulos 3 e 4.

Preenchida pelo Vinicius em 12/10, com os ajustes da revisão.

- **Quando existe** (aula 1.7): quando foi observado o valor que está na **nossa base**, ou seja, no treino. Em
  produção quase tudo existe no instante da publicação; o risco de vazamento mora no treino. As opções são: no
  instante da publicação, no último post anterior, no dia da coleta, ou só depois de publicar.
- **Família de risco** (aula 2.2): alvo, temporal, grupo, pré-processamento, duplicatas, ou nenhuma.
- **Teste:** o que confere que ela não vaza. Para as colunas da API, serve o `auditar_datas` do exercício da 1.7.

## Da API (já coletadas)

| Feature | Fonte | Quando existe | Família de risco | Teste |
| :--- | :--- | :--- | :--- | :--- |
| hora e dia da semana | `videos.list`, `snippet.publishedAt` (convertido para Brasília) | instante da publicação | nenhuma | `auditar_datas` |
| `duracao_s` | `videos.list`, `contentDetails.duration` | instante da publicação | nenhuma | `auditar_datas` |
| atributos do título (tamanho, pergunta, hashtags, emojis) | `videos.list`, `snippet.title` | dia da coleta (a API devolve a versão atual, que pode ter sido editada) | temporal (brando) | `auditar_datas`, com o título mapeado para `collected_at` |
| atributos da descrição | `videos.list`, `snippet.description` | dia da coleta (idem) | temporal (brando) | `auditar_datas`, com `collected_at` |
| tags | `videos.list`, `snippet.tags` | dia da coleta (idem) | temporal (brando) | `auditar_datas`, com `collected_at` |
| categoria | `videos.list`, `snippet.categoryId` | dia da coleta (idem) | temporal (brando) | `auditar_datas`, com `collected_at` |
| `log_inscritos` | `channels.list`, `statistics.subscriberCount` (`subscribers_at_collection`) | dia da coleta (o mesmo número para todos os Shorts do canal, inclusive os de 2020) | temporal | `auditar_datas`, com `collected_at`; retreinar sem ela e comparar |

O `auditar_datas` só pega esses riscos se o mapa disser a verdade. Com o título mapeado para `published_at`, ele
devolve 0% e a feature passa; com `collected_at`, devolve 100% e acende o alerta.

## Da curadoria

| Feature | Fonte | Quando existe | Família de risco | Teste |
| :--- | :--- | :--- | :--- | :--- |
| setor | `data/canais.csv`, coluna `setor` (revisão humana, aula 1.4) | dia da coleta | nenhuma | |
| tipo de canal (marca ou criador) | `data/canais.csv`, coluna `tipo` (revisão humana, aula 1.4) | dia da coleta | nenhuma | |

## Do histórico do canal (cálculo, módulo 3)

| Feature | Fonte | Quando existe | Família de risco | Teste |
| :--- | :--- | :--- | :--- | :--- |
| `mediana_madura` | `y` dos posts anteriores do canal com 30+ dias | último post anterior maduro | temporal | mudar os posts futuros não muda o valor (pytest, aula 2.5) |
| `media_ult_k` | `y` dos k posts anteriores do canal | último post anterior | temporal | idem |
| `n_anteriores` | contagem dos posts anteriores do canal | último post anterior | temporal | idem |
| `dias_desde_ultimo` | `published_at` do post anterior do canal | último post anterior | temporal | idem |

As curtidas dos posts anteriores foram lidas na coleta, com idade mediana de 324 dias (viés 3 do data card). Por
isso, mesmo a `mediana_madura` carrega curtidas que chegaram depois do post que ela descreve. A regra dos 30 dias
deixa esse resto pequeno quando as curtidas estabilizam, e a medição da maturação (19 a 26/10) diz quão pequeno.

## Do vídeo (extração, módulo 4)

| Feature | Fonte | Quando existe | Família de risco | Teste |
| :--- | :--- | :--- | :--- | :--- |
| rosto no gancho | arquivo do vídeo, primeiros segundos | instante da publicação | nenhuma | |
| texto na tela | arquivo do vídeo, frames | instante da publicação | nenhuma | |
| cortes | arquivo do vídeo | instante da publicação | nenhuma | |
| fala transcrita | áudio do vídeo | instante da publicação | nenhuma | |
| nota do gancho | VLM sobre os frames do gancho | instante da publicação | alvo, se o prompt receber métricas | conferir que o prompt não recebe views nem curtidas |

## Da capa (modelo final)

Acrescentada em 22/10/2026 para cobrir as features do modelo final (`configs/experimentos/final.yaml`), que vieram
depois desta tabela.

| Feature | Fonte | Quando existe | Família de risco | Teste |
| :--- | :--- | :--- | :--- | :--- |
| embedding da capa (`thumb_pc*`, PCA de 16 do SigLIP 2) | a thumbnail atual pela URL do `i.ytimg.com`, recortada na faixa 9:16 | dia da coleta (o canal pode trocar a capa depois de publicar) | temporal (brando); pré-processamento (o PCA) | `test_pca_ajustado_so_no_treino`; paridade com a tool em `test_paridade_treino_uso_com_extras` |
| sondas da capa (`sonda_*_thumb`, 10) | o mesmo embedding contra frases fixas (`configs/multimodal.yaml`) | dia da coleta | temporal (brando) | `test_paridade_com_o_caminho_do_treino` (a tool calcula igual ao treino) |
| aparência da capa (`thumb_brilho`, `thumb_contraste`, `thumb_saturacao`, `thumb_colorfulness`) | a mesma thumbnail, pelo OpenCV | dia da coleta | temporal (brando) | `test_paridade_com_o_caminho_do_treino` |

Na previsão, a capa é a que a marca envia, que existe antes de publicar: o risco mora só no treino, onde a capa pode
ser uma versão trocada depois. O data card registra isso como viés ("Capa da data da coleta").

## Removidas ou de controle

| Feature | Fonte | Quando existe | Família de risco | Teste |
| :--- | :--- | :--- | :--- | :--- |
| views, curtidas e comentários do próprio vídeo | `videos.list`, `statistics` (tabela `snapshots`) | depois de publicar | alvo | fora das features |
| `target_age_days` (idade do rótulo) | data do snapshot − `published_at` | depois de publicar | temporal | candidata a variável de controle: entra no treino e fica fixada em 30 na previsão (aula 1.7, viés 3; experimento do módulo 3) |
