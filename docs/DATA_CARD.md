# Data card: base de Shorts do Prévia (v0.1, 08/10/2026)

> Exercício da [aula 1.8](https://inteliacademyclub.github.io/autoestudos/aulas/ml-preditivo-e-agentes/formulacao-e-dados/documentacao-etica-e-termos/).
> Números de 08/10, antes da primeira coleta dos 18 canais incluídos nesse dia. Próximas versões: congelamento da
> base de treino (12/10) e última coleta de rótulos (19/10).

## Origem

- **Fonte principal:** YouTube Data API v3, com a chave de um projeto Google Cloud próprio. São dados públicos lidos
  sem autorização dos donos dos canais ("Non-Authorized Data" nas Developer Policies).
- **Endpoints:** `channels.list` (metadados e inscritos), `playlistItems.list` (playlist de uploads), `videos.list`
  (detalhes e estatísticas), `channelSections.list` (canais em destaque) e `search.list` (só para descobrir canais).
- **Fontes de apoio:** nenhuma em uso. SMTPD e MicroLens-100K, se entrarem, servem só para ensaio, nunca para teste
  (aula 1.3).

## Período

- **Publicação dos Shorts rotulados:** de 14/05/2014 a 07/09/2026. 77% foram publicados depois de 15/10/2024, e
  2,4% antes de julho de 2021, quando os Shorts foram lançados no mundo todo.
- **Coletas:** de 05/10 a 08/10/2026, 6 execuções (5 completas; a de 06/10 às 5h caiu por timeout de rede e foi
  refeita à tarde). Coleta diária às 05h15, agendada no launchd.
- **Cortes:** a base de treino congela em 12/10 (Shorts publicados até 12/09). A última coleta de rótulos é em 19/10
  (publicados até 19/09).

## Volume

11.203 Shorts rotulados de 52 canais, todos validados no esquema mínimo. A base inteira tem 78.353 vídeos, 23.246
deles Shorts; vira rótulo o Short com 30 dias ou mais, até 400 por canal.

Shorts rotulados por setor e faixa de inscritos:

| Setor | até 10 mil | 10–100 mil | 100 mil–1 mi | mais de 1 mi | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Automotivo | 123 | 625 | 3.804 | 1.683 | 6.235 |
| Beleza | 310 | 1.081 | 1.626 | 1.951 | 4.968 |
| **Total** | **433** | **1.706** | **5.430** | **3.634** | **11.203** |

Canais incluídos (70, já com os 18 da busca de 08/10, que entram na coleta de 09/10):

| Setor e tipo | até 10 mil | 10–100 mil | 100 mil–1 mi | mais de 1 mi | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Automotivo, criador | 0 | 6 | 11 | 4 | 21 |
| Automotivo, marca | 2 | 2 | 9 | 1 | 14 |
| Beleza, criador | 0 | 7 | 8 | 5 | 20 |
| Beleza, marca | 2 | 5 | 5 | 3 | 15 |

## Licença e termos

- **Termos:** YouTube API Services Terms of Service e Developer Policies, na versão de 14/09/2026 lida na aula 1.8.
- **Uso:** educacional e interno, sem produto e sem clientes reais consumindo as previsões.
- **Sem redistribuição:** o repositório (privado) tem o código, as planilhas de curadoria e este data card. Base,
  parquets, vídeos e frames ficam fora do git (`.gitignore`).
- **Retenção:** a coleta diária renova as estatísticas de todo vídeo cujo dado mais recente passou de 25 dias e apaga
  os que sumiram da API. Em seguida, uma auditoria confere se algum vídeo passou de 30 dias sem renovação; se passou,
  a execução falha e o Mac avisa.
- **Apagamento:** os dados brutos serão apagados em **30/10/2026**. Desse dia em diante, o `scripts/coletar.sh` não
  coleta e avisa todo dia, até o passo a passo abaixo ser feito.
- **Tensões conhecidas com as políticas (aula 1.8):** a regra dos 30 dias (III.E.4.d) contra os snapshots antigos,
  que são o rótulo; métricas derivadas (III.E.4.h) contra o alvo relativo; agregação de canais de donos diferentes
  (III.E.2); download e raspagem de mídia (III.E.1 e III.E.6) contra o módulo 4, que ainda não começou.

Passo a passo do apagamento, em 30/10:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.previa.coletor.plist
rm ~/Library/LaunchAgents/com.previa.coletor.plist
rm -f data/shorts.duckdb data/*.parquet data/cota*.json && rm -rf data/logs media/
```

## Coleta

- **Curadoria:** `data/canais.csv` lista incluídos e excluídos, com o motivo de cada um. Os critérios foram escritos
  antes de olhar resultados (`docs/PLANO_DA_BASE.md`). Origem dos 70 canais: 33 da lista manual, 19 da lista manual
  com o handle corrigido pela busca e 18 da busca por termo de 08/10 (15 pelo modo vídeo, 3 pelo modo canal).
- **Método:** playlist de uploads (`UU…`) → `playlistItems.list` → `videos.list`, de forma incremental. Snapshots
  diários dos Shorts com até 60 dias e renovação dos vencidos.
- **Regra de Short:** até 60 s se publicado antes de 15/10/2024, até 180 s depois (quando o limite mudou para 3
  minutos), e não horizontal: largura menor ou igual à altura no player, com formato desconhecido contando como
  Short.
- **Teto por canal:** 400 Shorts rotulados mais recentes. 12 canais estão no teto.
- **Versão:** `src/previa/coletor.py` no git. Mudanças relevantes: retry de rede (06/10), preferência por IPv4 e
  busca por termo (08/10), validação e auditoria a cada coleta (08/10).
- **Cota:** de 110 a 140 unidades por dia na coleta diária, de um limite de 10 mil. Picos: 3.805 na primeira coleta
  (05/10) e 8.615 na avaliação de 614 candidatos (08/10).

## Setores e estratificação

Beleza e automotivo, 35 canais cada. Nos Shorts rotulados, o automotivo tem 74% de criadores e a beleza tem 74% de
marcas; os 20 criadores de beleza incluídos em 08/10 devem equilibrar isso a partir da coleta de 09/10. Idioma e
país: canais brasileiros, em português. O país da API é opcional e vem vazio em muitos canais, por isso a conferência
final de país e idioma é humana (critério 2 do plano).

## Alvo e rótulo

- **Rótulo** (`metric_target`): o `likeCount` do primeiro snapshot com 30 dias ou mais de idade.
- **Alvo de treino** (README, aula 1.1): o log da razão entre as curtidas do Short e a mediana das curtidas dos posts
  anteriores do mesmo canal.
- **Idade do rótulo:** mediana de 324 dias, 10% com até 60 dias e 10% com mais de 1.247 dias. **53% dos rótulos
  foram medidos com mais de 300 dias**: a primeira coleta pegou de uma vez o histórico inteiro de cada canal.
- **Curtidas ocultas:** viram `None`, nunca 0, e atingem 0,42% dos Shorts rotulados. Canais com mais da metade das
  curtidas ocultas ficaram fora na curadoria, e o esquema falha se a base passar de 10%.
- **Distribuição:** mediana de 191 curtidas e percentil 90 de 5.148. É a cauda longa da aula 1.6.

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

## Esquema

`src/previa/esquema.py`: `Post` (Pydantic, um registro) e `BaseShorts` (pandera, a tabela de treino). Campos:
`post_id`, `channel_id`, `published_at`, `caption` (título + descrição), `media_ref` (link do Short, nunca arquivo),
`metric_target`, `target_age_days` (≥ 30), `subscribers_at_collection` (fora das features de treino, aula 1.7),
`collected_at` e `sector`. Checagens: ID único, coleta depois da publicação e no máximo 10% sem curtidas. O modo
estrito trava coluna faltando ou sobrando. A validação roda a cada coleta diária e grava `data/base_minima.parquet`.
Primeira validação na base real, em 08/10: 11.203 linhas, nenhum erro.

## Conformidade (checklist da aula 1.8, em 08/10)

| Item | Situação | O que falta e até quando |
| :--- | :--- | :--- |
| `collected_at` e `age_days` em toda linha | ✅ | — |
| Refresh ou exclusão em até 30 dias, agendado | ✅ renovação diária (25 dias) + auditoria diária que falha alto | — |
| Base, vídeos, frames e áudios fora de repositório ou pasta pública | ⚠️ base fora do git, mas `data/canais.csv`, versionado, guarda inscritos e contagens da API | Decidir até 30/10: tirar os números da API do arquivo versionado (e do histórico do git) ou registrar a exceção |
| Nenhum vídeo baixado guardado | ✅ por enquanto: nenhuma mídia baixada | Decidir como obter a mídia (aula 4.2, III.E.1 e III.E.6) antes do multimodal, até 16/10 |
| Chave da API só em variável de ambiente | ✅ `.env` fora do git; `sem_chave` limpa a chave dos logs | — |
| Nada confidencial ou pessoal no Gemini free tier | ✅ nada enviado; o VLM (camada 2) saiu do escopo | Manter se o VLM voltar |
| Comentários não coletados | ✅ só o `commentCount`; nenhum texto, nenhum autor | — |
| Sem reconhecimento facial | ✅ nada implementado | No módulo 4, só presença de rosto |
| Data card com "o que NÃO representa" e vieses | ✅ | — |
| Esquema validado com pandera a cada exportação | ✅ desde 08/10 | — |
| Previsão apresentada como estimativa do modelo | ❌ a tool ainda não existe | No contrato da tool e no README, até 20/10 |
| Plano de exclusão dos dados brutos, com data | ✅ 30/10/2026; a coleta desliga sozinha | Seguir o passo a passo acima |

## Contato e versão

- **Responsável:** Vinicius Castiglia. **Repositório:** `case-galaxies` (privado).
- **v0.1 (08/10/2026):** primeira versão, exercício da aula 1.8.
