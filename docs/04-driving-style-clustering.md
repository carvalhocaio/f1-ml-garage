# 04 — Clustering de Estilo de Pilotagem (Módulo 4, não-supervisionado)

PCA + k-means sobre telemetria agregada por volta, tentando descobrir
estrutura sem alvo — diferente de todo o Módulo 2 (supervisionado), aqui
não existe "certo"/"errado" a priori. O fio condutor da investigação foi
"os clusters formados batem com alguma coisa que já conhecemos
(composto)?", usada como checagem de sanidade, não como alvo de treino.

## Iteração 1 — telemetria bruta, sem filtro

Primeira tentativa: `tag_telemetry_with_lap` + `summarize_lap_telemetry`
direto, sem nenhum filtro de qualidade. Resultado real (Bahrain 2024, 1129
voltas): **silhouette 0.389**, nenhum cluster com composição de composto
muito diferente da média geral (todos entre 60-80% hard, a mesma faixa da
proporção geral da corrida).

## Iteração 2 — filtro de bandeira verde

Hipótese: telemetria de entrada/saída de pit e safety car (nunca filtrada
antes de chegar aqui) estaria contaminando os clusters com uma dimensão
que não tem nada a ver com composto/estilo. Implementado
`filter_to_clean_laps` (`features/telemetry_summary.py`) — reusa
`select_green_flag_laps` (movida de `features/pace.py` pra `data/laps.py`,
já que deixou de ser específica de modelagem de ritmo).

Ponto técnico que valia registrar: o filtro tem que ser aplicado DEPOIS do
`merge_asof` que associa telemetria a volta, nunca antes — filtrar `laps`
antes faz uma amostra que pertencia a uma volta removida (ex.: sob safety
car) ser incorretamente reatribuída à PRÓXIMA volta que sobrou na tabela
(`merge_asof direction="forward"` não sabe que aquela volta foi removida
de propósito). Confirmado com um exemplo mínimo antes de implementar.

Resultado: 1129 → 1006 voltas, **silhouette 0.399** (praticamente igual),
SVM de composto caiu ligeiramente (0.666 → 0.653 accuracy). **Hipótese não
confirmada** — a contaminação de pit/safety car não era a causa principal.

## Iteração 3 — features relativas ao piloto

Segunda hipótese, por analogia direta com o que funcionou no modelo de
ritmo (`compute_driver_delta_target`, `docs/01-pace-model.md`): valores
absolutos de telemetria (velocidade média, RPM médio...) são dominados por
qual carro é mais rápido no geral (Red Bull vs. Williams), não por
composto ou estilo. `build_driving_style_features(..., relative_to_driver=True)`
centraliza cada feature pela própria média do piloto na sessão.

Resultado: silhouette praticamente igual (0.399 → 0.401) — a separação
GEOMÉTRICA dos clusters não mudou muito. Mas a composição de cada cluster
sim:

```
            soft  hard  total  % soft
cluster 0   111   76    187     59.4%
cluster 1   81    315   396     20.5%
cluster 2   85    338   423     20.1%
```

Taxa geral de soft na corrida: ~27.5%. **Cluster 0 concentra o dobro
disso** (59.4%) — sinal real de composto que os valores absolutos
mascaravam. Mas clusters 1 e 2 são quase idênticos entre si em composição
(20.5% vs 20.1%) — o k-means separou "soft-ish" do resto, e dividiu o
resto (majoritariamente hard) em dois grupos que não se diferenciam por
composto nenhum.

**Leitura:** hipótese parcialmente confirmada. Silhouette não é a métrica
que capturaria essa mudança — ela mede compactação/separação geométrica,
não alinhamento com uma variável externa; por isso ficou estável enquanto
a composição dos clusters mudou de verdade. O padrão observado (dois
clusters quase idênticos em composto) sugere que `k=3` pode não ser o
número certo pra essa estrutura, ou que o que separa clusters 1/2 é outra
coisa — combustível/evolução de pista de novo (o mesmo fator que dominou
o modelo de ritmo antes de controlarmos por ele), não estilo de pilotagem.

## Próximos passos possíveis

- Testar `k=2` — a estrutura observada (soft concentrado vs. resto
  homogêneo) pode caber melhor em 2 clusters que 3.
- Investigar o que separa clusters 1/2: cruzar contra `lap_number` (proxy
  de combustível/evolução de pista, já validado como sinal forte no
  modelo de ritmo) em vez de só `compound`.
- Método do cotovelo ou silhouette por vários `k` — escolher `n_clusters`
  sistematicamente em vez de fixar 3 de antemão.
- DBSCAN como alternativa ao k-means — não assume clusters esféricos nem
  exige definir `k` de antemão, pode revelar estrutura diferente.
- GMM/EM — em vez de atribuição rígida a um cluster, probabilidade de
  pertencer a cada grupo; pode capturar melhor voltas "de transição"
  (ex.: pneu perto do fim da vida, começando a se comportar como outro
  composto).
