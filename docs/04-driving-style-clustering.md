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

## Iteração 4 — GMM/EM decide `k` objetivamente, não eu

Até aqui, `n_clusters=3` foi uma escolha arbitrária (o padrão do módulo,
nunca justificado). Implementado `fit_gmm` + `select_n_components_by_bic`
(BIC — Bayesian Information Criterion, penaliza complexidade contra
qualidade de ajuste) pra decidir `k` objetivamente em vez de fixar de
antemão.

Resultado real (mesma telemetria filtrada + relativa ao piloto da
iteração 3): BIC cai monoticamente de `k=2` até `k=6` testado — **nunca
melhora depois de 2**. Confirma a suspeita registrada na iteração 3: os
clusters 1/2 do `k=3` eram uma divisão artificial de uma estrutura que só
tem 2 grupos de verdade.

Com `k=2`:


``` 
          soft  hard  total  % soft
cluster 0 139   151   290     47.9%
cluster 1 138   578   716     19.3%
```

Taxa geral: 27.5%. Contraste bem mais nítido que o `k=3` anterior (onde
dois dos três clusters eram quase idênticos): cluster 0 fica perto de
meio-a-meio (quase o dobro da taxa geral de soft), cluster 1 bem mais
hard-dominante que a média.

Confiança média do GMM (probabilidade do componente mais provável) ficou
em **0.828** — boa, mas longe do >0.95 que o dataset sintético (separável
por construção) atinge nos testes. Leitura honesta: existe uma estrutura
real de 2 grupos nos dados, razoavelmente clara, mas com ambiguidade
genuína na fronteira entre eles — esperado em dado de corrida real, e uma
leitura mais confiável do que "encontrar" 3 clusters bem definidos que na
prática eram 1 real + 1 dividido ao meio sem motivo.

## Próximos passos possíveis

- Investigar diretamente o que caracteriza cluster 0 vs 1 (não só
  composto) — olhar `lap_number`/`tyre_life` médios de cada grupo, testar
  se bate com a hipótese de combustível/evolução de pista.
- Repetir esse fluxo (BIC → k → GMM) noutra corrida e comparar se a
  estrutura de 2 grupos se mantém, ou se é específica do Bahrain.
- t-SNE/UMAP pra visualizar a fronteira entre os dois componentes — os
  ~17% de confiança "perdida" (0.828 vs ~1.0) provavelmente vêm de um
  conjunto específico de voltas na fronteira; visualizar ajudaria a ver
  quais.
- DBSCAN como comparação — não assume forma gaussiana nem número fixo de
  clusters, pode confirmar (ou contestar) a estrutura de 2 grupos por um
  caminho totalmente diferente.
