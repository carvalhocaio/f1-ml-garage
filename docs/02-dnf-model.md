# 02 — Modelo de DNF (Módulo 2, aprendizado supervisionado)

Árvore de decisão prevendo se um piloto abandona a corrida, a partir só do
que se sabe ANTES da largada: posição de grid e equipe. Cruza dois tópicos
do currículo — dados desbalanceados e bias-variance — não só "árvore de
decisão".

## Por que várias corridas, não uma só

Uma corrida tem ~20 pilotos e poucos DNFs — amostra pequena demais pra
treinar ou avaliar qualquer coisa. `load_season_results` carrega e
concatena várias corridas (uma temporada inteira, ou um subconjunto de
rodadas), uma linha por piloto por corrida.

## Vazamento de alvo: por que só `grid_position` + `team`

`results.py` guarda várias colunas — `position`, `classified_position`,
`points`, `race_time_s`, `laps_completed`, `status` — mas todas são
RESULTADO da corrida, não algo conhecido antes da largada. Usar
`laps_completed` pra prever `dnf`, por exemplo, é quase usar a resposta
como pergunta (`laps_completed` baixo é praticamente a definição de DNF).
As únicas colunas seguras de `results.py` são `grid_position` e `team`.

## Bug 1 — a regex "+N Lap" nunca existiu na versão instalada

Primeira versão de `_is_dnf` replicava o critério do
`fastf1.core.DriverResult.dnf`, usando uma regex pra casar o formato
"+N Lap(s)" que a documentação oficial do FastF1 lista como exemplo de
`Status`. Contra a temporada 2024 real: **taxa de "DNF" de ~40%** —
implausível (o valor real de uma temporada de F1 fica em 10-20%).

Diagnóstico, em duas rodadas:

1. Primeira hipótese: a regex exigia dígito colado no `+` (`"+1 Lap"`), e a
   documentação mostra o formato com espaço (`"+ 1 Lap"`). Corrigi pra
   aceitar os dois formatos — e o resultado saiu **idêntico**, bit a bit,
   ao valor buggy. Isso por si só já era o sinal de que a hipótese estava
   errada (queimei um ciclo de debug com isso antes de perceber).
2. `results["status"].value_counts()` revelou a causa real: a versão
   instalada do FastF1 (v3.8.3) usa strings categóricas simples —
   `"Finished"`, `"Lapped"`, `"Retired"`, `"Did not start"`,
   `"Disqualified"` — e **nunca** usa o formato "+N Lap" que a
   documentação lista como exemplo. `138 (Lapped) + 49 (Retired) + 3 (DNS)
   + 2 (DSQ) = 192`; `192 / 479 = 0.4008` — bate exatamente com os ~40%
   observados. Todo piloto "Lapped" (terminou, só que voltas atrás — a
   maioria do grid, na prática) virava DNF por engano.

**Correção:** trocar a regex por uma allowlist direta contra os status
reais (`_FINISHED_STATUSES = frozenset({"Finished", "Lapped"})`), allowlist
de propósito — um status novo e desconhecido conta como DNF por padrão
(mais seguro que assumir "terminou" sem saber). Taxa de DNF corrigida:
**11.27%** — na faixa esperada.

**Lição prática:** a documentação de uma biblioteca é um exemplo, não uma
garantia do formato exato usado pela versão instalada. `value_counts()`
contra dado real teria resolvido isso de primeira, mais rápido que
qualquer hipótese sobre regex.

## Dados desbalanceados: `class_weight` em dado real

Com o alvo corrigido, comparei `class_weight="balanced"` contra sem peso
nenhum, no mesmo dataset (24 corridas, 2024):

```
unweighted: accuracy=0.879 precision=0.000 recall=0.000 f1=0.000
balanced: accuracy=0.553 precision=0.175 recall=0.724 f1=0.268
```

`unweighted` nunca prevê DNF nenhuma vez, em nenhum fold — accuracy alta
(88.7%, quase exatamente `1 - taxa_de_dnf`) escondendo um modelo que não
detecta abandono nenhum. `balanced` sacrifica bastante accuracy por recall
real. Essa é a demonstração mais limpa possível — em dado real, não
sintético — de por que accuracy sozinha engana com classes raras.

## Experimento (não confirmado): alvo restrito a "Retired"

Hipótese: talvez `dnf` (que mistura Retired + DNS + Disqualified — causas
bem diferentes) pudesse virar um alvo mais previsível se restrito só a
abandono em pista, excluindo DNS (nunca correu) e DSQ (tipicamente correu
a prova inteira, excluído depois por infração técnica).

Implementado como `select_race_starters` (remove DNS) +
`build_retirement_target` (`True` só pra `"Retired"`), reusando
`build_dnf_features` via um parâmetro `target_column`.

Resultado, mesmo dataset, `class_weight="balanced"`:

```
dnf (amplo): recall=0.724 precision=0.175 f1=0.268
retired (restrito): recall=0.514 precision=0.113 f1=0.184
```

**A hipótese não se confirmou** — todas as métricas de detecção pioraram
com o alvo mais restrito, não melhoraram. Só 5 linhas foram removidas (3
DNS + 2 DSQ) de 479, o que é pouco demais pra concluir que a causa é a
composição do alvo em si; o mais provável é instabilidade de amostra
pequena (poucos positivos espalhados em `StratifiedGroupKFold`, então
remover só 5 exemplos já reorganiza os folds o suficiente pra balançar a
métrica). Não dá pra separar as duas explicações com uma temporada só.

Mantive o código (`select_race_starters`, `build_retirement_target`,
testado) — não é retrabalho perdido, é ferramenta pronta pra reexaminar
essa pergunta com mais dado (múltiplas temporadas), onde haveria poder
estatístico real pra decidir.

## Árvore vs. regressão logística — mesmo problema, dois modelos

Refatorei a avaliação (`StratifiedGroupKFold` + métricas) pra um módulo
compartilhado (`models/evaluation.py`) — a lógica de CV não depende de qual
classificador está por trás, e duplicá-la pra cada modelo novo violaria
DRY. `build_dnf_tree_pipeline` e `build_dnf_logistic_pipeline` (ambas em
`models/dnf.py`) resolvem o mesmo problema (`features/dnf.py` não muda).

A regressão logística tem a mesma armadilha de colinearidade que o modelo
de ritmo já tinha exposto (`docs/01-pace-model.md`): `team` como one-hot
completo soma sempre 1, colinear com o intercepto. Desta vez apliquei a
correção (`fit_intercept=False`, todas as equipes com coeficiente próprio)
de propósito, não depois de descobrir o bug — o padrão já estava validado.

Comparando os dois no dataset completo da temporada 2024 (alvo `dnf`
original, `class_weight="balanced"` nos dois):

```
árvore: accuracy=0.553 precision=0.175 recall=0.724 f1=0.268
logística: accuracy=0.650 precision=0.189 recall=0.574 f1=0.273
```

F1 empata na prática — nenhum dos dois vence objetivamente com os
hiperparâmetros padrão (`max_depth=4`, `C=1.0`, nenhum ajustado ainda). A
árvore favorece recall (detecta mais DNF real, à custa de mais alarme
falso); a logística fica num ponto mais equilibrado do trade-off
precision/recall. Qual é "melhor" depende do custo relativo de cada erro
pro caso de uso, não é uma resposta absoluta.

## Ensembles: Random Forest e XGBoost — mais sofisticado não é melhor aqui

Adicionados `build_dnf_random_forest_pipeline` (bagging) e
`build_dnf_boosting_pipeline` (boosting via XGBoost — pacote
`xgboost-cpu`, não `xgboost` puro, que puxa ~300MB de dependência CUDA
irrelevante pra esse projeto). XGBoost usa `scale_pos_weight` pra classe
rara, não `class_weight` — não recalcula sozinho a partir do `y` de
treino como o sklearn faz, precisa ser calculado explicitamente
(`compute_scale_pos_weight`, proporção negativos/positivos).

Resultado real (temporada 2024, mesmas 2 features de sempre, limiar
padrão de 0.5):

```
árvore: accuracy=0.553 precision=0.175 recall=0.724 f1=0.268
logística: accuracy=0.650 precision=0.189 recall=0.574 f1=0.273
random forest: accuracy=0.711 precision=0.179 recall=0.422 f1=0.243
xgboost: accuracy=0.799 precision=0.246 recall=0.236 f1=0.206
```

Direção **oposta** à intuição de "modelo mais sofisticado, resultado
melhor": accuracy e precision sobem de árvore pra XGBoost, mas recall
despenca e F1 piora — XGBoost tem o PIOR F1 dos quatro, apesar da melhor
accuracy.

**Leitura inicial:** ensembles otimizam pra minimizar erro médio de
treino. Com só 2 features — pouca estrutura complexa pra explorar — e uma
classe rara, o jeito mais barato de reduzir erro médio é inclinar pra
classe majoritária. `scale_pos_weight`/`class_weight="balanced"`
reponderam o TREINO, mas a decisão final ainda usa o limiar padrão de 0.5
pra converter probabilidade em classe.

### Ajustando o limiar: a hipótese só se confirma em parte

Implementado `find_best_threshold` + `evaluate_classifier_with_tuned_threshold`
(`models/evaluation.py`) — escolhe o limiar que maximiza F1 via curva
precision-recall, usando probabilidades fora-da-dobra (`cross_val_predict`),
em vez do corte fixo de 0.5.

Resultado real, mesmos 4 modelos, limiar ajustado:

```
               limiar    F1 (0.5)   F1 (ajustado)
árvore:        0.379     0.268      0.276
logística:     0.496     0.273      0.290
random forest: 0.472     0.243      0.264
xgboost:       0.008     0.206      0.226
```

Ajustar o limiar melhorou os 4 modelos — mas **não mudou o ranking**.
XGBoost continua sendo o pior em F1, mesmo com o limiar otimizado (que
precisou cair pra 0.008 — "responde positivo pra quase tudo", accuracy
despenca pra 0.355 nesse ponto). A logística passa a ser a melhor das
quatro.

**Leitura final, mais honesta que a inicial:** o problema não era só o
limiar fixo em 0.5 — ajustar ajudou todo mundo igual, sem mudar a ordem.
Explicação mais provável: com só ~9 features (`grid_position` + ~8
dummies de `team`) e ~479 linhas, 200 árvores de boosting é capacidade
demais pra pouco dado/sinal real — o próprio XGBoost provavelmente tem
mais variância (overfit no CV) que a árvore única ou a logística. Mesmo
tema de bias-variance do currículo, de outro ângulo: modelo mais
sofisticado nem sempre vence quando o dado é pequeno e o sinal é
limitado — às vezes o modelo simples generaliza melhor por ter menos o
que decorar.

## Próximos passos possíveis

- Combinar múltiplas temporadas (2022-2024) — mais amostra, poder
  estatístico real pra revisitar o experimento do alvo restrito, e talvez
  dar ao XGBoost dado suficiente pra justificar sua capacidade extra.
- Features adicionais conhecidas antes da largada: histórico de
  confiabilidade da equipe (taxa de DNF em corridas anteriores DAQUELA
  temporada, cuidado pra não vazar o futuro), característica do circuito
  (rua vs permanente — Mônaco/Baku têm taxa de incidente bem diferente de
  Silverstone/Barcelona).
- Ajustar hiperparâmetros (`max_depth`, `C`, `n_estimators`) — reduzir a
  capacidade do XGBoost (menos estimadores, árvores mais rasas) pra
  testar diretamente a hipótese de overfit por capacidade excessiva.
- Stacking — combinar os 4 modelos com um meta-modelo, a peça que falta
  do tópico de ensembles do currículo.

O resto do Módulo 2 (SVM+kernels, classificar composto via telemetria)
está completo — ver `docs/03-tyre-model.md`.
