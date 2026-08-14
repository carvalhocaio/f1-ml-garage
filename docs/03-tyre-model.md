# 03 — Modelo de Composto via Telemetria (Módulo 2, SVM + kernels)

Classifica o composto do pneu (`soft`/`medium`/`hard`) a partir de
telemetria agregada por volta — velocidade, acelerador, freio, RPM, marcha.
Introduz um tipo de dado novo no projeto (telemetria de alta frequência, não
mais voltas/resultados) e um problema genuinamente multiclasse.

## O join: telemetria → volta

Telemetria vem em amostras (`~alguns Hz`), não uma linha por volta. Preciso
saber a qual volta cada amostra pertence antes de agregar. `laps.py` só
guarda o FIM de cada volta (`session_time_s`) — não o início. Resolvido com
`pd.merge_asof(..., direction="forward", by="driver")`: cada amostra é
associada à primeira volta cujo fim é `>=` o tempo da amostra.
`by="driver"` é essencial — tempos de sessão se sobrepõem entre pilotos
diferentes, sem isso uma amostra poderia grudar na volta de outro piloto.

## Multiclasse: por que `evaluate_multiclass_classifier`, não
`evaluate_classifier`

`precision`/`recall`/`f1` binários (usados em `dnf.py`) não fazem sentido
com 3 classes. A versão multiclasse usa `average="macro"` — cada composto
pesa igual no resultado, independente de quantas voltas tem. Confirmado na
prática (ver abaixo): "medium" sumiu inteiramente no Bahrain 2024, e
`"weighted"` esconderia esse tipo de desbalanceamento quase tão bem quanto
accuracy sozinha esconde nos modelos binários — `"macro"` não deixa.

## Por que StandardScaler não é opcional aqui

Diferente da árvore/regressão logística (`dnf.py`, features majoritariamente
dummies 0/1 + uma contínua), aqui TODAS as features são contínuas e em
escalas bem diferentes (`mean_rpm` ~11000, `brake_fraction` ~0.2). SVM
mede distância euclidiana pra definir a margem — sem escalar, RPM
dominaria a distância sozinho, não por ter mais sinal, só por ter número
maior.

## Resultado real (Bahrain 2024)

```
composto: hard=791, soft=338 (medium ausente — mesmo achado do modelo de
DNF, agora confirmado de um ângulo totalmente diferente: telemetria, não
resultado de corrida)

accuracy=0.666 precision_macro=0.642 recall_macro=0.640 f1_macro=0.623
```

> **Nota:** esses números são de antes de `filter_to_clean_laps` (filtro de
> bandeira verde) entrar no pipeline de telemetria — ver
> `docs/04-driving-style-clustering.md`, iteração 2. Com o filtro, a
> accuracy caiu ligeiramente pra 0.653 (precision/recall/f1 macro não
> foram remedidos formalmente); o efeito foi pequeno, não invalida a
> leitura abaixo.

Accuracy sozinha é enganosa de novo: com 791 vs 338, só prever "hard"
sempre já dá ~70% — nosso 66.6% é MENOR que esse baseline ingênuo, porque
`class_weight="balanced"` troca accuracy bruta por desempenho mais
equilibrado entre as duas classes (mesmo trade-off do DNF). F1 macro
(0.62) é a leitura mais honesta: bem acima de um chute aleatório
balanceado (0.5), longe de perfeito — seis estatísticas agregadas simples
capturam parte real do estilo de condução por composto, não tudo.

## Próximos passos possíveis

- Mais features por volta: variância de velocidade (não só média/máximo),
  tempo em zona de frenagem forte, uso de DRS, posição em curva (X/Y) —
  Módulo 5 (feature engineering) é onde isso cabe formalmente.
- Comparar `kernel="linear"` vs `"rbf"` — se o desempenho for parecido, a
  fronteira entre compostos é aproximadamente linear nesse espaço de
  features, sem precisar da flexibilidade extra do RBF.
- Combinar várias corridas (como `load_season_results` fez pro DNF) — mais
  amostra, e mais chance de ter "medium" representado de verdade.
