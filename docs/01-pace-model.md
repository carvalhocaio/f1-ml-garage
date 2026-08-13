# 01 — Modelo de Ritmo (Módulo 2, aprendizado supervisionado)

Regressão linear de tempo de volta a partir de composto/idade de pneu e
`lap_number` (proxy de combustível/evolução de pista). O objetivo declarado
desde o início (ver `models/pace.py`) não é maximizar acurácia — é entender
o efeito de cada variável no ritmo, e fazer a avaliação corretamente.

## Iteração 1 — alvo bruto: R² ≈ 0.02

Primeira tentativa: prever `lap_time_s` diretamente a partir de
`tyre_life` + `compound`, avaliado com `GroupKFold` agrupado por piloto
(ver `evaluate_pace_model`). Contra o Bahrain 2024 real: **R² ≈ 0.024**,
MAE ≈ 1.01s.

Diagnóstico: a diferença de ritmo *entre carros/pilotos* (facilmente
1-2s/volta em 2024) domina a variância de `lap_time_s` muito mais do que o
efeito de composto/idade de pneu (frações de segundo). Como o modelo nunca
vê "quem" está pilotando, essa diferença de baseline vira ruído
não-modelado que afoga o sinal real que queríamos medir.

## Iteração 2 — alvo delta por piloto: R² ≈ 0.08

Troca de alvo: `lap_time_s - média(lap_time_s) do próprio piloto na sessão`
(`compute_driver_delta_target`). Remove a diferença de baseline entre
carros da variável de resposta, sem vazar nada no `GroupKFold` — como o CV
já agrupa por piloto, a média usada pra centralizar nunca mistura dado de
treino com dado de teste.

Resultado: **R² ≈ 0.083**, MAE ≈ 0.89s — mais que triplicou, confirmando o
diagnóstico. Ainda baixo: faltavam dois efeitos sistemáticos grandes
(combustível, evolução de pista).

## Iteração 3 — adiciona `lap_number`: R² ≈ 0.81

`lap_number` como proxy de carga de combustível (carro mais leve e rápido
ao longo da corrida) e evolução de pista (asfalto mais rápido conforme
mais borracha é depositada). Não é colinear com `tyre_life` apesar de
ambos crescerem "com o tempo": `tyre_life` zera a cada pit stop,
`lap_number` não.

Resultado: **R² ≈ 0.81**, MAE ≈ 0.32s. Salto muito maior que o da iteração
2 — sinal de que combustível/evolução de pista pesam mais no ritmo do que
a degradação do pneu isolada, pelo menos em Bahrain 2024.

## Bug 1 — colinearidade por encoding

Com os 3 dummies de composto (`compound_soft`, `compound_medium`,
`compound_hard`) e um modelo com intercepto, a soma das 3 colunas é sempre
1 — igual à coluna implícita do intercepto. `LinearRegression` ainda
resolve isso via mínimos quadrados e a *previsão* sai correta, mas os
coeficientes individuais ficam matematicamente indeterminados (infinitas
combinações intercepto/coeficientes dão a mesma previsão).

Primeira correção (incompleta, ver Bug 2): dropar "medium" como categoria
de referência fixa, deixando o intercepto capturar seu baseline.

## Bug 2 — a correção do Bug 1 quebrou em dado real

Rodando os coeficientes contra Bahrain 2024:

```
tyre_life 0.109675
lap_number -0.070166
compound_soft 0.071774
compound_hard -0.071774
intercept 0.931681
```

`compound_soft` e `compound_hard` saíram **exatamente opostos** — não é
coincidência, é a mesma assinatura de colinearidade perfeita reaparecendo.
Causa: `green["compound"].value_counts()` mostrou **zero voltas de
"medium"** nessa corrida (estratégia soft→hard→hard, sem stint de medium
sob bandeira verde). Dropar uma referência fixa só funciona se ela aparecer
nos dados — se a categoria de referência está ausente, todo `compound_soft
+ compound_hard` volta a somar 1 em toda linha, e a colinearidade que a
referência deveria evitar reaparece, agora causada pelos dados, não pelo
encoding.

**Correção robusta:** manter as 3 dummies sempre presentes e tirar o
intercepto do modelo (`LinearRegression(fit_intercept=False)`, ver
`build_pace_pipeline`). Sem intercepto compartilhado, cada composto carrega
seu próprio coeficiente absoluto — e o design fica com posto completo
mesmo quando um composto está inteiramente ausente: só o coeficiente
*daquele* composto ausente fica sem sentido (retorna 0, o que é esperado —
não dá pra estimar efeito de algo nunca observado), sem contaminar os
demais. Coberto por teste de regressão
(`test_pace_coefficients_stable_when_a_compound_is_absent`) que reproduz
exatamente esse cenário.

## Coeficientes finais (Bahrain 2024, alvo delta)

```
tyre_life 0.109675
lap_number -0.070166
compound_soft 1.003455
compound_medium 0.000000 (ausente nos dados — não interpretável)
compound_hard 0.859907
```

- **Degradação de pneu:** ~0.11s mais lento por volta de idade do pneu.
- **Combustível/evolução de pista:** ~0.07s mais rápido por volta de
  corrida decorrida.
- **Composto:** a diferença `compound_soft - compound_hard` ≈ 0.14s (soft
  mais lento que hard, a tyre_life/lap_number iguais) é a leitura válida —
  os valores absolutos sozinhos correspondem ao modelo extrapolado pra
  `tyre_life=0, lap_number=0`, que nunca ocorre nos dados reais.

Essa diferença é contraintuitiva (esperaríamos soft mais rápido que hard em
pneu de idade equivalente). Hipótese mais provável, não verificada: o
modelo assume uma única taxa de degradação (`tyre_life`) compartilhada
entre todos os compostos — se soft degrada mais rápido que hard de verdade,
forçar uma inclinação única pode empurrar parte desse efeito pro
coeficiente de composto errado. Um termo de interação composto×tyre_life é
o candidato natural pra investigar isso.

## Próximos passos possíveis

- Interação composto×tyre_life (`tyre_life` por composto, não
  compartilhado) — testaria a hipótese acima diretamente.
- Comparar com Ridge/Lasso (regularização) e ver se muda a leitura dos
  coeficientes.
- Rodar contra outras corridas/circuitos e comparar os coeficientes — o
  efeito de combustível deveria ser relativamente estável entre pistas; o
  de composto, não (depende do carro/pneu daquele ano específico).
- Seguir pro resto do Módulo 2: regressão logística (undercut/overcut),
  SVM+kernels (classificar composto pela telemetria), árvores de decisão
  (DNF).
