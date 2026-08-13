"""Features para o classificador de DNF (Módulo 2, árvore de decisão).

O ponto mais importante deste módulo não é a árvore - é quais colunas de
`results.py` são seguras de usar. `position`, `classified_position`,
`points`, `race_time_s`, `laps_completed` e `status` são todos RESULTADO da
corrida - usá-los para prever `dnf` é vazamento de alvo (ex.: `laps_completed`
baixo é quase a própria definição de DNF, não uma feature preditiva dele).
As únicas colunas de `results.py` conhecidas ANTES da corrida largar são
`grid_position` e `team`.
"""

import pandas as pd


def build_dnf_features(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Monta a matriz de features (X), o alvo (y) e os grupos (piloto).

    X usa só `grid_position` (numérico) e `team` (one-hot, sem categorias
    fixas — ao contrário de `compound` em `features/pace.py`, o elenco de
    equipes é estável dentro de uma temporada mas muda de ano pra ano, então
    fixar uma lista global seria frágil). Isso não é problema pra árvore de
    decisão: diferente de regressão linear, não há colinearidade com
    intercepto pra se preocupar — a árvore só escolhe cortes, e um one-hot
    "completo" (sem dropar referência) funciona bem.

    `groups` (piloto) existe pro mesmo motivo do modelo de ritmo: evita que
    o mesmo piloto apareça em treino e teste, testando generalização pra
    pilotos não vistos, não memorização de propensão individual a acidente.
    """
    team_dummies = pd.get_dummies(results["team"], prefix="team")

    features = pd.concat(
        [results[["grid_position"]].reset_index(drop=True), team_dummies],
        axis=1,
    )
    target = results["dnf"].reset_index(drop=True)
    groups = results["driver"].reset_index(drop=True)

    return features, target, groups
