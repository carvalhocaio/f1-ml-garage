"""Features para o classificador de DNF (Módulo 2, árvore de decisão).

O ponto mais importante deste módulo não é a árvore — é quais colunas de
`results.py` são seguras de usar. `position`, `classified_position`,
`points`, `race_time_s`, `laps_completed` e `status` são todos RESULTADO da
corrida — usá-los pra prever `dnf` é vazamento de alvo (ex.: `laps_completed`
baixo é quase a própria definição de DNF, não uma feature preditiva dele).
As únicas colunas de `results.py` conhecidas ANTES da corrida largar são
`grid_position` e `team`.
"""

import pandas as pd


def select_race_starters(results: pd.DataFrame) -> pd.DataFrame:
    """Remove pilotos que nunca largaram a corrida ("Did not start").

    Prever "abandono em pista" não faz sentido pra quem nunca correu — as
    causas de DNS (doença, dano de uma sessão anterior) não têm relação
    com grid/equipe do jeito que abandono durante a corrida tem. Usado
    antes de `build_retirement_target`, não antes do `dnf` genérico (que
    continua incluindo DNS de propósito — ver `results.py`).
    """
    return results.loc[results["status"] != "Did not start"].reset_index(drop=True)


def build_retirement_target(results: pd.DataFrame) -> pd.Series:
    """Alvo mais restrito que `dnf`: True só pra abandono em pista
    ("Retired"), excluindo desclassificação pós-corrida ("Disqualified").

    Um DSQ tipicamente correu a prova inteira e só foi excluído depois por
    infração técnica — uma causa desconectada de grid/equipe, diferente de
    um abandono por acidente ou falha mecânica. Misturar os dois no mesmo
    alvo empurra o modelo a aprender um padrão que não existe pro DSQ,
    diluindo o sinal real. Este alvo é mais homogêneo (só causas
    relacionadas a carro/pista/incidente), ao custo de um dataset com
    fronteira de classe um pouco diferente do `dnf` genérico.

    Espera `results` já filtrado por `select_race_starters` — não filtra
    de novo aqui, mesma separação de responsabilidades de
    `features/pace.py`.
    """
    return results["status"] == "Retired"


def build_dnf_features(
    results: pd.DataFrame, target_column: str = "dnf"
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

    `target_column`: "dnf" (padrão, definição ampla) ou uma coluna que o
    chamador já adicionou a `results` — ex. `build_retirement_target` — pra
    testar um alvo mais restrito sem duplicar a lógica de X/groups.
    """
    team_dummies = pd.get_dummies(results["team"], prefix="team")

    features = pd.concat(
        [results[["grid_position"]].reset_index(drop=True), team_dummies],
        axis=1,
    )
    target = results[target_column].reset_index(drop=True)
    groups = results["driver"].reset_index(drop=True)

    return features, target, groups
