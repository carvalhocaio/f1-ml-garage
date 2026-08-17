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


def compute_team_reliability_feature(results: pd.DataFrame) -> pd.Series:
    """Taxa de DNF da equipe nas corridas ANTERIORES da mesma temporada —
    janela expansível, nunca olha a rodada atual nem o futuro.

    Requer `round_number` (marcado por `load_season_results`) e `dnf`.
    Agrega a taxa de DNF por (`team`, `round_number`) PRIMEIRO — cada
    equipe tem 2 pilotos correndo na mesma rodada, ao mesmo tempo; se o
    `shift(1)/expanding()` rodasse direto linha a linha (uma linha por
    piloto), o resultado de um piloto vazaria pro "histórico" do
    companheiro de equipe da MESMA rodada, que corre simultaneamente, não
    antes. Agregar a rodada inteira num só número antes de aplicar a
    janela evita esse vazamento.

    Primeira rodada de cada equipe na temporada vira `NaN` (sem histórico
    anterior) — decisão de preenchimento fica pra quem chama
    (`fill_missing_team_reliability`), não desta função: mistura filtro
    com imputação seria a mesma bagunça de responsabilidades que
    `select_green_flag_laps`/`build_pace_features` evitam em
    `features/pace.py`.
    """
    team_round_dnf_rate = (
        results.groupby(["team", "round_number"])["dnf"].mean().reset_index()
    )
    team_round_dnf_rate = team_round_dnf_rate.sort_values(["team", "round_number"])
    team_round_dnf_rate["team_reliability"] = team_round_dnf_rate.groupby("team")[
        "dnf"
    ].transform(lambda group: group.shift(1).expanding().mean())

    merged = (
        results.reset_index()
        .merge(
            team_round_dnf_rate[["team", "round_number", "team_reliability"]],
            on=["team", "round_number"],
            how="left",
        )
        .set_index("index")
    )
    return merged["team_reliability"].reindex(results.index)


def fill_missing_team_reliability(reliability: pd.Series) -> pd.Series:
    """Preenche times sem histórico ainda (primeira rodada da temporada)
    com a média geral de confiabilidade entre equipes que JÁ têm histórico
    — um prior razoável pra "não sei nada sobre essa equipe ainda".

    Calculado a partir do dataset completo (mesma simplificação já
    documentada em `compute_scale_pos_weight` e
    `evaluate_classifier_with_tuned_threshold`): não é uma estatística por
    exemplo individual, é agregada — vazamento pequeno e consistente com
    o resto do módulo, não escondido.
    """
    return reliability.fillna(reliability.mean())


def build_dnf_features(
    results: pd.DataFrame,
    target_column: str = "dnf",
    *,
    include_team_reliability: bool = False,
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

    `include_team_reliability=True` adiciona `team_reliability`
    (`compute_team_reliability_feature` + `fill_missing_team_reliability`)
    como terceira feature — opt-in, não padrão, pra manter a comparação
    "com" vs "sem" limpa (mesmo espírito de `relative_to_driver` em
    `features/driving_style.py`). Requer `results` vindo de
    `load_season_results` (precisa de `round_number`).
    """
    team_dummies = pd.get_dummies(results["team"], prefix="team")

    feature_frames = [
        results[["grid_position"]].reset_index(drop=True),
        team_dummies,
    ]
    if include_team_reliability:
        reliability = compute_team_reliability_feature(results)
        reliability = fill_missing_team_reliability(reliability)
        feature_frames.append(
            reliability.rename("team_reliability").reset_index(drop=True)
        )

    features = pd.concat(feature_frames, axis=1)
    target = results[target_column].reset_index(drop=True)
    groups = results["driver"].reset_index(drop=True)

    return features, target, groups
