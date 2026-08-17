import pandas as pd
import pytest

from f1_ml_garage.features.dnf import (
    build_dnf_features,
    build_retirement_target,
    compute_team_reliability_feature,
    fill_missing_team_reliability,
    select_race_starters,
)


def _results(**overrides: object) -> pd.DataFrame:
    """5 linhas no schema de saída de `normalize_results`, uma pra cada
    status real observado (`Finished`, `Lapped`, `Retired`, `Did not
    start`, `Disqualified`). Colunas de resultado (position, points,
    status...) presentes de propósito — o teste de exclusão de vazamento
    depende delas estarem aqui.
    """
    base = pd.DataFrame(
        {
            "driver": ["VER", "HAM", "LEC", "RUS", "NOR"],
            "team": [
                "Red Bull Racing",
                "Mercedes",
                "Ferrari",
                "Mercedes",
                "McLaren",
            ],
            "grid_position": [1.0, 3.0, 2.0, 5.0, 4.0],
            "dnf": [False, False, True, True, True],
            "position": [1.0, 2.0, float("nan"), float("nan"), 3.0],
            "classified_position": ["1", "2", "R", "E", "D"],
            "points": [25.0, 18.0, 0.0, 0.0, 15.0],
            "race_time_s": [5400.0, 5420.0, float("nan"), float("nan"), 5460.0],
            "laps_completed": [57.0, 57.0, 30.0, 0.0, 57.0],
            "status": [
                "Finished",
                "Lapped",
                "Retired",
                "Did not start",
                "Disqualified",
            ],
        }
    )
    return base.assign(**overrides)


@pytest.mark.unit
def test_build_dnf_features_one_hot_encodes_team():
    features, _, _ = build_dnf_features(_results())
    assert {
        "team_Red Bull Racing",
        "team_Mercedes",
        "team_Ferrari",
        "team_McLaren",
    }.issubset(features.columns)


@pytest.mark.unit
def test_build_dnf_features_includes_grid_position():
    results = _results()
    features, _, _ = build_dnf_features(results)
    assert list(features["grid_position"]) == list(results["grid_position"])


@pytest.mark.unit
def test_build_dnf_features_target_matches_dnf_column():
    results = _results()
    _, target, _ = build_dnf_features(results)
    assert list(target) == list(results["dnf"])


@pytest.mark.unit
def test_build_dnf_features_groups_match_driver_column():
    results = _results()
    _, _, groups = build_dnf_features(results)
    assert list(groups) == list(results["driver"])


@pytest.mark.unit
def test_build_dnf_features_excludes_outcome_columns():
    """Nenhuma coluna derivada do RESULTADO da corrida pode vazar pro X —
    são todas, por definição, posteriores à largada."""
    features, _, _ = build_dnf_features(_results())
    leaking_columns = {
        "position",
        "classified_position",
        "points",
        "race_time_s",
        "laps_completed",
        "status",
        "dnf",
    }
    assert not leaking_columns & set(features.columns)


@pytest.mark.unit
def test_select_race_starters_drops_did_not_start():
    results = _results()
    starters = select_race_starters(results)
    assert "Did not start" not in starters["status"].tolist()
    assert len(starters) == 4


@pytest.mark.unit
def test_build_retirement_target_true_only_for_retired():
    results = _results()
    retired = build_retirement_target(results)

    assert list(retired) == [False, False, True, False, False]


@pytest.mark.unit
def test_build_dnf_features_with_retirement_target_column():
    results = select_race_starters(_results())
    results = results.assign(retired=build_retirement_target(results))

    _, target, _ = build_dnf_features(results, target_column="retired")

    assert list(target) == list(results["retired"])


def _multi_round_results(**overrides: object) -> pd.DataFrame:
    """8 linhas: 2 equipes x 2 pilotos x 2 rodadas, formato de saída de
    `load_season_results` (com `round_number`)."""
    base = pd.DataFrame(
        {
            "driver": ["A1", "A2", "B1", "B2", "A1", "A2", "B1", "B2"],
            "team": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "round_number": [1, 1, 1, 1, 2, 2, 2, 2],
            "grid_position": [1.0, 5.0, 3.0, 8.0, 2.0, 6.0, 4.0, 9.0],
            "dnf": [True, False, False, False, False, True, False, False],
        }
    )
    return base.assign(**overrides)


@pytest.mark.unit
def test_team_reliability_first_round_has_no_history():
    results = _multi_round_results()
    reliability = compute_team_reliability_feature(results)

    assert reliability.iloc[0:4].isna().all()


@pytest.mark.unit
def test_team_reliability_does_not_leak_within_same_round():
    """Bug real encontrado e corrigido: dois pilotos da mesma equipe na
    MESMA rodada correm ao mesmo tempo — um não pode "ver" o resultado do
    outro como se fosse histórico anterior. Os dois têm que sair com o
    MESMO valor (agregado da rodada anterior), não um vazando pro outro
    dentro da mesma rodada."""
    results = _multi_round_results()
    reliability = compute_team_reliability_feature(results)

    # linhas 4/5 são os 2 pilotos da equipe A na rodada 2
    assert reliability.iloc[4] == pytest.approx(reliability.iloc[5])
    # equipe A: 1 DNF em 2 pilotos na rodada 1 -> 0.5
    assert reliability.iloc[4] == pytest.approx(0.5)


@pytest.mark.unit
def test_team_reliability_uses_only_strictly_prior_rounds():
    """3 rodadas pra equipe A: rodada 1 com 2 DNF em 2 (taxa 1.0), rodada 2
    sem histórico de rodada 3 nenhuma (ainda não aconteceu) — a rodada 3
    tem que usar só rodadas 1+2, nunca "ver" a si mesma."""
    results = pd.DataFrame(
        {
            "driver": ["A1", "A2", "A1", "A2", "A1", "A2"],
            "team": ["A", "A", "A", "A", "A", "A"],
            "round_number": [1, 1, 2, 2, 3, 3],
            "grid_position": [1.0, 5.0, 2.0, 6.0, 3.0, 7.0],
            "dnf": [True, True, False, False, False, False],
        }
    )
    reliability = compute_team_reliability_feature(results)

    assert reliability.iloc[2] == pytest.approx(1.0)  # rodada 2: só viu rodada 1
    # rodada 3: viu rodadas 1 (2 DNF/2) e 2 (0 DNF/2) -> 2/4 = 0.5
    assert reliability.iloc[4] == pytest.approx(0.5)


@pytest.mark.unit
def test_fill_missing_team_reliability_uses_mean_of_known_values():
    reliability = pd.Series([float("nan"), float("nan"), 0.5, 0.0])
    filled = fill_missing_team_reliability(reliability)

    assert not filled.isna().any()
    assert filled.iloc[0] == pytest.approx(0.25)
    assert filled.iloc[1] == pytest.approx(0.25)
    # valores já conhecidos não mudam
    assert filled.iloc[2] == pytest.approx(0.5)
    assert filled.iloc[3] == pytest.approx(0.0)


@pytest.mark.unit
def test_build_dnf_features_with_team_reliability_adds_column_no_nan():
    results = _multi_round_results()
    features, _, _ = build_dnf_features(results, include_team_reliability=True)

    assert "team_reliability" in features.columns
    assert not features["team_reliability"].isna().any()


@pytest.mark.unit
def test_build_dnf_features_without_team_reliability_is_unchanged():
    """Comportamento padrão (opt-in `False`) continua igual ao de antes —
    compatibilidade com quem já chama a função sem esse argumento."""
    results = _multi_round_results()
    features, _, _ = build_dnf_features(results)

    assert "team_reliability" not in features.columns
