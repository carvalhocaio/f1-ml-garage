import pandas as pd
import pytest

from f1_ml_garage.features.dnf import (
    build_dnf_features,
    build_retirement_target,
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
