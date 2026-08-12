import pandas as pd
import pytest

from f1_ml_garage.features.pace import build_pace_features, select_green_flag_laps


def _laps(**overrides: object) -> pd.DataFrame:
    """4 voltas normalizadas (schema de saída de `normalize_laps`), com
    variação em confiabilidade e status de pista.
    """
    base = pd.DataFrame(
        {
            "driver": ["VER", "VER", "HAM", "HAM"],
            "lap_time_s": [88.5, 89.0, 90.1, 90.5],
            "compound": ["soft", "soft", "medium", "medium"],
            "tyre_life": [1.0, 2.0, 1.0, 2.0],
            "track_status": ["1", "1", "1", "1"],
            "is_accurate": [True, True, True, True],
            "deleted": [False, False, False, False],
        }
    )
    return base.assign(**overrides)


@pytest.mark.unit
def test_select_green_flag_laps_drops_non_green_status():
    raw = _laps(track_status=["1", "2", "1", "1"])
    green = select_green_flag_laps(raw)
    assert len(green) == 3


@pytest.mark.unit
def test_select_green_flag_laps_drops_inaccurate_laps():
    raw = _laps(is_accurate=[True, True, False, True])
    green = select_green_flag_laps(raw)
    assert len(green) == 3


@pytest.mark.unit
def test_build_pace_features_encodes_all_compound_categories():
    """Mesmo um subconjunto sem voltas de "hard" tem que gerar a coluna
    `compound_hard` (zerada) — o shape de X não pode depender de quais
    compostos aparecem no subconjunto recebido."""
    features, _, _ = build_pace_features(_laps())

    assert {"compound_soft", "compound_medium", "compound_hard"}.issubset(
        features.columns
    )
    assert (features["compound_hard"] == 0).all()


@pytest.mark.unit
def test_build_pace_features_one_hot_is_mutually_exclusive():
    features, _, _ = build_pace_features(_laps())
    dummy_cols = ["compound_soft", "compound_medium", "compound_hard"]
    assert (features[dummy_cols].sum(axis=1) == 1).all()


@pytest.mark.unit
def test_build_pace_features_returns_aligned_lengths():
    laps = _laps()
    features, target, groups = build_pace_features(laps)

    assert len(features) == len(laps)
    assert len(target) == len(laps)
    assert len(groups) == len(laps)


@pytest.mark.unit
def test_build_pace_features_groups_match_driver_column():
    laps = _laps()
    _, _, groups = build_pace_features(laps)
    assert list(groups) == list(laps["driver"])


@pytest.mark.unit
def test_build_pace_features_target_matches_lap_time():
    laps = _laps()
    _, target, _ = build_pace_features(laps)
    assert list(target) == list(laps["lap_time_s"])
