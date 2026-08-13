import pandas as pd
import pytest

from f1_ml_garage.features.pace import (
    build_pace_features,
    compute_driver_delta_target,
    select_green_flag_laps,
)


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


@pytest.mark.unit
def test_delta_target_is_zero_for_driver_with_constant_pace():
    laps = _laps(lap_time_s=[90.0, 90.0, 95.0, 95.0])
    delta = compute_driver_delta_target(laps)
    assert (delta == 0.0).all()


@pytest.mark.unit
def test_delta_target_centers_each_driver_independently():
    """VER e HAM têm baseline de ritmo bem diferente (90s vs 95s), mas o
    mesmo padrão relativo dentro do próprio stint (-0.5s/+0.5s). O delta
    tem que zerar essa diferença de baseline e só sobrar o padrão relativo,
    igual pros dois pilotos."""
    laps = _laps(lap_time_s=[89.5, 90.5, 94.5, 95.5])
    delta = compute_driver_delta_target(laps)
    assert list(delta) == pytest.approx([-0.5, 0.5, -0.5, 0.5])


@pytest.mark.unit
def test_delta_target_does_not_leak_across_drivers():
    """Mudar só o ritmo médio do HAM não pode afetar o delta do VER."""
    laps_a = _laps(lap_time_s=[89.5, 90.5, 94.5, 95.5])
    laps_b = _laps(lap_time_s=[89.5, 90.5, 120.5, 121.5])

    delta_a = compute_driver_delta_target(laps_a)
    delta_b = compute_driver_delta_target(laps_b)

    assert list(delta_a[:2]) == pytest.approx(list(delta_b[:2]))
