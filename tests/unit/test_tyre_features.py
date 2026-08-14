import pandas as pd
import pytest

from f1_ml_garage.features.tyre import build_tyre_features


def _lap_summary() -> pd.DataFrame:
    """Formato de saída de `telemetry_summary.summarize_lap_telemetry`,
    construído direto (sem passar pelo join) pra testar só
    `build_tyre_features` isoladamente."""
    return pd.DataFrame(
        {
            "driver": ["VER", "VER", "HAM"],
            "lap_number": [1, 2, 1],
            "compound": ["soft", "soft", "hard"],
            "mean_speed_kmh": [290.0, 292.0, 275.0],
            "max_speed_kmh": [310.0, 312.0, 295.0],
            "mean_throttle_pct": [85.0, 86.0, 70.0],
            "brake_fraction": [0.15, 0.14, 0.22],
            "mean_rpm": [11800.0, 11850.0, 11300.0],
            "mean_gear": [6.5, 6.6, 5.8],
        }
    )


@pytest.mark.unit
def test_build_tyre_features_returns_aligned_shapes():
    summary = _lap_summary()
    features, target, groups = build_tyre_features(summary)

    assert len(features) == len(summary)
    assert len(target) == len(summary)
    assert len(groups) == len(summary)


@pytest.mark.unit
def test_build_tyre_features_target_matches_compound():
    summary = _lap_summary()
    _, target, _ = build_tyre_features(summary)

    assert list(target) == list(summary["compound"])


@pytest.mark.unit
def test_build_tyre_features_groups_match_driver():
    summary = _lap_summary()
    _, _, groups = build_tyre_features(summary)

    assert list(groups) == list(summary["driver"])


@pytest.mark.unit
def test_build_tyre_features_excludes_identifier_columns():
    """driver/lap_number/compound não são features numéricas do modelo —
    só identificam a linha ou são o alvo."""
    summary = _lap_summary()
    features, _, _ = build_tyre_features(summary)

    assert not {"driver", "lap_number", "compound"} & set(features.columns)
