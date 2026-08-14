import pandas as pd
import pytest

from f1_ml_garage.features.driving_style import (
    FEATURE_COLUMNS,
    build_driving_style_features,
)


def _lap_summary() -> pd.DataFrame:
    """Formato de saída de `telemetry_summary.summarize_lap_telemetry`."""
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
def test_build_driving_style_features_returns_only_feature_columns():
    features, _ = build_driving_style_features(_lap_summary())
    assert list(features.columns) == FEATURE_COLUMNS


@pytest.mark.unit
def test_build_driving_style_features_metadata_has_no_target_no_features():
    """Metadados são só pra interpretar depois — não podem vazar de volta
    pras features nem virar alvo (clustering não tem alvo)."""
    _, metadata = build_driving_style_features(_lap_summary())
    assert list(metadata.columns) == ["driver", "lap_number", "compound"]


@pytest.mark.unit
def test_build_driving_style_features_aligned_lengths():
    summary = _lap_summary()
    features, metadata = build_driving_style_features(summary)
    assert len(features) == len(summary)
    assert len(metadata) == len(summary)


@pytest.mark.unit
def test_build_driving_style_features_metadata_matches_source():
    summary = _lap_summary()
    _, metadata = build_driving_style_features(summary)
    assert list(metadata["driver"]) == list(summary["driver"])
    assert list(metadata["compound"]) == list(summary["compound"])
