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


def _two_driver_summary_with_different_baselines() -> pd.DataFrame:
    """2 pilotos com baselines de velocidade bem diferentes (VER ~297,
    SAR ~252), mas o MESMO padrão relativo dentro do próprio desempenho
    (mais rápido na volta 2 que na 1, por +2.5 km/h nos dois casos)."""
    return pd.DataFrame(
        {
            "driver": ["VER", "VER", "SAR", "SAR"],
            "lap_number": [1, 2, 1, 2],
            "compound": ["soft", "soft", "soft", "soft"],
            "mean_speed_kmh": [295.0, 300.0, 250.0, 255.0],
            "max_speed_kmh": [310.0, 315.0, 265.0, 270.0],
            "mean_throttle_pct": [85.0, 87.0, 80.0, 82.0],
            "brake_fraction": [0.15, 0.14, 0.20, 0.19],
            "mean_rpm": [11800.0, 11900.0, 11000.0, 11100.0],
            "mean_gear": [6.5, 6.6, 6.0, 6.1],
        }
    )


@pytest.mark.unit
def test_relative_to_driver_removes_baseline_difference_between_drivers():
    """Apesar de VER e SAR terem velocidade absoluta bem diferente, o
    padrão relativo (delta dentro do próprio desempenho) tem que sair
    idêntico pros dois — é exatamente o que deveria sobrar depois de
    remover a diferença de baseline entre carros/pilotos."""
    summary = _two_driver_summary_with_different_baselines()
    features, _ = build_driving_style_features(summary, relative_to_driver=True)

    assert list(features["mean_speed_kmh"]) == pytest.approx([-2.5, 2.5, -2.5, 2.5])


@pytest.mark.unit
def test_relative_to_driver_each_driver_group_sums_to_zero():
    """Centralizar pela própria média implica que a soma dos deltas de
    cada piloto é zero, por construção."""
    summary = _two_driver_summary_with_different_baselines()
    features, metadata = build_driving_style_features(summary, relative_to_driver=True)

    features_with_driver = features.assign(driver=metadata["driver"])
    totals = features_with_driver.groupby("driver")["mean_speed_kmh"].sum()
    assert totals.to_numpy() == pytest.approx([0.0, 0.0], abs=1e-9)


@pytest.mark.unit
def test_relative_to_driver_false_is_still_the_default():
    """Sem o flag, comportamento continua absoluto (compatibilidade com
    quem já chama a função sem esse argumento)."""
    summary = _two_driver_summary_with_different_baselines()
    features, _ = build_driving_style_features(summary)

    assert list(features["mean_speed_kmh"]) == pytest.approx(
        [295.0, 300.0, 250.0, 255.0]
    )
