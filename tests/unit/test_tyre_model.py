import numpy as np
import pandas as pd
import pytest

from f1_ml_garage.models.evaluation import evaluate_multiclass_classifier
from f1_ml_garage.models.tyre import build_tyre_svm_pipeline

N_DRIVERS = 15
LAPS_PER_COMPOUND = 4

# Três "estilos de condução" bem separados por composto, com um pouco de
# ruído gaussiano por cima — não são números reais de telemetria, só uma
# regra sintética conhecida por construção, mesmo espírito oráculo dos
# outros modelos (pace/dnf).
CLUSTERS = {
    "soft": {
        "mean_speed_kmh": 290,
        "max_speed_kmh": 310,
        "mean_throttle_pct": 85,
        "brake_fraction": 0.15,
        "mean_rpm": 11800,
        "mean_gear": 6.5,
    },
    "medium": {
        "mean_speed_kmh": 280,
        "max_speed_kmh": 300,
        "mean_throttle_pct": 75,
        "brake_fraction": 0.20,
        "mean_rpm": 11500,
        "mean_gear": 6.0,
    },
    "hard": {
        "mean_speed_kmh": 270,
        "max_speed_kmh": 290,
        "mean_throttle_pct": 65,
        "brake_fraction": 0.25,
        "mean_rpm": 11200,
        "mean_gear": 5.5,
    },
}
NOISE_STD = {
    "mean_speed_kmh": 4,
    "max_speed_kmh": 4,
    "mean_throttle_pct": 3,
    "brake_fraction": 0.02,
    "mean_rpm": 100,
    "mean_gear": 0.15,
}


def _separable_tyre_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    rows = []
    for driver_idx in range(N_DRIVERS):
        for compound, base in CLUSTERS.items():
            for _ in range(LAPS_PER_COMPOUND):
                row = {key: base[key] + rng.normal(0, NOISE_STD[key]) for key in base}
                row["compound"] = compound
                row["driver"] = f"D{driver_idx}"
                rows.append(row)

    data = pd.DataFrame(rows)
    feature_columns = list(CLUSTERS["soft"].keys())
    return data[feature_columns], data["compound"], data["driver"]


@pytest.mark.unit
def test_svm_recovers_well_separated_compound_clusters():
    features, target, groups = _separable_tyre_dataset()

    result = evaluate_multiclass_classifier(
        build_tyre_svm_pipeline(), features, target, groups, n_splits=5
    )

    assert result["accuracy"] > 0.9
    assert result["f1_macro"] > 0.9


@pytest.mark.unit
def test_returns_expected_metric_keys():
    features, target, groups = _separable_tyre_dataset()
    result = evaluate_multiclass_classifier(
        build_tyre_svm_pipeline(), features, target, groups, n_splits=5
    )

    assert set(result.keys()) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    }


@pytest.mark.unit
def test_unscaled_feature_does_not_dominate():
    """`mean_rpm` (~11000) tem escala ~1000x maior que `brake_fraction`
    (~0.2) — sem o StandardScaler do pipeline, a distância euclidiana do
    SVM seria dominada quase inteiramente por RPM, ignorando as outras
    features na prática. Confirma que o pipeline generaliza bem mesmo com
    essa diferença de escala real entre as features."""
    features, target, groups = _separable_tyre_dataset()
    assert features["mean_rpm"].std() > 50 * features["brake_fraction"].std()

    result = evaluate_multiclass_classifier(
        build_tyre_svm_pipeline(), features, target, groups, n_splits=5
    )
    assert result["accuracy"] > 0.9
