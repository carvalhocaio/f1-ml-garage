"""Teste de integração ponta-a-ponta: telemetria real de uma corrida ->
PCA -> k-means de estilo de pilotagem.

Mesmo padrão de `test_tyre_pipeline.py` — pulado por padrão, roda com
`make test-integration`, e igualmente mais lento que os testes de
voltas/resultados (telemetria de todos os pilotos de uma corrida).
"""

import math
import os

import pytest

from f1_ml_garage.data.session import (
    enable_cache,
    load_session_laps,
    load_session_telemetry,
)
from f1_ml_garage.features.driving_style import build_driving_style_features
from f1_ml_garage.features.telemetry_summary import (
    filter_to_clean_laps,
    summarize_lap_telemetry,
    tag_telemetry_with_lap,
)
from f1_ml_garage.models.clustering import (
    evaluate_clustering,
    fit_kmeans,
    fit_pca,
    standardize_features,
)

pytestmark = pytest.mark.integration

RUN_INTEGRATION = os.getenv("F1_ML_GARAGE_RUN_INTEGRATION") == "1"
SKIP_REASON = "defina F1_ML_GARAGE_RUN_INTEGRATION=1 para bater na API real do FastF1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
def test_clustering_pipeline_runs_end_to_end_on_real_session():
    enable_cache()
    laps = load_session_laps(2024, "Bahrain", "R")
    telemetry = load_session_telemetry(2024, "Bahrain", "R")

    tagged = tag_telemetry_with_lap(telemetry, laps)
    clean = filter_to_clean_laps(tagged, laps)
    summary = summarize_lap_telemetry(clean)
    features, _ = build_driving_style_features(summary)
    assert len(features) > 0

    scaled = standardize_features(features)
    pca, coords = fit_pca(scaled, n_components=2)
    _, labels = fit_kmeans(coords, n_clusters=3)
    result = evaluate_clustering(coords, labels)

    assert not any(math.isnan(value) for value in result.values())
    assert not any(math.isnan(ratio) for ratio in pca.explained_variance_ratio_)
    assert -1.0 <= result["silhouette_score"] <= 1.0
