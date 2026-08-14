"""Teste de integração ponta-a-ponta: telemetria real de uma corrida ->
features por volta -> classificador de composto (SVM).

Mesmo padrão dos outros testes de integração — pulado por padrão, roda com
`make test-integration`. Mais lento que os demais: `load_session_telemetry`
baixa telemetria de TODOS os pilotos de uma corrida (ordem de grandeza a
mais de dado que voltas/resultados), mesmo com cache quente.
"""

import math
import os

import pytest

from f1_ml_garage.data.session import (
    enable_cache,
    load_session_laps,
    load_session_telemetry,
)
from f1_ml_garage.features.tyre import (
    build_tyre_features,
    summarize_lap_telemetry,
    tag_telemetry_with_lap,
)
from f1_ml_garage.models.evaluation import evaluate_multiclass_classifier
from f1_ml_garage.models.tyre import build_tyre_svm_pipeline

pytestmark = pytest.mark.integration

RUN_INTEGRATION = os.getenv("F1_ML_GARAGE_RUN_INTEGRATION") == "1"
SKIP_REASON = "defina F1_ML_GARAGE_RUN_INTEGRATION=1 para bater na API real do FastF1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
def test_tyre_pipeline_runs_end_to_end_on_real_session():
    enable_cache()
    laps = load_session_laps(2024, "Bahrain", "R")
    telemetry = load_session_telemetry(2024, "Bahrain", "R")

    tagged = tag_telemetry_with_lap(telemetry, laps)
    summary = summarize_lap_telemetry(tagged)
    assert len(summary) > 0

    features, target, groups = build_tyre_features(summary)
    result = evaluate_multiclass_classifier(
        build_tyre_svm_pipeline(), features, target, groups, n_splits=3
    )

    assert not any(math.isnan(value) for value in result.values())
    assert 0.0 <= result["accuracy"] <= 1.0
