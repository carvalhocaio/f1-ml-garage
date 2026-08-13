"""Teste de integração ponta-a-ponta: várias corridas reais -> features ->
classificador de DNF.

Mesmo padrão dos outros testes de integração — pulado por padrão, roda com
`make test-integration`. Limita a poucas rodadas (`rounds=[1, 2, 3, 4, 5]`)
de propósito: o objetivo é validar que o pipeline não quebra em dado real
(times/pilotos variados, proporção de DNF real), não treinar um modelo de
verdade — isso é trabalho pra rodar manualmente com a temporada inteira.
"""

import math
import os

import pytest

from f1_ml_garage.data.session import enable_cache, load_season_results
from f1_ml_garage.features.dnf import build_dnf_features
from f1_ml_garage.models.dnf import evaluate_dnf_model

pytestmark = pytest.mark.integration

RUN_INTEGRATION = os.getenv("F1_ML_GARAGE_RUN_INTEGRATION") == "1"
SKIP_REASON = "defina F1_ML_GARAGE_RUN_INTEGRATION=1 para bater na API real do FastF1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
def test_dnf_pipeline_runs_end_to_end_on_real_season_slice():
    enable_cache()
    results = load_season_results(2024, rounds=[1, 2, 3, 4, 5])
    assert len(results) > 0
    assert {"round_number", "event_name"}.issubset(results.columns)

    features, target, groups = build_dnf_features(results)
    result = evaluate_dnf_model(features, target, groups, n_splits=3)

    assert not any(math.isnan(value) for value in result.values())
    assert 0.0 <= result["dnf_rate"] <= 1.0
