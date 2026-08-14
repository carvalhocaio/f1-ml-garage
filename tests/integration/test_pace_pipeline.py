"""Teste de integração ponta-a-ponta: sessão real -> features -> modelo.

Mesmo padrão de `test_session_loading.py` — pulado por padrão, roda com
`make test-integration`. As asserções aqui são propositalmente frouxas
(sanidade, não qualidade do modelo): o objetivo é pegar quebras de pipeline
em dado real e "sujo" (NaNs, categorias inesperadas, contagem de voltas
variável por piloto) que fixtures sintéticas não expõem — não validar se
0.03s/volta de degradação é um R² "bom".
"""

import math
import os

import pytest

from f1_ml_garage.data.laps import select_green_flag_laps
from f1_ml_garage.data.session import enable_cache, load_session_laps
from f1_ml_garage.features.pace import (
    build_pace_features,
    compute_driver_delta_target,
)
from f1_ml_garage.models.pace import evaluate_pace_model

pytestmark = pytest.mark.integration

RUN_INTEGRATION = os.getenv("F1_ML_GARAGE_RUN_INTEGRATION") == "1"
SKIP_REASON = "defina F1_ML_GARAGE_RUN_INTEGRATION=1 para bater na API real do FastF1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
def test_pace_pipeline_runs_end_to_end_on_real_session():
    enable_cache()
    laps = load_session_laps(2024, "Bahrain", "R")

    green = select_green_flag_laps(laps)
    assert len(green) > 0

    features, _, groups = build_pace_features(green)
    delta_target = compute_driver_delta_target(green)
    result = evaluate_pace_model(features, delta_target, groups, n_splits=5)

    assert not any(math.isnan(value) for value in result.values())
    assert result["mae_s"] > 0
    assert result["r2"] <= 1.0
