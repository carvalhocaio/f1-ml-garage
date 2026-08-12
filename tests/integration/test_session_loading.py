"""Testes de integração: batem na rede real do FastF1.

Desabilitados por padrão porque dependem de conectividade e podem levar
minutos na primeira chamada, antes do cache local aquecer. Rode
explicitamente com `make test-integration` (ou
`F1_ML_GARAGE_RUN_INTEGRATION=1 uv run pytest -m integration`) quando quiser
validar contra a API real.
"""

import os

import pytest

from f1_ml_garage.data.session import (
    enable_cache,
    load_driver_telemetry,
    load_session_laps,
    load_session_results,
)

pytestmark = pytest.mark.integration

RUN_INTEGRATION = os.getenv("F1_ML_GARAGE_RUN_INTEGRATION") == "1"
SKIP_REASON = "defina F1_ML_GARAGE_RUN_INTEGRATION=1 para bater na API real do FastF1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
def test_loads_real_race_session():
    enable_cache()
    laps = load_session_laps(2024, "Bahrain", "R")

    assert len(laps) > 0
    assert {"driver", "lap_time_s", "compound", "stint"}.issubset(laps.columns)


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
def test_loads_real_race_results():
    enable_cache()
    results = load_session_results(2024, "Bahrain", "R")

    assert len(results) > 0
    assert {"driver", "position", "status", "dnf", "points"}.issubset(results.columns)


@pytest.mark.skipif(not RUN_INTEGRATION, reason=SKIP_REASON)
def test_loads_real_driver_telemetry():
    enable_cache()
    telemetry = load_driver_telemetry(2024, "Bahrain", "VER", "R")

    assert len(telemetry) > 0
    assert {"speed_kmh", "throttle_pct", "brake", "x_m", "y_m", "on_track"}.issubset(
        telemetry.columns
    )
