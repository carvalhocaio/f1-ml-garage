import pandas as pd
import pytest

from f1_ml_garage.data.telemetry import normalize_telemetry
from f1_ml_garage.exceptions import MissingColumnsError


def _raw_telemetry(**overrides: object) -> pd.DataFrame:
    """Monta um DataFrame bruto no schema combinado de telemetria do FastF1
    (`Telemetry._COLUMNS`) com 3 amostras de 1 volta.
    """
    base = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01 12:00:00", "2024-01-01 12:00:01", "2024-01-01 12:00:02"]
            ),
            "Time": pd.to_timedelta(["0s", "1s", "2s"]),
            "SessionTime": pd.to_timedelta(["100s", "101s", "102s"]),
            "Speed": [280.0, 290.0, 300.0],
            "RPM": [11500.0, 11800.0, 12000.0],
            "nGear": [7, 7, 8],
            "Throttle": [95.0, 104.0, 100.0],
            "Brake": [False, False, True],
            "DRS": [0, 0, 12],
            "Source": ["car", "car", "car"],
            "X": [1000.0, 1050.0, 1100.0],
            "Y": [2000.0, 2010.0, 2020.0],
            "Z": [30.0, 30.0, 31.0],
            "Status": ["OnTrack", "OnTrack", "OffTrack"],
        }
    )
    return base.assign(**overrides)


@pytest.mark.unit
def test_converts_time_columns_to_seconds():
    telemetry = normalize_telemetry(_raw_telemetry())

    assert telemetry.loc[1, "time_s"] == pytest.approx(1.0)
    assert telemetry.loc[0, "session_time_s"] == pytest.approx(100.0)


@pytest.mark.unit
def test_flags_error_throttle_value_as_invalid_and_masks_it():
    """104 é o valor de erro documentado pelo FastF1, não 104% real."""
    telemetry = normalize_telemetry(_raw_telemetry())

    assert telemetry.loc[1, "throttle_invalid"]
    assert pd.isna(telemetry.loc[1, "throttle_pct"])
    assert not telemetry.loc[0, "throttle_invalid"]
    assert telemetry.loc[0, "throttle_pct"] == pytest.approx(95.0)


@pytest.mark.unit
def test_converts_position_from_decimeters_to_meters():
    telemetry = normalize_telemetry(_raw_telemetry())
    assert telemetry.loc[0, "x_m"] == pytest.approx(100.0)
    assert telemetry.loc[0, "y_m"] == pytest.approx(200.0)


@pytest.mark.unit
def test_flags_on_track_status():
    telemetry = normalize_telemetry(_raw_telemetry())
    assert telemetry.loc[0, "on_track"]
    assert not telemetry.loc[2, "on_track"]


@pytest.mark.unit
def test_preserves_row_count_including_invalid_throttle_samples():
    telemetry = normalize_telemetry(_raw_telemetry())
    assert len(telemetry) == 3


@pytest.mark.unit
def test_raises_on_missing_required_column():
    raw = _raw_telemetry().drop(columns=["Speed"])
    with pytest.raises(MissingColumnsError):
        normalize_telemetry(raw)
