import pandas as pd
import pytest

from f1_ml_garage.data.results import normalize_results
from f1_ml_garage.exceptions import MissingColumnsError


def _raw_results(**overrides: object) -> pd.DataFrame:
    """Monta um DataFrame bruto no schema do FastF1 (`SessionResults._COLUMNS`)
    com 3 pilotos: um vendedor, um abandono clássico, e um "classificado com
    voltas de atraso" (caso que testa o regex de `_is_dnf`).
    """
    base = pd.DataFrame(
        {
            "DriverNumber": ["1", "44", "16"],
            "Abbreviation": ["VER", "HAM", "LEC"],
            "FullName": ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc"],
            "TeamName": ["Red Bull Racing", "Mercedes", "Ferrari"],
            "CountryCode": ["NED", "GBR", "MON"],
            "Position": [1.0, 2.0, float("nan")],
            "ClassifiedPosition": ["1", "2", "R"],
            "GridPosition": [1.0, 3.0, 5.0],
            "Q1": pd.to_timedelta(
                ["0 days 00:01:30", "0 days 00:01:31", "0 days 00:01:29"]
            ),
            "Q2": pd.to_timedelta(
                ["0 days 00:01:29", "0 days 00:01:30", "0 days 00:01:28"]
            ),
            "Q3": pd.to_timedelta(["0 days 00:01:28", "0 days 00:01:29", pd.NaT]),
            "Time": pd.to_timedelta(["0 days 01:32:00", "0 days 01:32:15", pd.NaT]),
            "Status": ["Finished", "+1 Lap", "Accident"],
            "Points": [25.0, 18.0, 0.0],
            "Laps": [57.0, 57.0, 40.0],
        }
    )
    return base.assign(**overrides)


@pytest.mark.unit
def test_converts_qualifying_and_race_times_to_seconds():
    results = normalize_results(_raw_results())

    assert results.loc[0, "q3_s"] == pytest.approx(88.0)
    assert results.loc[0, "race_time_s"] == pytest.approx(5520.0)


@pytest.mark.unit
def test_finished_driver_is_not_dnf():
    results = normalize_results(_raw_results())
    assert not results.loc[0, "dnf"]


@pytest.mark.unit
def test_lapped_but_classified_driver_is_not_dnf():
    """ "+1 Lap" significa que o piloto terminou, só que voltas atrás do
    líder - não é um abandono."""
    results = normalize_results(_raw_results())
    assert not results.loc[1, "dnf"]


@pytest.mark.unit
def test_retired_driver_is_dnf():
    results = normalize_results(_raw_results())
    assert results.loc[2, "dnf"]


@pytest.mark.unit
def test_double_digit_lapped_driver_is_not_dnf():
    """Caso que o slice fixo Status[3:6] do FastF1 erra: "+10 Laps" tem que
    ser reconhecido como classificado, não como abandono."""
    raw = _raw_results(Status=["Finished", "+10 Laps", "Accident"])
    results = normalize_results(raw)
    assert not results.loc[1, "dnf"]


@pytest.mark.unit
def test_computes_positions_gained_from_grid():
    results = normalize_results(_raw_results())
    assert results.loc[0, "positions_gained"] == pytest.approx(0.0)
    assert results.loc[1, "positions_gained"] == pytest.approx(1.0)


@pytest.mark.unit
def test_raises_on_missing_required_column():
    raw = _raw_results().drop(columns=["Status"])
    with pytest.raises(MissingColumnsError):
        normalize_results(raw)
