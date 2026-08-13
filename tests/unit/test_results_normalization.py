import pandas as pd
import pytest

from f1_ml_garage.data.results import normalize_results
from f1_ml_garage.exceptions import MissingColumnsError


def _raw_results(**overrides: object) -> pd.DataFrame:
    """Monta um DataFrame bruto no schema do FastF1 (`SessionResults._COLUMNS`)
    com 5 pilotos, um pra cada valor real de Status observado na temporada
    2024 (`Finished`, `Lapped`, `Retired`, `Did not start`, `Disqualified`)
    — não o formato "+N Lap" que a documentação sugere mas que a versão
    instalada do FastF1 não usa de fato (ver comentário em `results.py`).
    """
    base = pd.DataFrame(
        {
            "DriverNumber": ["1", "44", "16", "63", "4"],
            "Abbreviation": ["VER", "HAM", "LEC", "RUS", "NOR"],
            "FullName": [
                "Max Verstappen",
                "Lewis Hamilton",
                "Charles Leclerc",
                "George Russell",
                "Lando Norris",
            ],
            "TeamName": [
                "Red Bull Racing",
                "Mercedes",
                "Ferrari",
                "Mercedes",
                "McLaren",
            ],
            "CountryCode": ["NED", "GBR", "MON", "GBR", "GBR"],
            "Position": [1.0, 2.0, float("nan"), 3.0, float("nan")],
            "ClassifiedPosition": ["1", "2", "R", "3", "D"],
            "GridPosition": [1.0, 3.0, 5.0, 4.0, 2.0],
            "Q1": pd.to_timedelta(["0 days 00:01:30"] * 5),
            "Q2": pd.to_timedelta(["0 days 00:01:29"] * 5),
            "Q3": pd.to_timedelta(
                [
                    "0 days 00:01:28",
                    "0 days 00:01:29",
                    pd.NaT,
                    "0 days 00:01:30",
                    pd.NaT,
                ]
            ),
            "Time": pd.to_timedelta(
                [
                    "0 days 01:32:00",
                    "0 days 01:32:15",
                    pd.NaT,
                    "0 days 01:32:40",
                    pd.NaT,
                ]
            ),
            "Status": [
                "Finished",
                "Lapped",
                "Retired",
                "Did not start",
                "Disqualified",
            ],
            "Points": [25.0, 18.0, 0.0, 15.0, 0.0],
            "Laps": [57.0, 57.0, 40.0, 0.0, 57.0],
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
def test_lapped_driver_is_not_dnf():
    """ "Lapped" é o valor real que o FastF1 usa pra "terminou, mas voltas
    atrás do líder" — não é um abandono. (Não existe "+N Lap" na versão
    instalada; ver comentário em `results.py` sobre o bug que isso causou.)
    """
    results = normalize_results(_raw_results())
    assert not results.loc[1, "dnf"]


@pytest.mark.unit
def test_retired_driver_is_dnf():
    results = normalize_results(_raw_results())
    assert results.loc[2, "dnf"]


@pytest.mark.unit
def test_did_not_start_driver_is_dnf():
    results = normalize_results(_raw_results())
    assert results.loc[3, "dnf"]


@pytest.mark.unit
def test_disqualified_driver_is_dnf():
    results = normalize_results(_raw_results())
    assert results.loc[4, "dnf"]


@pytest.mark.unit
def test_unknown_status_defaults_to_dnf():
    """Allowlist, não denylist: um status nunca visto (ex.: mudança futura
    no schema do FastF1, ou backend Ergast pra temporadas <2018) tem que
    contar como DNF por padrão — mais seguro que assumir "terminou" sem
    saber."""
    raw = _raw_results(
        Status=["Finished", "Withdrawn", "Retired", "Did not start", "Disqualified"]
    )
    results = normalize_results(raw)
    assert results.loc[1, "dnf"]


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
