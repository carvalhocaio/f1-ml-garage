import pandas as pd
import pytest

from f1_ml_garage.data.laps import (
    filter_accurate_laps,
    normalize_laps,
    select_green_flag_laps,
)
from f1_ml_garage.exceptions import MissingColumnsError


def _raw_laps(**overrides: object) -> pd.DataFrame:
    """Monta um DataFrame bruto no schema do FastF1 (`Laps._COLUMNS`) com
    2 voltas de 1 piloto e valores padrão razoáveis. `overrides` substitui
    colunas inteiras — útil para casos de borda (NaT, voltas deletadas,
    pit stops etc.).
    """
    base = pd.DataFrame(
        {
            "Time": pd.to_timedelta(["0 days 00:01:35", "0 days 00:03:03"]),
            "Driver": ["VER", "VER"],
            "DriverNumber": ["1", "1"],
            "Team": ["Red Bull Racing", "Red Bull Racing"],
            "LapNumber": [1.0, 2.0],
            "LapTime": pd.to_timedelta(["0 days 00:01:35.256", "0 days 00:01:28.104"]),
            "Sector1Time": pd.to_timedelta(["0 days 00:00:28", "0 days 00:00:27"]),
            "Sector2Time": pd.to_timedelta(["0 days 00:00:35", "0 days 00:00:34"]),
            "Sector3Time": pd.to_timedelta(
                ["0 days 00:00:32.256", "0 days 00:00:27.104"]
            ),
            "Stint": [1.0, 1.0],
            "Compound": ["SOFT", "SOFT"],
            "TyreLife": [1.0, 2.0],
            "FreshTyre": [True, True],
            "TrackStatus": ["1", "1"],
            "Position": [1.0, 1.0],
            "IsAccurate": [True, True],
            "Deleted": [False, False],
            "PitInTime": pd.to_timedelta([pd.NaT, pd.NaT]),
            "PitOutTime": pd.to_timedelta([pd.NaT, pd.NaT]),
        }
    )
    return base.assign(**overrides)


@pytest.mark.unit
def test_converts_lap_and_sector_times_to_seconds():
    laps = normalize_laps(_raw_laps())

    assert laps.loc[0, "lap_time_s"] == pytest.approx(95.256)
    assert laps.loc[0, "sector1_s"] == pytest.approx(28.0)
    assert laps.loc[1, "lap_time_s"] == pytest.approx(88.104)


@pytest.mark.unit
def test_preserves_row_count_and_does_not_filter():
    """normalize_laps é puramente estrutural: filtragem é responsabilidade
    de filter_accurate_laps, não desta função."""
    raw = _raw_laps(IsAccurate=[True, False], Deleted=[False, True])
    laps = normalize_laps(raw)
    assert len(laps) == 2


@pytest.mark.unit
def test_missing_lap_time_becomes_nan_not_error():
    raw = _raw_laps(LapTime=pd.to_timedelta([pd.NaT, "0 days 00:01:28.104"]))
    laps = normalize_laps(raw)
    assert pd.isna(laps.loc[0, "lap_time_s"])


@pytest.mark.unit
def test_flags_pit_in_and_out_laps():
    raw = _raw_laps(
        PitOutTime=pd.to_timedelta(["0 days 00:01:30", pd.NaT]),
        PitInTime=pd.to_timedelta([pd.NaT, "0 days 00:03:00"]),
    )
    laps = normalize_laps(raw)

    assert laps.loc[0, "is_pit_out_lap"]
    assert not laps.loc[0, "is_pit_in_lap"]
    assert laps.loc[1, "is_pit_in_lap"]
    assert not laps.loc[1, "is_pit_out_lap"]


@pytest.mark.unit
def test_compound_is_lowercased():
    laps = normalize_laps(_raw_laps())
    assert (laps["compound"] == "soft").all()


@pytest.mark.unit
def test_raises_on_missing_required_column():
    raw = _raw_laps().drop(columns=["LapTime"])
    with pytest.raises(MissingColumnsError):
        normalize_laps(raw)


@pytest.mark.unit
def test_filter_accurate_laps_drops_inaccurate_deleted_and_no_time():
    raw = _raw_laps(
        IsAccurate=[True, False],
        Deleted=[False, False],
        LapTime=pd.to_timedelta(["0 days 00:01:28", pd.NaT]),
    )
    laps = normalize_laps(raw)
    accurate = filter_accurate_laps(laps)

    assert len(accurate) == 1
    assert accurate.loc[0, "lap_time_s"] == pytest.approx(88.0)


@pytest.mark.unit
def test_select_green_flag_laps_drops_non_green_status():
    raw = _raw_laps(TrackStatus=["1", "2"])
    laps = normalize_laps(raw)
    green = select_green_flag_laps(laps)
    assert len(green) == 1


@pytest.mark.unit
def test_select_green_flag_laps_drops_inaccurate_laps():
    raw = _raw_laps(IsAccurate=[True, False])
    laps = normalize_laps(raw)
    green = select_green_flag_laps(laps)
    assert len(green) == 1
