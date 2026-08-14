import pandas as pd
import pytest

from f1_ml_garage.features.telemetry_summary import (
    summarize_lap_telemetry,
    tag_telemetry_with_lap,
)


def _laps() -> pd.DataFrame:
    """3 voltas de VER, subset mínimo de colunas que `tag_telemetry_with_lap`
    usa (`session_time_s` marca o FIM de cada volta)."""
    return pd.DataFrame(
        {
            "driver": ["VER", "VER", "VER"],
            "lap_number": [1, 2, 3],
            "session_time_s": [100.0, 190.0, 275.0],
            "compound": ["soft", "soft", "medium"],
        }
    )


def _telemetry() -> pd.DataFrame:
    """5 amostras de telemetria de VER, espalhadas pelas 3 voltas acima."""
    return pd.DataFrame(
        {
            "driver": ["VER", "VER", "VER", "VER", "VER"],
            "session_time_s": [50.0, 95.0, 150.0, 200.0, 270.0],
            "speed_kmh": [280.0, 300.0, 250.0, 260.0, 265.0],
            "throttle_pct": [80.0, 100.0, 60.0, 70.0, 75.0],
            "brake": [False, False, True, False, False],
            "rpm": [11000.0, 11800.0, 10500.0, 10800.0, 10900.0],
            "gear": [6, 7, 5, 6, 6],
        }
    )


@pytest.mark.unit
def test_tag_telemetry_with_lap_assigns_correct_lap():
    tagged = tag_telemetry_with_lap(_telemetry(), _laps())

    # amostras em 50s/95s -> volta 1 (termina em 100s)
    # amostra em 150s -> volta 2 (termina em 190s)
    # amostras em 200s/270s -> volta 3 (termina em 275s)
    assert list(tagged["lap_number"]) == [1, 1, 2, 3, 3]
    assert list(tagged["compound"]) == ["soft", "soft", "soft", "medium", "medium"]


@pytest.mark.unit
def test_tag_telemetry_with_lap_respects_driver_boundaries():
    """Tempos de sessão se sobrepõem entre pilotos — sem separar por
    `driver`, a amostra de um poderia grudar na volta de outro."""
    laps = pd.concat(
        [
            _laps(),
            pd.DataFrame(
                {
                    "driver": ["HAM"],
                    "lap_number": [1],
                    "session_time_s": [60.0],
                    "compound": ["hard"],
                }
            ),
        ],
        ignore_index=True,
    )
    telemetry = pd.concat(
        [
            _telemetry(),
            pd.DataFrame(
                {
                    "driver": ["HAM"],
                    "session_time_s": [55.0],
                    "speed_kmh": [200.0],
                    "throttle_pct": [50.0],
                    "brake": [False],
                    "rpm": [9000.0],
                    "gear": [4],
                }
            ),
        ],
        ignore_index=True,
    )

    tagged = tag_telemetry_with_lap(telemetry, laps)

    ham_row = tagged.loc[tagged["driver"] == "HAM"]
    assert list(ham_row["compound"]) == ["hard"]
    # a amostra de VER em 50s continua indo pra volta 1 do VER (soft), não
    # é afetada pela volta do HAM que termina em 60s.
    ver_first = tagged.loc[
        (tagged["driver"] == "VER") & (tagged["session_time_s"] == 50.0)
    ]
    assert list(ver_first["compound"]) == ["soft"]


@pytest.mark.unit
def test_summarize_lap_telemetry_aggregates_correctly():
    tagged = tag_telemetry_with_lap(_telemetry(), _laps())
    summary = summarize_lap_telemetry(tagged)

    lap1 = summary.loc[
        (summary["driver"] == "VER") & (summary["lap_number"] == 1)
    ].iloc[0]

    # volta 1: amostras de 280/300 km/h, throttle 80/100, brake sempre False
    assert lap1["mean_speed_kmh"] == pytest.approx(290.0)
    assert lap1["max_speed_kmh"] == pytest.approx(300.0)
    assert lap1["mean_throttle_pct"] == pytest.approx(90.0)
    assert lap1["brake_fraction"] == pytest.approx(0.0)
    assert lap1["compound"] == "soft"


@pytest.mark.unit
def test_summarize_lap_telemetry_one_row_per_driver_lap():
    tagged = tag_telemetry_with_lap(_telemetry(), _laps())
    summary = summarize_lap_telemetry(tagged)
    # 3 voltas de VER na telemetria de teste (a volta 2 não tem amostra
    # própria nesse fixture pequeno, mas a 1 e a 3 sim)
    unique_lap_count = summary[["driver", "lap_number"]].drop_duplicates().shape[0]
    assert len(summary) == unique_lap_count
