"""Normalização do DataFrame `SessionResults` bruto do FastF1.

Mesmo espírito de `laps.py`: schema de terceiros conhecido só aqui, tempos
convertidos para segundos, nenhuma lista filtrada. `dnf` é a única coluna
verdadeiramente derivada - computada, não apenas renomeada, porque é uma
feature de destino recorrente nos módulos seguintes (dados desbalanceados,
classificação de abandono).
"""

import pandas as pd

from f1_ml_garage.data.timeutils import timedelta_to_seconds
from f1_ml_garage.exceptions import MissingColumnsError

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "DriverNumber",
    "Abbreviation",
    "FullName",
    "TeamName",
    "CountryCode",
    "Position",
    "ClassifiedPosition",
    "GridPosition",
    "Q1",
    "Q2",
    "Q3",
    "Time",
    "Status",
    "Points",
    "Laps",
)

# Réplica, em regex, do critério usado por `fastf1.core.DriverResult.dnf`
# (`Status[3:6] == "Lap"` ou `Status == "Finished"`). Preferimos regex ao
# slice fixo do FastF1: o slice assume um único dígito no número de voltas
# de atraso e falha silenciosamente para "+10 Laps" ou mais (índices [3:6]
# caem em " La", não "Lap"). Raro, mas acontece em corridas com muitos
# safety cars/abandonos.
_LAPPED_BUT_CLASSIFIED = r"^\+\d+\s*Laps?$"


def _is_dnf(status: pd.Series) -> pd.Series:
    """True quando o piloto não é considerado como tendo terminado a prova."""
    finished = status == "Finished"
    lapped = status.str.match(_LAPPED_BUT_CLASSIFIED)
    return ~(finished | lapped)


def normalize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o DataFrame `SessionResults` bruto do FastF1.

    Levanta:
        MissingColumnsError: se `raw` não contiver alguma das colunas em
            `REQUIRED_RAW_COLUMNS`.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise MissingColumnsError(
            f"colunas ausentes no resultado bruto do FastF1: {missing}"
        )

    normalized = pd.DataFrame(index=raw.index)
    normalized["driver"] = raw["Abbreviation"]
    normalized["driver_number"] = raw["DriverNumber"]
    normalized["driver_name"] = raw["FullName"]
    normalized["team"] = raw["TeamName"]
    normalized["country_code"] = raw["CountryCode"]
    normalized["position"] = raw["Position"]
    normalized["classified_position"] = raw["ClassifiedPosition"]
    normalized["grid_position"] = raw["GridPosition"]
    normalized["positions_gained"] = raw["GridPosition"] - raw["Position"]
    normalized["q1_s"] = timedelta_to_seconds(raw["Q1"])
    normalized["q2_s"] = timedelta_to_seconds(raw["Q2"])
    normalized["q3_s"] = timedelta_to_seconds(raw["Q3"])
    normalized["race_time_s"] = timedelta_to_seconds(raw["Time"])
    normalized["status"] = raw["Status"]
    normalized["points"] = raw["Points"]
    normalized["laps_completed"] = raw["Laps"]
    normalized["dnf"] = _is_dnf(raw["Status"])

    return normalized
