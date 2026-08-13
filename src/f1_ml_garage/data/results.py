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

# Valores de Status observados no schema do FastF1 (backend nativo,
# >=2018): confirmado contra a temporada 2024 real via
# `results["status"].value_counts()` -> "Finished", "Lapped", "Retired",
# "Did not start", "Disqualified". NENHUM deles usa o formato "+N Lap(s)"
# que a documentação oficial do FastF1 lista como exemplo — a versão
# instalada (v3.8.3) usa strings categóricas simples, não esse formato.
# Uma primeira versão desta função usava uma regex pra casar "+N Lap(s)",
# que nunca batia com "Lapped" — classificando 138 de 479 pilotos (a
# temporada inteira de 2024) como DNF por engano (taxa de "DNF" de ~40%,
# quando o valor real é bem mais baixo).
#
# Allowlist (não denylist) de propósito: um status novo e desconhecido (de
# uma temporada futura, ou do backend Ergast pra temporadas <2018) conta
# como DNF por padrão — mais seguro que o contrário, já que assumir
# "terminou" por engano esconderia um abandono real do alvo.
_FINISHED_STATUSES = frozenset({"Finished", "Lapped"})


def _is_dnf(status: pd.Series) -> pd.Series:
    """True quando o piloto não é considerado como tendo terminado a prova."""
    return ~status.isin(_FINISHED_STATUSES)


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
