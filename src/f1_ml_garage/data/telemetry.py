"""Normalização de telemetria do FastF1.

Cobre o schema combinado retornado por
`session.laps.pick_driver(...).get_telemetry()`: canais de carro (velocidade,
RPM, marcha, acelerador, freio, DRS) e de posição (X/Y/Z, status em/fora de
pista), amostrados em alta frequência — uma linha por amostra, não por
volta.
"""

import pandas as pd

from f1_ml_garage.data.timeutils import timedelta_to_seconds
from f1_ml_garage.exceptions import MissingColumnsError

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "Date",
    "Time",
    "SessionTime",
    "Speed",
    "RPM",
    "nGear",
    "Throttle",
    "Brake",
    "DRS",
    "Source",
    "X",
    "Y",
    "Z",
    "Status",
)

# FastF1 documenta 104 como valor de erro/indisponibilidade em `Throttle`,
# distinto de um acelerador saturado em 100% — tratamos como amostra
# inválida em vez de aceitar como leitura real.
THROTTLE_ERROR_VALUE = 104

# `X`/`Y`/`Z` são reportados pelo FastF1 em décimos de metro (ver docstring
# de `fastf1.core.Telemetry`). Convertemos para metros aqui para não deixar
# essa unidade não-óbvia vazar para os módulos de clustering/distância.
POSITION_SCALE_TO_METERS = 0.1


def normalize_telemetry(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza um DataFrame de telemetria bruto do FastF1.

    Assim como `normalize_laps`, não filtra nenhuma amostra - inclusive as
    marcadas como `throttle_invalid`, já que descartá-las aqui impediria um
    módulo futuro de, por exemplo, contar quão frequente é o erro por
    sessão.

    Levanta:
        MissingColumnsError: se `raw` não contiver alguma coluna exigida.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise MissingColumnsError(
            f"colunas ausentes na telemetria bruta do FastF1: {missing}"
        )

    normalized = pd.DataFrame(index=raw.index)
    normalized["timestamp"] = raw["Date"]
    normalized["time_s"] = timedelta_to_seconds(raw["Time"])
    normalized["session_time_s"] = timedelta_to_seconds(raw["SessionTime"])
    normalized["speed_kmh"] = raw["Speed"]
    normalized["rpm"] = raw["RPM"]
    normalized["gear"] = raw["nGear"]
    normalized["throttle_invalid"] = raw["Throttle"] == THROTTLE_ERROR_VALUE
    normalized["throttle_pct"] = raw["Throttle"].where(~normalized["throttle_invalid"])
    normalized["brake"] = raw["Brake"].astype(bool)
    normalized["drs"] = raw["DRS"]
    normalized["source"] = raw["Source"]
    normalized["x_m"] = raw["X"] * POSITION_SCALE_TO_METERS
    normalized["y_m"] = raw["Y"] * POSITION_SCALE_TO_METERS
    normalized["z_m"] = raw["Z"] * POSITION_SCALE_TO_METERS
    normalized["on_track"] = raw["Status"] == "OnTrack"

    return normalized
