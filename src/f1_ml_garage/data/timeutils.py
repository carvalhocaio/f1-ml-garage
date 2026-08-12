"""Utilitários de tempo compartilhados entre os módulos de normalização.

FastF1 representa tempo como `pd.Timedelta` em várias partes do schema —
voltas, setores, classificação, corrida, telemetria. Centralizar a conversão
aqui evita reimplementar a mesma linha em `laps.py`, `results.py` e
`telemetry.py`.
"""

import pandas as pd


def timedelta_to_seconds(series: pd.Series) -> pd.Series:
    """Converte uma coluna `timedelta64[ns]` do FastF1 para segundos (float).

    `NaT` vira `NaN` naturalmente via `.dt.total_seconds()`.
    """
    return series.dt.total_seconds()
