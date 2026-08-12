"""Normalização do DataFrame `Laps` bruto do FastF1.

Este é o único módulo do projeto que conhece o schema de terceiros do
FastF1 (`fastf1.core.Laps._COLUMNS`). Todo módulo de ML downstream (feature
engineering, clustering, modelos) opera sobre o schema normalizado definido
aqui, em snake_case e com tempos em segundos - se uma versão futura do
FastF1 renomear ou remover uma coluna, o ajuste fica isolado neste arquivo.

As funções aqui são puras (DataFrame in, DataFrame out) e não tocam rede:
isso é o que torna possível testá-las com fixtures pequenas, sem depender
da API real da F1. O carregamento via rede vive em `session.py`.
"""

import pandas as pd

from f1_ml_garage.exceptions import MissingColumnsError

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "Time",
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "TrackStatus",
    "Position",
    "IsAccurate",
    "Deleted",
    "PitInTime",
    "PitOutTime",
)


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:
    """Converte uma coluna `timedelta64[ns]` do FastF1 para segundos (float).

    FastF1 representa tempos de volta/setor como `pd.Timedelta`. Resolver
    isso para segundos logo na normalização evita que módulos de ML tenham
    que lidar com aritmética de Timedelta mais adiante; `NaT` vira `NaN`
    naturalmente via `.dt.total_seconds()`.
    """
    return series.dt.total_seconds()


def normalize_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o DataFrame `Laps` bruto do FastF1 para um schema estável.

    Deliberamente NÃO filtra nenhuma linha (voltas de entrada/saída, sob
    safety car, deletadas continuam presentes com o mesmo número de linhas
    de entrada) - filtragem é decisão de negócio de cada módulo consumidor,
    não desta função. Ver `filter_accurate_laps` para o filtro padrão usado
    em modelos de ritmo de corrida.

    Levanta:
        MissingColumnsError: se `raw` não contiver alguma das colunas em
            `REQUIRED_RAW_COLUMNS`.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise MissingColumnsError(
            f"colunas ausentes no DataFrame bruto do FastF1: {missing}"
        )

    normalized = pd.DataFrame(index=raw.index)
    normalized["driver"] = raw["Driver"]
    normalized["driver_number"] = raw["DriverNumber"]
    normalized["team"] = raw["Team"]
    normalized["lap_number"] = raw["LapNumber"]
    normalized["session_time_s"] = _timedelta_to_seconds(raw["Time"])
    normalized["lap_time_s"] = _timedelta_to_seconds(raw["LapTime"])
    normalized["sector1_s"] = _timedelta_to_seconds(raw["Sector1Time"])
    normalized["sector2_s"] = _timedelta_to_seconds(raw["Sector2Time"])
    normalized["sector3_s"] = _timedelta_to_seconds(raw["Sector3Time"])
    normalized["stint"] = raw["Stint"]
    normalized["compound"] = raw["Compound"].str.lower()
    normalized["tyre_life"] = raw["TyreLife"]
    normalized["fresh_tyre"] = raw["FreshTyre"]
    normalized["track_status"] = raw["TrackStatus"]
    normalized["position"] = raw["Position"]
    normalized["is_accurate"] = raw["IsAccurate"]
    normalized["deleted"] = raw["Deleted"].astype("boolean")
    normalized["is_pit_in_lap"] = raw["PitInTime"].notna()
    normalized["is_pit_out_lap"] = raw["PitOutTime"].notna()

    return normalized


def filter_accurate_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas voltas cronometradas confiáveis para modelagem de ritmo.

    Critério: `is_accurate` verdadeiro (o próprio FastF1 já avalia e marca
    anomalias de tempo), não deletada, e com `lap_time_s` presente - isso
    descarta automaticamente voltas de entrada/saída do pit e voltas sob
    safety car/red flag, que não têm `LapTime` computado.

    Recebe e devolve o schema normalizado (saída de `normalize_laps`), não o
    DataFrame bruto do FastF1.
    """
    mask = (
        laps["is_accurate"].fillna(False)
        & ~laps["deleted"].fillna(False)
        & laps["lap_time_s"].notna()
    )
    return laps.loc[mask].reset_index(drop=True)
