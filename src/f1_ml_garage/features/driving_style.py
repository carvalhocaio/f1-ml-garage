"""Features de estilo de pilotagem por volta, pra clustering (Módulo 4).

Reusa o mesmo resumo por volta de `telemetry_summary.py` que o SVM de
composto usa (`tyre.py`) — as mesmas estatísticas de condução (velocidade,
acelerador, freio, RPM, marcha) que distinguem composto também distinguem
estilo de pilotagem; é a mesma matéria-prima, com uma pergunta diferente.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "mean_speed_kmh",
    "max_speed_kmh",
    "mean_throttle_pct",
    "mean_rpm",
    "mean_gear",
]


def build_driving_style_features(
    lap_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta a matriz de features (X) e os metadados de cada linha
    (`driver`, `lap_number`, `compound`) a partir de saída de
    `telemetry_summary.summarize_lap_telemetry`.

    Sem alvo nem grupos aqui - clustering não tem rótulo pra prever, nem
    validação cruzada supervisionada. Os metadados voltam separados para
    interpretar os clusters DEPOIS de formatos (ex.: "o cluster 2 é quase
    todo volta de composto macio?"), não pra treinar nada.
    """
    features = lap_summary[FEATURE_COLUMNS].reset_index(drop=True)
    meatadata = lap_summary[["driver", "lap_number", "compound"]].reset_index(drop=True)
    return features, meatadata
