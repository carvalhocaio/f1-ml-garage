"""Features para classificar composto de pneu a partir de telemetria
(Módulo 2, SVM + kernels).

O join telemetria->volta e a agregação por volta moraram aqui antes; agora
vivem em `telemetry_summary.py` (reusados também pelo Módulo 4, clustering
de estilo de pilotagem). Este módulo fica só com o que é específico de
composto: ligar o resumo por volta ao alvo `compound`.
"""

import pandas as pd


def build_tyre_features(
    lap_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Monta a matriz de features (X), o alvo (y) e os grupos (piloto) a
    partir da saída de `telemetry_summary.summarize_lap_telemetry`.

    Todas as features são contínuas (diferente de `dnf`/`pace`, sem
    dummies categóricas aqui) — SVM é sensível à escala delas, mas isso é
    responsabilidade do pipeline do modelo (`StandardScaler`), não desta
    função.
    """
    feature_columns = [
        "mean_speed_kmh",
        "max_speed_kmh",
        "mean_throttle_pct",
        "brake_fraction",
        "mean_rpm",
        "mean_gear",
    ]
    features = lap_summary[feature_columns].reset_index(drop=True)
    target = lap_summary["compound"].reset_index(drop=True)
    groups = lap_summary["driver"].reset_index(drop=True)

    return features, target, groups
