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
    "brake_fraction",
    "mean_rpm",
    "mean_gear",
]


def build_driving_style_features(
    lap_summary: pd.DataFrame, *, relative_to_driver: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta a matriz de features (X) e os metadados de cada linha
    (`driver`, `lap_number`, `compound`) a partir da saída de
    `telemetry_summary.summarize_lap_telemetry`.

    Sem alvo nem grupos aqui — clustering não tem rótulo pra prever, nem
    validação cruzada supervisionada. Os metadados voltam separados pra
    interpretar os clusters DEPOIS de formados (ex.: "o cluster 2 é quase
    todo volta de composto macio?"), não pra treinar nada.

    `relative_to_driver=True` centraliza cada feature pela própria média
    do piloto na sessão (`valor - média do piloto`) — mesma correção que
    funcionou no modelo de ritmo (`compute_driver_delta_target`,
    `features/pace.py`). Em valores absolutos, velocidade/RPM médios são
    dominados por qual carro é mais rápido no geral (Red Bull vs. Williams,
    não pneu ou estilo); centralizar por piloto remove essa diferença de
    baseline, isolando a variação que sobra DENTRO do desempenho de cada
    piloto — o que de fato deveria refletir composto/estilo.
    """
    if relative_to_driver:
        source = lap_summary.copy()
        for column in FEATURE_COLUMNS:
            source[column] = lap_summary[column] - lap_summary.groupby("driver")[
                column
            ].transform("mean")
    else:
        source = lap_summary

    features = source[FEATURE_COLUMNS].reset_index(drop=True)
    metadata = lap_summary[["driver", "lap_number", "compound"]].reset_index(drop=True)
    return features, metadata
