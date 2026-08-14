"""Features de estilo de pilotagem por volta, pra clustering (Módulo 4).

Reusa o mesmo resumo por volta de `telemetry_summary.py` que o SVM de
composto usa (`tyre.py`) — as mesmas estatísticas de condução (velocidade,
acelerador, freio, RPM, marcha) que distinguem composto também distinguem
estilo de pilotagem; é a mesma matéria-prima, com uma pergunta diferente.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "mean_speed_kmh",
    "max_speed_kmh",
    "mean_throttle_pct",
    "brake_fraction",
    "mean_rpm",
    "mean_gear",
]


def _detrend_by_driver(lap_summary: pd.DataFrame, column: str) -> pd.Series:
    """Resíduo de `column ~ lap_number`, ajustado SEPARADAMENTE pra cada
    piloto. Resíduos de uma regressão com intercepto sempre têm média
    zero, então isso remove tanto a média do piloto (mesmo efeito de
    `relative_to_driver`) quanto a tendência linear ao longo da corrida
    numa única operação — ver `build_driving_style_features`.
    """

    def residuals(group: pd.DataFrame) -> pd.Series:
        x = group["lap_number"].to_numpy(dtype=float)
        y = group[column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        return pd.Series(y - (slope * x + intercept), index=group.index)

    return lap_summary.groupby("driver", group_keys=False).apply(
        residuals, include_groups=False
    )


def build_driving_style_features(
    lap_summary: pd.DataFrame,
    *,
    relative_to_driver: bool = False,
    detrend_lap_number: bool = False,
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
    dominados por qual carro é mais rápido no geral (Red Bull vs.
    Williams, não pneu ou estilo); centralizar por piloto remove essa
    diferença de baseline, isolando a variação que sobra DENTRO do
    desempenho de cada piloto.

    `detrend_lap_number=True` vai além: ajusta uma regressão linear de
    cada feature contra `lap_number`, separadamente por piloto, e usa os
    resíduos (`_detrend_by_driver`) — remove baseline do piloto E a
    tendência ao longo da corrida (combustível/evolução de pista) numa
    única operação. Motivado pela iteração 5 de
    `docs/04-driving-style-clustering.md`: os clusters encontrados com
    `relative_to_driver` sozinho batiam com FASE DA CORRIDA
    (`lap_number`/stint), não com composto ou estilo em si.

    Limitação real, não escondida: composto e `lap_number` são
    naturalmente confundidos na prática — estratégia amarra os dois
    (stint 1 tende a soft, stint 3 tende a hard). Um detrend linear não
    separa perfeitamente um efeito tipo "degrau" (troca de composto num
    ponto específico) de uma tendência contínua quando os dois estão
    correlacionados no tempo: parte do efeito real de composto pode ser
    absorvida pela reta ajustada, sobrando um resíduo atenuado, não o
    efeito completo. Ainda assim, o que sobrar é mais interpretável como
    "estilo/composto" do que os valores brutos ou só centralizados por
    piloto.
    """
    if detrend_lap_number:
        source = lap_summary.copy()
        for column in FEATURE_COLUMNS:
            source[column] = _detrend_by_driver(lap_summary, column)
    elif relative_to_driver:
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
