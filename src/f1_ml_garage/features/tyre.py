"""Features para classificar composto de pneu a partir de telemetria
(Módulo 2, SVM + kernels).

Telemetria vem em amostras de alta frequência (uma linha por instante), não
uma linha por volta. Este módulo faz a ponte: associa cada amostra à volta
correspondente, agrega em um vetor de features por volta, e liga ao alvo
(`compound`) que já vem de `laps.py`.
"""

import pandas as pd


def tag_telemetry_with_lap(telemetry: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Associa cada amostra de telemetria à volta correspondente.

    `session_time_s` de uma volta (`laps.py`) marca o FIM daquela volta —
    não tem coluna de "início". Uma amostra de telemetria pertence à
    primeira volta cujo fim (`session_time_s`) é >= o `session_time_s` da
    amostra — daí `direction="forward"` no `merge_asof` (join "as of" por
    tempo, não por igualdade exata, já que telemetria e voltas não
    compartilham timestamps exatos).

    `by="driver"` é essencial: telemetrias de pilotos diferentes têm
    `session_time_s` sobrepostos (é tempo de sessão, não de volta
    individual) — sem isso, a amostra de um piloto poderia ser associada à
    volta de outro.
    """
    return pd.merge_asof(
        telemetry.sort_values("session_time_s"),
        laps[["driver", "lap_number", "session_time_s", "compound"]].sort_values(
            "session_time_s"
        ),
        on="session_time_s",
        by="driver",
        direction="forward",
    )


def summarize_lap_telemetry(tagged_telemetry: pd.DataFrame) -> pd.DataFrame:
    """Agrega amostras de telemetria por volta em um vetor de features.

    Uma linha por (`driver`, `lap_number`), com estatísticas simples dos
    canais de carro — média/máximo de velocidade, marcha média, média de
    RPM, % de acelerador médio, fração da volta freando. Não é feature
    engineering exaustivo (Módulo 5 fica pra isso); é o suficiente pra dar
    ao SVM alguma coisa que distingue estilo de condução por composto
    (pneu mais mole tende a mais tração/menos deslizamento, refletido em
    frenagem/aceleração diferentes).
    """
    grouped = tagged_telemetry.groupby(["driver", "lap_number"], as_index=False)
    summary = grouped.agg(
        compound=("compound", "first"),
        mean_speed_kmh=("speed_kmh", "mean"),
        max_speed_kmh=("speed_kmh", "max"),
        mean_throttle_pct=("throttle_pct", "mean"),
        brake_fraction=("brake", "mean"),
        mean_rpm=("rpm", "mean"),
        mean_gear=("gear", "mean"),
    )
    return summary


def build_tyre_features(
    lap_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Monta a matriz de features (X), o alvo (y) e os grupos (piloto) a
    partir da saída de `summarize_lap_telemetry`.

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
