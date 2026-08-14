"""Telemetria por amostra -> vetor de features por volta.

Extraído de `features/tyre.py`: associar telemetria a voltas a agregar em
estatísticas por volta não é específico de classificar composto - o
Módulo 4 (clustering de estilo de pilotagem) precisa exatamente da mesma
transformação, só que sem alvo nenhum. Um só lugar para essa lógica, dois
consumidores (`tyre.py`, `driving_style.py`).
"""

import pandas as pd


def tag_telemetry_with_lap(telemetry: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Associa cada amostra de telemetria à volta correspondente.

    `session_time_s` de uma volta (`laps.py`) marca o FIM daquela volta -
    não tem coluna de "início". Uma amostra de telemetria pertence à
    primeira volta cujo fim (`session_time_s`) é >= o `session_time_s` da
    amostra - daí `direction="forward" no `merge_asof` (join "as of" por
    tempo, não por igualdade exata, já que telemetria e voltas não
    compartilham timestamps exatos).

    `by="driver" é essencial: telemetrias de pilotos diferentes têm
    `session_time_s` sobrepostos (é tempo de sessão, não de volta
    individual) - sem isso, a amostra de um piloto poderia ser associada à
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
    canais de carro - média/máximo de velocidade, marcha média, média de
    RPM, % de acelerador médio, fração da volta freando. Não é feature
    engineering exaustivo (Módulo 5 fica par isso); é o suficiente para dar
    sinal tanto pro SVM de composto quanto pro clustering de estilo de
    pilotagem (pneu mais mole ou estilo mais agressivo tendem a refletir
    nos mesmos canais - frenagem/aceleração).
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
