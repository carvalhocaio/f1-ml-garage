"""Telemetria por amostra -> vetor de features por volta.

Extraído de `features/tyre.py`: associar telemetria a voltas e agregar em
estatísticas por volta não é específico de classificar composto — o
Módulo 4 (clustering de estilo de pilotagem) precisa exatamente da mesma
transformação, só que sem alvo nenhum. Um só lugar pra essa lógica, dois
consumidores (`tyre.py`, `driving_style.py`).
"""

import pandas as pd

from f1_ml_garage.data.laps import select_green_flag_laps


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


def filter_to_clean_laps(
    tagged_telemetry: pd.DataFrame, laps: pd.DataFrame
) -> pd.DataFrame:
    """Restringe amostras de telemetria já marcadas por volta
    (`tag_telemetry_with_lap`) às voltas de bandeira verde limpa
    (`select_green_flag_laps`) — descarta telemetria de entrada/saída de
    pit, safety car, VSC, e voltas não cronometradas com confiança.

    Filtra DEPOIS de marcar, nunca antes: se `laps` fosse filtrado antes
    do `merge_asof` em `tag_telemetry_with_lap`, uma amostra que pertence a
    uma volta removida (ex.: sob safety car) seria incorretamente
    associada à PRÓXIMA volta que sobrou na tabela — o `merge_asof`
    `direction="forward"` não sabe que aquela volta foi removida de
    propósito, só vê o próximo limite disponível e gruda nele. Filtrar
    depois, pela chave (`driver`, `lap_number`) via `merge` com
    `how="inner"`, descarta a amostra corretamente em vez de realocá-la
    pra volta errada.

    Sem esse filtro, telemetria de voltas anômalas (velocidade de pit
    lane, bunching atrás de safety car) mistura com telemetria de corrida
    normal — contaminação que pesa tanto pro SVM de composto
    (`models/tyre.py`) quanto pro clustering de estilo de pilotagem
    (`models/clustering.py`).
    """
    clean_laps = select_green_flag_laps(laps)[["driver", "lap_number"]]
    return tagged_telemetry.merge(clean_laps, on=["driver", "lap_number"], how="inner")


def summarize_lap_telemetry(tagged_telemetry: pd.DataFrame) -> pd.DataFrame:
    """Agrega amostras de telemetria por volta em um vetor de features.

    Uma linha por (`driver`, `lap_number`), com estatísticas simples dos
    canais de carro — média/máximo de velocidade, marcha média, média de
    RPM, % de acelerador médio, fração da volta freando. Não é feature
    engineering exaustivo (Módulo 5 fica pra isso); é o suficiente pra dar
    sinal tanto pro SVM de composto quanto pro clustering de estilo de
    pilotagem (pneu mais mole ou estilo mais agressivo tendem a refletir
    nos mesmos canais — frenagem/aceleração).
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
