"""Carregamento de sessões de F1 via FastF1, com cache local em disco.

Este é o único módulo do projeto que fala com a rede (servidores de timing
da F1, através da biblioteca FastF1). É deliberadamente fino: busca a
sessão, garante o cache, e delega toda a normalização de schema para
`f1_ml_garage.data.laps`. Mantê-lo fino é o que torna `laps.py` testável
sem rede.
"""

from pathlib import Path

import fastf1
import pandas as pd

from f1_ml_garage.data.laps import normalize_laps
from f1_ml_garage.data.results import normalize_results
from f1_ml_garage.data.telemetry import normalize_telemetry

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "f1-ml-garage" / "fastf1"


def enable_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Habilita o cache local de sessões do FastF1.

    Sem cache, toda chamada a `load_session_laps` refaz o download completo
    da sessão — é isso que torna iterar em notebooks/testes viável, e não é
    opcional na prática.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def load_session_laps(
    year: int, gp: str | int, session_type: str = "R"
) -> pd.DataFrame:
    """Carrega e normaliza as voltas de uma sessão de F1.

    Args:
        year: temporada (ex.: 2024).
        gp: nome ou identificador do Grande Prêmio (ex.: "Bahrain", "Monza").
        session_type: "R" (corrida), "Q" (classificação), "S" (sprint),
            "FP1"/"FP2"/"FP3" (treinos livres).
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return normalize_laps(session.laps)


def load_session_results(
    year: int, gp: str | int, session_type: str = "R"
) -> pd.DataFrame:
    """Carrega e normaliza a classificação final de uma sessão de F1.

    Args mesmo formato de `load_session_laps`.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return normalize_results(session.results)


def load_driver_telemetry(
    year: int, gp: str | int, driver: str, session_type: str = "R"
) -> pd.DataFrame:
    """Carrega e normaliza a telemetria completa de um piloto na sessão
    (todas as voltas combinadas, uma linha por amostra).

    Args:
        driver: código de 3 letras (ex.: "VER") ou número do piloto.
        demais args: mesmo formato de `load_session_laps`.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    raw_telemetry = session.laps.pick_drivers(driver).get_telemetry()
    return normalize_telemetry(raw_telemetry)


def load_season_results(year: int, rounds: list[int] | None = None) -> pd.DataFrame:
    """Carrega e concatena a classificação final de várias corridas de uma
    temporada, uma linha por piloto por corrida.

    Uma única corrida tem ~20 pilotos e poucos DNFs — pouca amostra pra
    treinar ou avaliar qualquer classificador. Combinar várias corridas dá
    o volume que o Módulo 2 (árvore de decisão de DNF) precisa.

    Adiciona `round_number` e `event_name` a cada linha — rastreabilidade e
    possível grupo de CV mais adiante (hoje `features/dnf.py` agrupa por
    piloto, não por corrida, mas a coluna fica disponível se isso mudar).

    Args:
        year: temporada.
        rounds: números de rodada específicos a carregar (ex.: `[1, 2, 3]`
            pras 3 primeiras corridas — útil pra testar rápido sem baixar a
            temporada inteira). `None` carrega todas as rodadas de corrida
            do calendário (eventos de teste já vêm excluídos).
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if rounds is not None:
        schedule = schedule[schedule["RoundNumber"].isin(rounds)]

    frames = []
    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        results = load_session_results(year, round_number, "R")
        frames.append(
            results.assign(round_number=round_number, event_name=event["EventName"])
        )

    return pd.concat(frames, ignore_index=True)


def load_session_telemetry(
    year: int, gp: str | int, session_type: str = "R"
) -> pd.DataFrame:
    """Carrega e normaliza a telemetria de TODOS os pilotos de uma sessão,
    concatenada, com a coluna `driver` marcada em cada linha.

    O schema normalizado de `telemetry.py` não inclui `driver` — é
    informação externa ao dado bruto do FastF1 (vem de qual piloto a
    telemetria foi pedida, não de uma coluna do payload). Marcamos aqui,
    no ponto de carregamento, mesmo padrão de `load_season_results`
    marcando `round_number`/`event_name`.

    Volume de dado bem maior que `load_session_laps`/`load_session_results`
    — telemetria é amostrada em alta frequência, então isso baixa (e
    ocupa cache) uma ordem de grandeza a mais de dado que os outros
    loaders desta sessão.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    frames = []
    for driver in session.laps["Driver"].unique():
        raw_telemetry = session.laps.pick_drivers(driver).get_telemetry()
        frames.append(normalize_telemetry(raw_telemetry).assign(driver=driver))

    return pd.concat(frames, ignore_index=True)
