"""Carregamento de sessões de F1 via FastF1, com cache local em disco.

Este é o único módulo do projeto que fala com a rede (servidores de timing
da F1, atráves da biblioteca FastF1). É deliberamente fino: busca a
sessão, garante o cache, e delega toda a normalização de schema para
`f1_ml_garage.data.laps`. Mantẽ-lo fino é o que torna `laps.py` testável
sem rede.
"""

from pathlib import Path

import fastf1
import pandas as pd

from f1_ml_garage.data.laps import normalize_laps

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "f1-ml-garage" / "fastf1"


def enable_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Habilita o cache local de sessões do FastF1.

    Sem cache, toda chamada a `load_session_laps` refaz o download completo
    de sessão - é isso que torna iterar em notebooks/testes viável, e não é
    opcional na prática.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def load_session_laps(year: int, gp: str, session_type: str = "R") -> pd.DataFrame:
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
