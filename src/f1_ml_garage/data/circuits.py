"""Classificação de circuito (rua vs. permanente) — conhecimento de
domínio estático, não vem do FastF1.

Não existe uma fonte de dado confiável e gratuita pra isso via API, e é
uma característica que praticamente não muda ano a ano pros circuitos
existentes — um dicionário com curadoria manual é uma escolha razoável
aqui, mesmo espírito de `_FINISHED_STATUSES` (`data/results.py`) ou
`COMPOUND_CATEGORIES` (`features/pace.py`): conhecimento fixo, não
calculado a partir de dado nenhum.

Casos de fronteira, documentados pra não esconder a decisão: Miami e
Jeddah (Arábia Saudita) são construídos especificamente pra corrida, com
muros próximos e pouco espaço de escape — características físicas de
circuito de rua — mas às vezes chamados de "híbrido". Contam como rua
aqui, do lado que mais se aproxima do risco de incidente que a feature
tenta capturar (a motivação original, ver `docs/02-dnf-model.md`).
"""

STREET_CIRCUIT_KEYWORDS = (
    "Monaco",
    "Azerbaijan",  # Baku
    "Singapore",
    "Las Vegas",
    "Miami",
    "Saudi Arabian",  # Jeddah
)


def is_street_circuit(event_name: str) -> bool:
    """True se `event_name` (de `load_season_results`, ex.: "Monaco Grand
    Prix") corresponde a um circuito de rua conhecido.

    Checagem por substring, não igualdade exata — nomes de evento variam
    ano a ano (patrocínio, edição especial: "Emilia Romagna Grand Prix"
    virou "Made in Italy Grand Prix" em anos diferentes, por exemplo), mas
    a palavra-chave do país/cidade costuma ser estável. Insensível a
    maiúsculas/minúsculas de propósito — não há motivo pra essa checagem
    depender de capitalização exata.
    """
    normalized = event_name.lower()
    return any(keyword.lower() in normalized for keyword in STREET_CIRCUIT_KEYWORDS)
