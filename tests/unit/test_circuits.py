import pytest

from f1_ml_garage.data.circuits import is_street_circuit


@pytest.mark.unit
@pytest.mark.parametrize(
    "event_name",
    [
        "Monaco Grand Prix",
        "Azerbaijan Grand Prix",
        "Singapore Grand Prix",
        "Las Vegas Grand Prix",
        "Miami Grand Prix",
        "Saudi Arabian Grand Prix",
    ],
)
def test_is_street_circuit_true_for_known_street_circuits(event_name):
    assert is_street_circuit(event_name)


@pytest.mark.unit
@pytest.mark.parametrize(
    "event_name",
    [
        "Bahrain Grand Prix",
        "Italian Grand Prix",
        "British Grand Prix",
        "São Paulo Grand Prix",
    ],
)
def test_is_street_circuit_false_for_permanent_circuits(event_name):
    assert not is_street_circuit(event_name)


@pytest.mark.unit
def test_is_street_circuit_matches_by_substring_not_exact_equality():
    """Nomes de evento variam ano a ano (patrocínio, edição especial) —
    a checagem tem que sobreviver a isso, não exigir igualdade exata."""
    assert is_street_circuit("Monaco Grand Prix 2024")
    assert is_street_circuit("70th Anniversary Monaco Grand Prix")


@pytest.mark.unit
def test_is_street_circuit_is_case_insensitive():
    assert is_street_circuit("FORMULA 1 MONACO GRAND PRIX 2024")
    assert is_street_circuit("monaco grand prix")


@pytest.mark.unit
def test_is_street_circuit_unknown_event_defaults_to_false():
    assert not is_street_circuit("Some Future Grand Prix Nobody Has Heard Of")
