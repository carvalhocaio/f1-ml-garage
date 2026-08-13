import numpy as np
import pandas as pd
import pytest

from f1_ml_garage.models.dnf import evaluate_dnf_model

N_DRIVERS = 20
RACES_PER_DRIVER = 6
TEAMS = ("Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "TeamX")
HIGH_RISK_TEAM = "TeamX"
HIGH_RISK_GRID = 18.0


def _separable_dnf_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset sintético com uma regra determinística (sem ruído) por
    construção: DNF se o grid é >= 18 OU o time é "TeamX". Serve de
    oráculo — uma árvore rasa o suficiente pra representar um OR de duas
    condições (profundidade 2 já basta) tem que recuperar isso quase
    perfeitamente.
    """
    rows = []
    for driver_idx in range(N_DRIVERS):
        team = TEAMS[driver_idx % len(TEAMS)]
        for race in range(RACES_PER_DRIVER):
            grid_position = float(((driver_idx + race) % 20) + 1)
            dnf = (grid_position >= HIGH_RISK_GRID) or (team == HIGH_RISK_TEAM)
            rows.append(
                {
                    "driver": f"D{driver_idx}",
                    "team": team,
                    "grid_position": grid_position,
                    "dnf": dnf,
                }
            )

    data = pd.DataFrame(rows)
    features = pd.concat(
        [data[["grid_position"]], pd.get_dummies(data["team"], prefix="team")],
        axis=1,
    )
    return features, data["dnf"], data["driver"]


@pytest.mark.unit
def test_recovers_known_separable_pattern():
    features, target, groups = _separable_dnf_dataset()

    result = evaluate_dnf_model(features, target, groups, n_splits=4)

    assert result["recall"] == pytest.approx(1.0, abs=1e-6)
    assert result["precision"] == pytest.approx(1.0, abs=1e-6)
    assert result["f1"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_returns_expected_metric_keys():
    features, target, groups = _separable_dnf_dataset()
    result = evaluate_dnf_model(features, target, groups, n_splits=4)

    assert set(result.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "dnf_rate",
    }


def _noisy_imbalanced_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset sintético com ruído e desbalanceamento real (~11% DNF): grid
    alto AUMENTA a chance de DNF, não garante — diferente do dataset
    separável acima, de propósito, pra expor a diferença entre
    `class_weight="balanced"` e sem peso nenhum.
    """
    rng = np.random.default_rng(42)
    n = 200
    grid_position = rng.integers(1, 21, size=n).astype(float)
    teams = np.array(["Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "Haas"])
    team = rng.choice(teams, size=n)
    dnf_probability = np.where(grid_position >= 15, 0.35, 0.05)
    dnf = pd.Series(rng.random(n) < dnf_probability)

    features = pd.concat(
        [
            pd.Series(grid_position, name="grid_position"),
            pd.get_dummies(pd.Series(team), prefix="team"),
        ],
        axis=1,
    )
    groups = pd.Series([f"D{i}" for i in range(n)])
    return features, dnf, groups


@pytest.mark.unit
def test_class_weight_balanced_improves_recall_on_imbalanced_data():
    """A lição de "dados desbalanceados" do currículo, em números: sem
    peso de classe, o modelo prefere prever "não abandona" quase sempre —
    accuracy parece boa, mas recall (a métrica que importa pra um evento
    raro) desmorona. `class_weight="balanced"` troca uma fatia de accuracy
    por muito mais recall — o trade-off certo quando o custo de não
    detectar um DNF é maior que o de um falso positivo.
    """
    features, target, groups = _noisy_imbalanced_dataset()

    balanced = evaluate_dnf_model(
        features, target, groups, n_splits=5, class_weight="balanced"
    )
    unweighted = evaluate_dnf_model(
        features, target, groups, n_splits=5, class_weight=None
    )

    assert balanced["recall"] > 0.5
    assert unweighted["recall"] < 0.3
    assert balanced["recall"] > unweighted["recall"]
