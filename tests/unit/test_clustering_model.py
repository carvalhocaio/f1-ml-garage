import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from f1_ml_garage.models.clustering import (
    evaluate_clustering,
    fit_kmeans,
    fit_pca,
    standardize_features,
)

N_PER_CLUSTER = 40

# Três "estilos de pilotagem" sintéticos bem separados, mesmo espírito
# oráculo dos outros modelos: se PCA+k-means funcionam direito, têm que
# recuperar essa estrutura quase perfeitamente (ARI ~= 1.0).
CLUSTERS = {
    0: {
        "mean_speed_kmh": 290,
        "max_speed_kmh": 310,
        "mean_throttle_pct": 85,
        "brake_fraction": 0.15,
        "mean_rpm": 11800,
        "mean_gear": 6.5,
    },
    1: {
        "mean_speed_kmh": 250,
        "max_speed_kmh": 270,
        "mean_throttle_pct": 55,
        "brake_fraction": 0.35,
        "mean_rpm": 10200,
        "mean_gear": 4.5,
    },
    2: {
        "mean_speed_kmh": 270,
        "max_speed_kmh": 290,
        "mean_throttle_pct": 65,
        "brake_fraction": 0.25,
        "mean_rpm": 11200,
        "mean_gear": 5.5,
    },
}
NOISE_STD = {
    "mean_speed_kmh": 5,
    "max_speed_kmh": 5,
    "mean_throttle_pct": 4,
    "brake_fraction": 0.02,
    "mean_rpm": 150,
    "mean_gear": 0.2,
}


def _separable_style_dataset() -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.default_rng(0)
    rows, truth = [], []
    for cluster_id, base in CLUSTERS.items():
        for _ in range(N_PER_CLUSTER):
            rows.append(
                {key: base[key] + rng.normal(0, NOISE_STD[key]) for key in base}
            )
            truth.append(cluster_id)
    return pd.DataFrame(rows), truth


@pytest.mark.unit
def test_standardize_features_centers_and_scales():
    features, _ = _separable_style_dataset()
    scaled = standardize_features(features)

    assert scaled.mean(axis=0) == pytest.approx(np.zeros(features.shape[1]), abs=1e-8)
    assert scaled.std(axis=0) == pytest.approx(np.ones(features.shape[1]), abs=1e-8)


@pytest.mark.unit
def test_fit_pca_explained_variance_sums_to_at_most_one():
    features, _ = _separable_style_dataset()
    scaled = standardize_features(features)
    pca, coords = fit_pca(scaled, n_components=2)

    assert coords.shape == (len(features), 2)
    assert pca.explained_variance_ratio_.sum() <= 1.0
    # componente 1 explica mais variância que o 2 (ordem decrescente, por
    # definição de PCA)
    assert pca.explained_variance_ratio_[0] >= pca.explained_variance_ratio_[1]


@pytest.mark.unit
def test_kmeans_recovers_known_clusters():
    features, truth = _separable_style_dataset()
    scaled = standardize_features(features)
    _, coords = fit_pca(scaled, n_components=2)
    _, labels = fit_kmeans(coords, n_clusters=3)

    # adjusted_rand_score compara partições ignorando o número/ordem dos
    # rótulos (o cluster "0" do k-means não precisa ser o "0" da verdade
    # sintética, só precisa agrupar os mesmos pontos juntos)
    assert adjusted_rand_score(truth, labels) > 0.9


@pytest.mark.unit
def test_evaluate_clustering_returns_silhouette_in_valid_range():
    features, _ = _separable_style_dataset()
    scaled = standardize_features(features)
    _, coords = fit_pca(scaled, n_components=2)
    _, labels = fit_kmeans(coords, n_clusters=3)

    result = evaluate_clustering(coords, labels)

    assert set(result.keys()) == {"silhouette_score"}
    assert -1.0 <= result["silhouette_score"] <= 1.0
    # clusters bem separados por construção -> silhouette alto
    assert result["silhouette_score"] > 0.5
