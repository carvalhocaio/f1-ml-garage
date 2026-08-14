"""PCA + k-means/GMM pra explorar estilo de pilotagem via telemetria
(Módulo 4).

Diferente dos modelos supervisionados do Módulo 2, clustering não tem alvo
pra validar contra — o fluxo de trabalho é outro: escalar, reduzir
dimensionalidade (PCA), agrupar (k-means ou GMM), e interpretar os grupos
contra alguma variável conhecida (aqui, `compound`) como checagem de
sanidade, não como alvo de treino. Por isso funções separadas e compostas,
em vez de um único pipeline `fit`/`evaluate` como nos modelos
supervisionados — o trabalho real de clustering é inspecionar cada etapa
(quanta variância o PCA explica, qual k faz sentido) antes de seguir pra
próxima.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DEFAULT_N_COMPONENTS = 2
DEFAULT_N_CLUSTERS = 3
DEFAULT_RANDOM_STATE = 0


def standardize_features(features: pd.DataFrame) -> np.ndarray:
    """Escala as features antes de PCA/k-means.

    As duas técnicas dependem de distância euclidiana — mesmo motivo do
    `StandardScaler` no SVM de composto (`models/tyre.py`): sem escalar,
    `mean_rpm` (~11000) dominaria a distância sozinho.
    """
    return StandardScaler().fit_transform(features)


def fit_pca(
    scaled_features: np.ndarray, n_components: int = DEFAULT_N_COMPONENTS
) -> tuple[PCA, np.ndarray]:
    """Ajusta PCA e retorna o objeto ajustado — pra inspecionar
    `.explained_variance_ratio_` antes de decidir se `n_components` faz
    sentido — junto com as coordenadas transformadas.
    """
    pca = PCA(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    coords = pca.fit_transform(scaled_features)
    return pca, coords


def fit_kmeans(
    coords: np.ndarray, n_clusters: int = DEFAULT_N_CLUSTERS
) -> tuple[KMeans, np.ndarray]:
    """Ajusta k-means sobre as coordenadas (tipicamente já reduzidas por
    PCA) e retorna o modelo ajustado junto com o rótulo de cluster de cada
    linha.
    """
    model = KMeans(
        n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init="auto"
    )
    labels = model.fit_predict(coords)
    return model, labels


def evaluate_clustering(coords: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette score: quão bem separados os clusters ficam (-1 a 1,
    maior é melhor; perto de 0 indica clusters sobrepostos).

    Não precisa de rótulo verdadeiro — é a métrica padrão pra clustering
    quando não existe "certo"/"errado" a priori, diferente dos modelos
    supervisionados do Módulo 2.
    """
    return {"silhouette_score": float(silhouette_score(coords, labels))}


def fit_gmm(
    coords: np.ndarray, n_components: int = DEFAULT_N_CLUSTERS
) -> tuple[GaussianMixture, np.ndarray]:
    """Ajusta um GMM (Gaussian Mixture Model, via EM) sobre as coordenadas
    e retorna o modelo ajustado junto com o rótulo mais provável de cada
    linha (`fit_predict`, atribuição rígida — pra atribuição probabilística,
    ver `gmm_component_probabilities`).

    Diferença real em relação a `fit_kmeans`, não só troca de biblioteca:
    k-means atribui cada ponto a exatamente um cluster (partição rígida,
    fronteiras implicitamente esféricas); GMM modela cada componente como
    uma distribuição gaussiana (com sua própria forma/orientação) e
    calcula a PROBABILIDADE de cada ponto pertencer a cada uma —
    atribuição suave. Relevante aqui porque o k-means encontrou dois
    clusters quase idênticos em composição de composto
    (`docs/04-driving-style-clustering.md`, iteração 3) — se essa
    proximidade é real (não uma partição forçada de algo que não deveria
    ser dividido), o GMM deveria mostrar probabilidades ambíguas (perto de
    50/50) entre esses dois componentes pra várias voltas.
    """
    model = GaussianMixture(
        n_components=n_components, random_state=DEFAULT_RANDOM_STATE
    )
    labels = model.fit_predict(coords)
    return model, labels


def gmm_component_probabilities(
    model: GaussianMixture, coords: np.ndarray
) -> np.ndarray:
    """Probabilidade de cada ponto pertencer a cada componente do GMM —
    o que de fato distingue GMM de k-means na prática (atribuição suave,
    não só um rótulo). Uma coluna por componente, uma linha por ponto,
    linhas somando 1.
    """
    return model.predict_proba(coords)


def select_n_components_by_bic(
    coords: np.ndarray, candidates: list[int]
) -> dict[int, float]:
    """BIC (Bayesian Information Criterion) para cada número de
    componentes candidato — o critério padrão pra escolher `k` num GMM,
    penalizando complexidade (mais componentes = mais parâmetros livres)
    contra qualidade do ajuste (log-verossimilhança). Menor BIC é melhor.

    Diferente do "método do cotovelo" usado informalmente com k-means
    (olhar um gráfico e decidir visualmente), BIC dá um número direto e
    comparável entre diferentes `k` — o candidato com menor BIC é a
    escolha objetiva, não uma leitura visual.
    """
    scores = {}
    for n in candidates:
        model = GaussianMixture(n_components=n, random_state=DEFAULT_RANDOM_STATE)
        model.fit(coords)
        scores[n] = float(model.bic(coords))
    return scores
