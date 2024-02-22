import numpy as np
from typing import Any


class PrincipalComponentAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, indices[: self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        class_labels = np.unique(y)
        n_classes = len(class_labels)
        n_features = X.shape[1]

        class_means = np.zeros((n_classes, n_features))
        for i, label in enumerate(class_labels):
            class_means[i] = np.mean(X[y == label], axis=0)

        overall_mean = np.mean(X, axis=0)

        between_class_scatter = np.zeros((n_features, n_features))
        within_class_scatter = np.zeros((n_features, n_features))

        for i, label in enumerate(class_labels):
            n_samples = np.sum(y == label)
            diff = (class_means[i] - overall_mean).reshape(-1, 1)
            between_class_scatter += n_samples * np.dot(diff, diff.T)
            within_class_scatter += np.cov(X[y == label], rowvar=False) * (
                n_samples - 1
            )

        eigenvalues, eigenvectors = np.linalg.eig(
            np.dot(np.linalg.inv(within_class_scatter), between_class_scatter)
        )
        indices = np.argsort(np.abs(eigenvalues.real))[::-1]
        self.components = eigenvectors[:, indices[: self.n_components]].real

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.components)


class AdversarialExamples:
    def __init__(self) -> None:
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        cluster1_mean = [-4, 2]
        cluster2_mean = [2, -5]  # Increased separation between means
        cluster1_covariance = [[1, 0.2], [0.2, 1]]
        cluster2_covariance = [[1, 0.2], [0.2, 1]]

        cluster1_data = np.random.multivariate_normal(
            cluster1_mean, cluster1_covariance, n_samples
        )
        cluster2_data = np.random.multivariate_normal(
            cluster2_mean, cluster2_covariance, n_samples
        )

        X = np.vstack((cluster1_data, cluster2_data))
        y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

        return X, y
