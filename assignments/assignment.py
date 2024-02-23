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
    def __init__(self):
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        # Define cluster means and covariance matrices
        mean1 = np.array([0, 5])
        cov1 = np.array([[1, 0.5], [0.5, 2]])
        mean2 = np.array([10, 5])
        cov2 = np.array([[2, -0.5], [-0.5, 1]])

        # Generate samples from each cluster
        X1 = np.random.multivariate_normal(mean1, cov1, size=n_samples // 2)
        X2 = np.random.multivariate_normal(mean2, cov2, size=n_samples // 2)

        # Combine samples and labels
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

        return X, y
