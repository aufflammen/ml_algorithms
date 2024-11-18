import numpy as np
import pandas as pd

from .metrics import Distance


class BaseKNN:

    def __init__(
        self, 
        k=5,
        metric='euclidean',
        weight='uniform'
    ):
        self.k = k
        self.metric = getattr(Distance, metric)
        self.weight = weight
        self.train_size = None
        self.features = None
        self.X_train = None
        self.y_train = None

    def _prepare_data(self, X: pd.DataFrame, expand_dims: int) -> np.ndarray:
        """
        Преобразует тестовые данные в numpy-формат с учетом порядка признаков.
        Добавляеет пустую ось.
        """
        return np.expand_dims(X[self.features].to_numpy(), axis=expand_dims)

    def _prepare_targets(self, y: pd.Series) -> np.ndarray:
        return y.to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_size = X.shape
        self.features = X.columns
        # Валидация параметров
        self._validate_parameters()
        self.X_train = self._prepare_data(X, expand_dims=1)
        self.y_train = self._prepare_targets(y)

    def _get_nearest_targets(self, X_test: np.ndarray) -> (np.ndarray, np.ndarray):
        distances = self.metric(self.X_train, X_test).T
        nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        nearest_targets = self.y_train[nearest_indices]
        # Для корректной индексации многомерным массивом создадим вспомогательный вектор столбец
        row_indices = np.arange(nearest_indices.shape[0])[:, np.newaxis]
        nearest_distances = distances[row_indices, nearest_indices]
        return nearest_targets, nearest_distances

    def _validate_parameters(self) -> None:
        if self.k > self.train_size[0]:
            raise ValueError("Parameter 'k' cannot be greater than the number of training samples.")
        if self.weight not in {'uniform', 'rank', 'distance'}:
            raise ValueError("Parameter 'weight' must be one of {'uniform', 'rank', 'distance'}.")

    def __str__(self) -> str:
        return f'{self.__class__.__name__}: k={self.k}, metric={self.metric.__name__}, weight={self.weight}'


class KNNRegressor(BaseKNN):
    """
    Parameters
    ----------
    k : int, default=5

    metric : {'euclidean', 'manhattan', 'chebyshev', 'cosine'}, default='euclidean'

    weight : {'uniform', 'rank', 'distance'}, default='uniform'
    """

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        n_samples = X_test.shape[0]
        X_test = self._prepare_data(X_test, expand_dims=0)
        nearest_targets, nearest_distances = self._get_nearest_targets(X_test)

        if self.weight == 'uniform':
            pred = np.mean(nearest_targets, axis=1)
    
        elif self.weight == 'rank':
            ranks = 1 / np.arange(1, self.k+1)
            weights = ranks / sum(ranks)
            pred = nearest_targets @ weights

        elif self.weight == 'distance':
            weights = 1 / np.maximum(nearest_distances, 1e-12)
            weights /= np.sum(weights, axis=1)[:, np.newaxis]
            pred = np.einsum('ij,ij->i', nearest_targets, weights) # аналог np.sum(nearest_targets * weights, axis=1)

        return pred


class KNNClassifier(BaseKNN):
    """
    Parameters
    ----------
    k : int, default=5

    metric : {'euclidean', 'manhattan', 'chebyshev', 'cosine'}, default='euclidean'

    weight : {'uniform', 'rank', 'distance'}, default='uniform'
    """

    def _prepare_targets(self, y: pd.Series) -> np.ndarray:
        self.classes = np.unique(y)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        # Метки классов преобразуем в индексы классов
        return np.vectorize(self.class_to_idx.get)(y.to_numpy())

    def predict_proba(self, X_test: pd.DataFrame, full_probs=False) -> np.ndarray:
        n_samples = X_test.shape[0]
        X_test = self._prepare_data(X_test, expand_dims=0)
        nearest_targets, nearest_distances = self._get_nearest_targets(X_test)
        
        if self.weight == 'uniform':
            # Подсчитываем количество классов, принадлежащих ближайшим соседям
            counts = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=self.n_classes),
                axis=1,
                arr=nearest_targets
            )
            proba = counts / self.k
    
        elif self.weight == 'rank':
            weights = 1 / np.arange(1, self.k+1)
            proba = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                for j in range(self.k):
                    proba[i, nearest_targets[i, j]] += weights[j]
            proba /= np.sum(weights)

        elif self.weight == 'distance':
            weights = 1 / np.maximum(nearest_distances, 1e-12)
            proba = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                for j in range(self.k):
                    proba[i, nearest_targets[i, j]] += weights[i, j]
            proba /= np.sum(weights, axis=1)[:, np.newaxis]

        return proba

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        pred_proba = self.predict_proba(X_test)
        return self.classes[np.argmax(pred_proba, axis=1)]