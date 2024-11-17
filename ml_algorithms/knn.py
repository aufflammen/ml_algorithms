import numpy as np
import pandas as pd

from .metrics import Distance


class BaseKNN:

    def _get_nearest_targets(self, X_test: np.ndarray) -> (np.ndarray, np.ndarray):
        distances = self.metric(self.X_train, X_test).T
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_targets = self.y_train[k_nearest_indices]
        # Для корректной индексации многомерным массивом создадим вспомогательный вектор столбец
        row_indices = np.arange(k_nearest_indices.shape[0])[:, np.newaxis]
        k_nearest_distances = distances[row_indices, k_nearest_indices]
        return k_nearest_targets, k_nearest_distances

    def __str__(self):
        return f'{self.__class__.__name__} class: k={self.k}'


# KNNRegressor
class MyKNNReg(BaseKNN):

    def __init__(self, k=3):
        self.k = k


class KNNClassifier(BaseKNN):
    """
    Parameters
    ----------
    k : int, default=5

    metric : {'euclidean', 'manhattan', 'chebyshev', 'cosine'}, default='euclidean'

    weight : {'uniform', 'rank', 'distance'}, default='uniform'
    """

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
        self.classes = None
        self.class_to_idx = None
        self.n_classes = None
        self.X_train = None
        self.y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_size = X.shape
        self.features = X.columns
        self.classes = np.unique(y)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.n_classes = len(self.classes)

        if self.k > self.train_size[0]:
            raise ValueError("Parameter 'k' cannot be greater than the number of training samples.")

        # Добавляем пустую ось
        self.X_train = np.expand_dims(X.to_numpy(), axis=1)
        # Метки классов преобразуем в индексы классов
        self.y_train = np.vectorize(self.class_to_idx.get)(y.to_numpy())

    def predict_proba(self, X_test: pd.DataFrame, full_probs=False) -> np.ndarray:
        n_samples = X_test.shape[0]
        # Устанавливаем такой же порядок признаков, как в обучающей выборке,
        # и добавляем пустую ось
        X_test = np.expand_dims(X_test[self.features].to_numpy(), axis=0)
        k_nearest_targets, k_nearest_distances = self._get_nearest_targets(X_test)
        
        if self.weight == 'uniform':
            # Подсчитываем количество классов, принадлежащих ближайшим соседям
            counts = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=self.n_classes),
                axis=1,
                arr=k_nearest_targets
            )
            proba = counts / self.k
    
        elif self.weight == 'rank':
            weights = 1 / np.arange(1, self.k+1)
            proba = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                for j in range(self.k):
                    proba[i, k_nearest_targets[i, j]] += weights[j]
            proba /= np.sum(weights)

        elif self.weight == 'distance':
            weight = 1 / (k_nearest_distances + 1e-12)
            proba = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                for j in range(self.k):
                    proba[i, k_nearest_targets[i, j]] += weight[i, j]
            proba /= np.sum(weight, axis=1)[:, np.newaxis]

        else:
            raise ValueError("Parameter 'weight' can be {'uniform', 'rank', 'distance'}")

        return proba

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        pred_proba = self.predict_proba(X_test)
        return self.classes[np.argmax(pred_proba, axis=1)]
