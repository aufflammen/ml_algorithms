import numpy as np
import pandas as pd

from .metrics import *

# KNNRegressor
# KNNClassifier

class Distance:

    @staticmethod
    def euclidean(u, v):
        return np.sqrt(np.sum(np.square(u - v), axis=-1))

    @staticmethod
    def manhattan(u, v):
        return np.sum(np.abs(u - v), axis=-1)

    @staticmethod
    def chebyshev(u, v):
        return np.max(np.abs(u - v), axis=-1)

    @staticmethod
    def cosine(u, v):
        l2_norm = lambda x: np.sqrt(np.sum(np.square(x), axis=-1))
        return 1 - np.sum(u * v, axis=-1) / (l2_norm(u) * l2_norm(v))


class BaseKNN:

    def _get_nearest_targets_idx(self, X_test: np.ndarray):
        metric = getattr(Distance, self.metric)
        distances = metric(X_test, self.X_train)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_targets = self.y_train[k_nearest_indices]
        # Преобразуем метки классов в индексы классов
        k_nearest_targets_idx = np.vectorize(self.class_to_idx.get)(k_nearest_targets)
        return k_nearest_targets_idx


class MyKNNClf(BaseKNN):
    """
    Parameters
    ----------
    k : int, default=3

    metric : {'euclidean', 'manhattan', 'chebyshev', 'cosine'}, default='euclidean'

    weight : {'uniform', 'rank', 'distance'}, default='uniform'
    """

    def __init__(
        self, 
        k=3, 
        metric='euclidean',
        weight='uniform'
    ):
        self.k = k
        self.metric = metric
        sefl.weight = weight
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
        self.class_to_idx = {v: k for k, v in enumerate(self.classes)}
        self.n_classes = len(self.classes)

        if self.k > self.train_size[0]:
            raise ValueError("Parameter k cannot be greater than the number of training samples.")

        self.X_train = np.expand_dims(X.to_numpy(), axis=0)
        self.y_train = y.to_numpy()

    def predict_proba(self, X_test: pd.DataFrame, full_probs=False):
        X_test = np.expand_dims(X_test[self.features].to_numpy(), axis=1)
        k_nearest_targets_idx = self._get_nearest_targets_idx(X_test)
        # Подсчетываем количество классов, принадлежащих ближайшим соседям
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),
            axis=1,
            arr=k_nearest_targets_idx
        )
        return (counts / self.k)[:, 1]

    def predict(self, X_test: pd.DataFrame):
        # pred_proba = self.predict_proba(X_test)
        # return self.classes[np.argmax(pred_proba, axis=1)]
        return (self.predict_proba(X_test) >= .5).astype(int)

    def __str__(self):
        return f'{self.__class__.__name__} class: k={self.k}'