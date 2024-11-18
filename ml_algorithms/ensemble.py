import numpy as np
import pandas as pd
import random
from collections import Counter

from .decision_tree import DecisionTreeRegressor, DecisionTreeClassifier
from .metrics import ClassificationMetric, RegressionMetric

class BaseRandomForest:

    def __init__(
        self, 
        n_estimators=10,
        max_features=.5,
        max_samples=.5,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.leafs_cnt = 0
        self.fi = None # Feature importance
        self.ensemble = []

    def __str__(self):
        valid_params = set([
            'n_estimators', 
            'max_features', 
            'max_samples',  
            'max_depth', 
            'min_samples_split', 
            'max_leafs', 
            'bins',
            'random_state'
            ])

        params_print = ', '.join(
            f'{k}={v}' for k, v in self.__dict__.items() if k in valid_params
        )
        return f'{self.__class__.__name__} class: {params_print}'


#RandomForestRegressor
class MyForestReg(BaseRandomForest):

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        n_samples, n_features = X.shape
        features = X.columns.to_list()
        self.fi = Counter({k: 0 for k in X.columns})
        # Рассчитываем количество объектов и фичей, 
        # на которых будет обучаться каждое отдельное дерево
        cnt_cols = round(n_features * self.max_features)
        cnt_rows = round(n_samples * self.max_samples)
        random.seed(self.random_state)

        for _ in range(self.n_estimators):
            col_indices = random.sample(features, cnt_cols)
            row_indices = random.sample(range(n_samples), cnt_rows)
            model = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split, 
                max_leafs=self.max_leafs, 
                bins=self.bins
            )
            # Переопределяем параметр n_samples в каждом дереве
            # для корректного подсчета важности фичей
            model.n_samples = n_samples
            model.fit(X.loc[row_indices, col_indices], y.loc[row_indices])
            self.leafs_cnt += model.leafs_cnt
            self.fi += Counter(model.fi)
            self.ensemble.append(model)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.zeros(X.shape[0])
        for model in self.ensemble:
            pred += model.predict(X)
        pred /= self.n_estimators
        
        return pred
