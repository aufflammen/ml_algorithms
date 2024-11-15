import numpy as np
import pandas as pd

from .metrics import *


class Base:

    @staticmethod
    def _to_numpy_with_bias(X: pd.DataFrame, y: pd.Series=None):
        X = X.to_numpy()
        # Добавляем столбец с единицами для включения bias
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        if y is not None:
            y = y.to_numpy()
            return X, y
        else:
            return X