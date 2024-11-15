import numpy as np
import pandas as pd

from .metrics import *
from .base import *


class TreeNode:
    def __init__(self):
        self.samples = None
        self.entropy = None
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None


class BaseDecisionTree:

    # Вывод структуры дерева
    def print_tree(self) -> str:
        # Рекурсивная функция для обхода дерева
        def traverse_print_tree(node, parent=None, depth=1):
            params = f'      (entropy: {node.entropy:.4f}, samples: {node.samples})'
            if node.feature is not None:
                print(f"{'-' * depth}{node.feature} > {node.threshold:.4f}", params)
                if node.left is not None:
                    traverse_print_tree(node.left, 'left', depth + 1)
                if node.right is not None:
                    traverse_print_tree(node.right, 'right', depth + 1)
            else:
                print(f"{'-' * depth} {'leaf_' + parent} = {node.prediction}", params)

        return traverse_print_tree(self.root)

    def __str__(self) -> str:
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'


class MyTreeClf(Base, BaseDecisionTree):

    def __init__(
        self, 
        max_depth=5,
        min_samples_split=2,
        max_leafs=20
    ):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.root = self._build_tree(TreeNode(), X, y, depth=0)

    # def predict_proba(self, X: pd.DataFrame) -> pd.Series:

    def _build_tree(self, node: TreeNode, X: pd.DataFrame, y: pd.Series, depth: int):
        node.samples = len(y)
        node.prediction = y.mode()[0]
        node.entropy = self._entropy(y)
        
        # Условие остановки ветвления: проверяем, достигли ли мы максимальной глубины или минимального числа листьев
        if any((
            depth >= self.max_depth, 
            max(2, self.max_leafs) <= self.leafs_cnt, 
            len(y) < self.min_samples_split, 
            len(np.unique(y)) == 1
        )):
            return node

        # Выбираем наилучший признак и порог для разбиения
        best_feature, best_threshold = self._get_best_split(X, y)
        if best_feature is None or best_threshold is None:
            return node

        node.feature = best_feature
        node.threshold = best_threshold
        self.leafs_cnt += 1

        # Разделяем данные на левую и правую части
        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        # Рекурсивно строим поддеревья для каждого разбиения
        node.left = self._build_tree(TreeNode(), left_X, left_y, depth + 1)
        node.right = self._build_tree(TreeNode(), right_X, right_y, depth + 1)

        return node

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series):
        features = X.columns
        best_entopy = self._entropy(y) # энтропия до разделения
        best_feature = best_threshold = None
    
        for feature in features:
            unique_feature_values = np.unique(X[feature])
            means_thresholds = np.convolve(unique_feature_values, [.5, .5], mode='valid')   # аналог (a[:-1] + a[1:]) / 2
    
            for threshold in means_thresholds:
                left_indices = X[feature] <= threshold
                right_indices = X[feature] > threshold
                left_y, right_y = y[left_indices], y[right_indices]
                len_left, len_right = len(left_y), len(right_y)
                len_total = len_left + len_right

                entropy_split = len_left / len_total * self._entropy(left_y) + \
                                len_right / len_total * self._entropy(right_y)

                # проверяем, уменьшает ли текущее разделение энтропию
                if best_entopy > entropy_split:
                    best_entopy = entropy_split
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    @staticmethod
    def _entropy(sample: pd.Series) -> float:
        eps = 1e-15
        _, counts = np.unique(sample, return_counts=True)
        probabilities = counts / np.sum(counts)
        return -(probabilities @ np.log2(probabilities + eps))

