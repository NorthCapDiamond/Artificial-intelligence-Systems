import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, children=None, value=None, typenode=None, threshold=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.value = value
        self.typenode = typenode

    def set_leaf(self):
        self.typenode = "leaf"

    def is_leaf(self):
        return self.typenode == "leaf"


class DecisionTree:
    def __init__(self, max_depth=10, min_features_split=1, min_samples_split=10, n_features=None):
        self.max_depth = max_depth
        self.min_features_split = min_features_split
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def _create_path(self, X, y, current_depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Checking for ending criteria:
        if (n_samples < self.min_samples_split or current_depth >= self.max_depth or n_feats < self.min_features_split or n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, typenode="leaf")

        # Else
        # We're gonna split all our values and check their quality later...
        feats_idx = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresholds = self._best_split(X, y, feats_idx)
        children_split = self._split(X[:, best_feature], best_thresholds)
        children = []

        for i, el in enumerate(children_split):
            children.append(self._create_path(
                X[el, :], y[el], current_depth + 1))

        return Node(feature=best_feature, threshold=best_thresholds, children=children)

    def _best_split(self, X, y, feats_idx):
        best_gain = - 101
        split_idxs = None
        split_thresholds = None

        for feat_idx in feats_idx:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            gain = self._information_gain(y, X_column, thresholds)

            if gain > best_gain:
                best_gain = gain
                split_thresholds = thresholds
                split_idxs = feat_idx

        return split_idxs, split_thresholds

    def _split(self, X_column, thresholds):
        split_data = []
        for i, j in enumerate(thresholds):
            tmp = (np.argwhere(X_column == j).flatten())
            if len(tmp) > 0:
                split_data.append(tmp)
        return split_data.copy()

    def _most_common_label(self, y):
        if (not len(y)):
            print("0 length detected returned 0")
            return 0
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _information_gain(self, y, X_column, thresholds):
        # parent entropy
        parent_entropy = self._entropy(y)
        n = len(y)
        n_children = []
        e_children = []
        child_entropy = 0

        # create children
        children = self._split(X_column, thresholds)

        for i, el in enumerate(children):
            if not len(el):
                return 0
            n_children.append(len(el))
            e_children.append(self._entropy(y[el]))

            child_entropy += n_children[i] / n * e_children[i]

        # calculate the IG

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def fit(self, X, y):
        if(self.n_features):
            self.n_features = min(X.shape[1], self.n_features)
        else:
            self.n_features = X.shape[1]

        self.root = self._create_path(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        child_node = None
        for child in node.children:
            if x[node.feature] == child.feature:
                child_node = child
                break

        if child_node:
            return self._traverse_tree(x, child_node)
        return self._traverse_tree(x, node.children[0])
