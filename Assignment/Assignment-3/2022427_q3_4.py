import numpy as np
import pandas as pd

def gini(y):
    m = y.shape[0]
    if m == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / m
    return 1 - np.sum(probs ** 2)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        if (self.max_depth is not None and depth >= self.max_depth) or \
           (num_samples < self.min_samples_split) or \
           (len(unique_classes) == 1):
            return self._create_leaf_node(y)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return self._create_leaf_node(y)

        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs

        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left, 'right': right}

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        selected_features = np.random.choice(X.shape[1], self.max_features, replace=False)

        for feature in selected_features:
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)
            if len(unique_values) < 2:
                continue
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            for threshold in thresholds:
                left_idxs = feature_values <= threshold
                if np.sum(left_idxs) == 0 or np.sum(left_idxs) == len(feature_values):
                    continue
                left_gini = gini(y[left_idxs])
                right_gini = gini(y[~left_idxs])
                weighted_gini = (left_gini * np.sum(left_idxs) + right_gini * np.sum(~left_idxs)) / len(y)
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _create_leaf_node(self, y):
        counts = np.bincount(y)
        return {'value': np.argmax(counts)}

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if 'value' in node:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])
        
    #DecisionTree class for visualization
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree
        if 'value' in node:
            print(f"{indent}[Leaf] Value = {'Yes' if node['value'] == 1 else 'No'}")
        else:
            print(f"{indent}Feature {node['feature']} <= {node['threshold']:.2f}?")
            print(f"{indent}├── True:", end="")
            self.print_tree(node['left'], indent + "│   ")
            print(f"{indent}└── False:", end="")
            self.print_tree(node['right'], indent + "    ")

class Bagging:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.oob_indices = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.trees = []
        self.oob_indices = []

        for _ in range(self.n_trees):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_mask = np.zeros(n_samples, dtype=bool)
            oob_mask[bootstrap_indices] = True
            oob_indices = np.where(~oob_mask)[0]
            self.oob_indices.append(oob_indices)

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def compute_oob_error(self, X, y):
        n_samples = X.shape[0]
        oob_preds = [[] for _ in range(n_samples)]

        for tree_idx, tree in enumerate(self.trees):
            oob = self.oob_indices[tree_idx]
            if len(oob) == 0:
                continue
            preds = tree.predict(X[oob])
            for i, pred in zip(oob, preds):
                oob_preds[i].append(pred)

        y_pred = []
        for i in range(n_samples):
            if len(oob_preds[i]) == 0:
                y_pred.append(np.random.choice([0, 1]))
            else:
                y_pred.append(np.argmax(np.bincount(oob_preds[i])))
        y_pred = np.array(y_pred)
        return np.mean(y_pred != y)

# Dataset
data = [
    [25, 'High', 'No', 'Fair', 'No'],
    [30, 'High', 'No', 'Excellent', 'No'],
    [35, 'Medium', 'No', 'Fair', 'Yes'],
    [40, 'Low', 'No', 'Fair', 'Yes'],
    [45, 'Low', 'Yes', 'Fair', 'Yes'],
    [50, 'Low', 'Yes', 'Excellent', 'No'],
    [55, 'Medium', 'Yes', 'Excellent', 'Yes'],
    [60, 'High', 'No', 'Fair', 'No']
]
columns = ['Age', 'Income', 'Student', 'Credit Rating', 'Buy Computer']
df = pd.DataFrame(data, columns=columns)

# Encode data
income_map = {'High': 2, 'Medium': 1, 'Low': 0}
student_map = {'Yes': 1, 'No': 0}
credit_map = {'Fair': 0, 'Excellent': 1}
buy_map = {'Yes': 1, 'No': 0}

df['Income'] = df['Income'].map(income_map)
df['Student'] = df['Student'].map(student_map)
df['Credit Rating'] = df['Credit Rating'].map(credit_map)
df['Buy Computer'] = df['Buy Computer'].map(buy_map)

X = df[['Age', 'Income', 'Student', 'Credit Rating']].values
y = df['Buy Computer'].values

# Task 1 & 2
dt = DecisionTree(max_depth=3, min_samples_split=2)
dt.fit(X, y)
# decision tree structure :
print("\nDecision Tree Structure:")
dt.print_tree()

new_data = np.array([[42, 0, 0, 1]])  # Encoded new data
pred = dt.predict(new_data)
print(f"Task 2 Prediction: {'Yes' if pred[0] == 1 else 'No'}")

# Task 3
bagging = Bagging(n_trees=10, max_depth=3, min_samples_split=2, max_features=None)
bagging.fit(X, y)
oob_error_task3 = bagging.compute_oob_error(X, y)
print(f"Task 3 OOB Error: {oob_error_task3:.4f}")

# Task 4
bagging2 = Bagging(n_trees=10, max_depth=3, min_samples_split=2, max_features=2)
bagging2.fit(X, y)
oob_error_task4 = bagging2.compute_oob_error(X, y)
print(f"Task 4 OOB Error: {oob_error_task4:.4f}")