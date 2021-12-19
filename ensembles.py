import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from time import time


def RMSE(y_1, y_2):
    return np.mean((y_1 - y_2) ** 2) ** 0.5


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.feat_idx = []
        self.models = []
        self.DTR = DecisionTreeRegressor
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if X.shape[0] != y.shape[0]:
            raise Exception("X.shape[0] != y.shape[0]!!!")

        if X_val is not None:
            if y_val is None:
                raise Exception("y_val is None!!!")
            if X_val.shape[0] != y_val.shape[0]:
                raise Exception("X_val.shape[0] != y_val.shape[0]!!!")
            if X.shape[1] != X_val.shape[1]:
                raise Exception("X.shape[1] != X_val.shape[1]!!!")

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        loss_train = []
        loss_val = []
        times = []

        sum_pred_train = np.zeros(X.shape[0])
        if X_val is not None:
            sum_pred_val = np.zeros(X_val.shape[0])

        start_time = time()

        for idx in range(self.n_estimators):
            obj_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            self.feat_idx.append(np.random.choice(X.shape[1], self.feature_subsample_size, replace=False))
            X_train = X[obj_idx, :][:, self.feat_idx[-1]]
            y_train = y[obj_idx]

            self.models.append(self.DTR(max_depth=self.max_depth, **self.trees_parameters).fit(X_train, y_train))

            sum_pred_train += self.models[-1].predict(X[:, self.feat_idx[-1]])
            loss_train.append(RMSE(sum_pred_train / len(self.models), y))

            if X_val is not None:
                sum_pred_val += self.models[-1].predict(X_val[:, self.feat_idx[-1]])
                loss_val.append(RMSE(sum_pred_val / len(self.models), y_val))
            times.append(time() - start_time)
        
        if X_val is not None:
            return loss_train, loss_val, times
        else:
            return loss_train, times

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predicts = np.zeros((self.n_estimators, X.shape[0]))
        for alg in range(len(self.models)):
            ind = self.feat_idx[alg]
            dtr = self.models[alg]
            predicts[alg] = dtr.predict(X[:, ind])
            
        res = np.sum(predicts, axis = 0) / np.count_nonzero(predicts, axis = 0)
        return res


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
      

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
       
