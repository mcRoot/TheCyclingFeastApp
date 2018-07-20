from scipy.sparse import lil_matrix
import sklearn.base as base
import numpy as np

class ResidualEstimator(base.BaseEstimator, base.RegressorMixin):

    def __init__(self, est_1, est_2):
        self.est_1 = est_1
        self.est_2 = est_2

    def fit(self, X, y):
        self.est_1.fit(lil_matrix(X), y)
        self.est_2.fit(lil_matrix(X), y - self.est_1.predict(X))
        return self

    def predict(self, X):
        p_1 = self.est_1.predict(X)
        pred = self.est_2.predict(X)
        return np.array(p_1 + pred)


class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return X[self.col_names].values
        # print(self.df[self.col_names].values.shape)
        # return self.df[self.col_names].values.flatten().tolist()