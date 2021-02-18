from sklearn.base import BaseEstimator


class RandomClassifier(BaseEstimator):


    def fit(self,X,y):
        self.shape = y.shape[1:]
        self.num_classes = 2

    def predict(self,X):
        num_samples = X.shape[1]
