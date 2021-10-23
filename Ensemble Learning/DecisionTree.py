import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class DecisionTreeClassifierWithWeight:
    def __init__(self):
        self.best_err = 1  
        self.best_fea_id = 0  
        self.best_thres = 0  
        self.best_op = 1 

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        n = X.shape[1]
        for i in range(n):
            feature = X[:, i] 
            fea_unique = np.sort(np.unique(feature))  
            for j in range(len(fea_unique)-1):
                thres = (fea_unique[j] + fea_unique[j+1]) / 2 
                for op in (0, 1):
                    y_ = 2*(feature >= thres)-1 if op==1 else 2*(feature < thres)-1 
                    err = np.sum((y_ != y)*sample_weight)
                    if err < self.best_err:  
                        self.best_err = err
                        self.best_op = op
                        self.best_fea_id = i
                        self.best_thres = thres
        return self
    
    def predict(self, X):
        feature = X[:, self.best_fea_id]
        return 2*(feature >= self.best_thres)-1 if self.best_op==1 else 2*(feature < self.best_thres)-1
    
    def score(self, X, y, sample_weight=None):
        y_pre = self.predict(X)
        if sample_weight is not None:
            return np.sum((y_pre == y)*sample_weight)
        return np.mean(y_pre == y)


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    y = 2*y-1 
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(type(X_train))
