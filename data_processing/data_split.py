import pandas as pd
import numpy as np


def KFold(X, y, test_size = 0.2):


    n_test  = int(len(X) * test_size)
    X_cross_validate = X.reindex(np.random.permutation(X.index))
    y_cross_validate = y.copy()
    y_cross_validate.index = X_cross_validate.index


    return X_cross_validate[:-n_test], y_cross_validate[:-n_test], X_cross_validate[:n_test], y_cross_validate[:n_test]

def cross_validate( X, y, test_size = 0.2, K = 10, ):

    data_cv = dict()

    metrics = []
    for i in range(K):
        X_train, y_train, X_test, y_test = KFold(X, y, test_size = 0.2)
        data_cv[f"fold{i+1}"] = {
            "X_train"   : X_train,
            "y_train"   : y_train,
            "X_test"    : X_test,
            "y_test"    : y_test

        }


    return data_cv









if __name__== "__main__":
    data = pd.read_csv("../data/Salary_Data.csv")
    data.info()




