import pandas as pd
import numpy as np

from data_processing.data_split import cross_validate
from metrics.regressor import *
from model.regressor import linreg



if __name__ == "__main__":

    data = pd.read_csv("data/Salary_Data.csv")
    X = data.iloc[:, 0]
    y = data.iloc[:, 1]



    data_cv = cross_validate(X,y)

    metrics = []

    for k in data_cv.keys():
        X_train = data_cv[k]["X_train"]
        y_train = data_cv[k]["y_train"]
        X_test  = data_cv[k]["X_test"]
        y_test  = data_cv[k]["y_test"]


        print(type(X_train), type(y_train))

        print(X_train.index, y_train.index,sep="\n")
        w_0, w_1 = linreg(np.array(X_train),np.array(y_train))

        f = lambda x: w_0 + w_1 * x

        metrics.append(
            mean_absolute_error(
            y_test, f(X_test)
            )
        )

    print(np.mean(metrics))





