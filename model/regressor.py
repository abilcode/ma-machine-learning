import numpy as np

# fungsi jumlah utk memudahkan
def sum(x):
    n = np.size(x)
    sum = 0
    for i in range(n):
        sum += x[i]
    return sum


# fungsi regresi linear, outputnya koefisien persamaan linear y = w_0 + w_1*x
def linreg(x = None, y = None):
    n = np.size(x)
    mean_y = sum(y)/n
    mean_x = sum(x)/n
    mean_xy = sum(x*y)/n
    mean_xx = sum(x*x)/n
    w_1 = (mean_xy - mean_x * mean_y) / (mean_xx - mean_x**2)
    w_0 = mean_y - w_1 * mean_x

    return w_0, w_1

if __name__== "__main__":
    import pandas as pd

    data = pd.read_csv("../data/Salary_Data.csv")
    X = data.iloc[:,0]
    y = data.iloc[:, 1]
    print(type(X))
    w0 , w1 = linreg(X, y)
    f = lambda x : w0 + w1 * x


    print(f(np.array([1000,10002,20002])))

