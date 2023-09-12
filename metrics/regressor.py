import numpy as np

def mean_absolute_error(ytrue, ypred):
    mae = np.mean(np.absolute(ytrue-ypred))

    return mae

def mean_square_error(ytrue, ypred):
    mse = np.mean(np.square(ytrue - ypred))

    return mse

