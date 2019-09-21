import numpy as np

from model import *
from LDA import *

if __name__ == "__main__":

    data = np.array(
        [[2,0,2,0],
         [4,2,3,1],
         [9,1,2,0],
         [9,1,2,0],
         [1,5,5,1],
         [3,1,0,1],
         [3,1,0,1],
         [1,0,1,0]]
    )

    # # uncomment for testing logistic regression:
    # X = data[:,:-1]
    # y = np.transpose(data[:,-1:])
    # W_0 = np.zeros_like(X[0])
    # lr = 0.2
    # epochs = 100
    # stop_criterion = 0
    #
    # logistic_regression(W_0, X, y, lr, epochs, stop_criterion)


    # testing LDA:
    datapoint = np.zeros_like(data[:,:-1][0])
    print(predict_log_odds(data , datapoint))
