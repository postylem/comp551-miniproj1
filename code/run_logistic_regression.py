import numpy as np
from logistic_regression import *
from k_fold import *

if __name__ == "__main__":

    features = ['pH', 'volatile acidity', 'citric acid', 'sulphates', 'alcohol']
    # X = init_x("training.csv", features)
    # W_0 = init_weights(X)
    # Y = init_y("training.csv", 'quality')
    learning_rate = 0.02
    epochs = 700
    stop_criterion = 0.1
    # weights = fit(W_0, X, Y, learning_rate, epochs, stop_criterion)
    training_set = "training.csv"
    k = 5
    weights = fit(training_set, learning_rate, epochs, stop_criterion,k)
    


    # y = predict(weights,"test.csv", features)
    # print(y)

    # acc = accuracy(y, "test.csv", 'quality')
    # print(acc)