import numpy as np
from model import *
#from LDA import *

if __name__ == "__main__":

    # train_data = np.array(
    #     [[2,0,2,0],
    #      [4,2,3,1],
    #      [9,1,2,0],
    #      [9,1,2,0],
    #      [1,5,5,1],
    #      [3,1,0,1],
    #      [3,1,0,1],
    #      [1,0,1,0]]
    # )

    # # uncomment for testing logistic regression:
    # X = train_data[:,:-1]
    # y = (train_data[:,-1:]).flatten()
    # W_0 = np.zeros_like(X[0])
    # learning_rate = 0.2
    # epochs = 100
    # stop_criterion = 0

    features = ['pH', 'volatile acidity', 'citric acid', 'sulphates', 'alcohol']
    df = pd.read_csv("training.csv", delimiter = ';')
    X = init_x(df, features)
    W_0 = init_weights(X)
    Y = init_y(df, 'quality')
    learning_rate = 0.02
    epochs = 700
    stop_criterion = 0.1
    weights = fit(W_0, X, Y, learning_rate, epochs, stop_criterion)
    print(weights)

    df = pd.read_csv("test.csv", delimiter = ';')
    y = predict(weights,df, features)
    print(y)

    acc = accuracy(y, "test.csv", 'quality')
    print(acc)

    # testing LDA:
    data = np.array(
        [[2,0,0,1],
         [2,0,4,1],
         [4,2,3,0],
         [9,1,2,0],
         [9,1,4,0],
         [9,1,2,0],
         [2,1,1,0]]
    )


    # predicted_odds = np.empty([np.size(data,0),1])
    # for i in range(np.size(data,0)):
    #         predicted_odds[i] = predict_log_odds(train_data , data[i][:-1])
    # print(predicted_odds)import numpy as np