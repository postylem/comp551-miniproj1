import numpy as np
from model import *
from k_fold import *
#from LDA import *

if __name__ == "__main__":

    k=5
    df = pd.read_csv("training.csv", delimiter = ';')
    k_folds = k_fold(df, k)
    features = ['pH', 'volatile acidity', 'citric acid', 'sulphates', 'alcohol']
    learning_rate = 0.02
    epochs = 700
    stop_criterion = 0.1

    weight_list = []
    prediction_list = []
    error_list = []

    for i in range(0,2*k,2):
        dataf = k_folds[i]
        X = init_x(dataf, features)
        W_0 = init_weights(X)
        Y = init_y(dataf,'quality')
        weights = fit(W_0, X, Y, learning_rate, epochs, stop_criterion)
        weight_list.append(weights)

        dataf = k_folds[i+1]
        y = predict(weights,dataf, features)
        prediction_list.append(y)

        err = error(y, k_folds[i+1], 'quality')
        error_list.append(err)
    
    print(error_list)
    print(np.mean(error_list))


    
    
    
    
    
    
    
    # df = pd.read_csv("test.csv", delimiter = ';')
    # y = predict(weights,df, features)
    # print(y)

    # acc = accuracy(y, "test.csv", 'quality')
    # print(acc)

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