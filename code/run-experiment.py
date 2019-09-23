import numpy as np
from model import *
from k_fold import *
#from LDA import *

if __name__ == "__main__":

    #
    #
    #
    #
    # RUNNING LOGISTIC REGRESSION ON WINE DATASET
    #
    #
    #
    #

    k=5

    print("running logistic regression on wine data:")

    wine_df = pd.read_csv("winequality-red.train.csv")
    k_folds = k_fold(wine_df, k)
    features = ['density', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol']
    learning_rate = 0.02
    epochs = 600
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
        print("Fold", int(i/2), "error:", err)
        error_list.append(err)

    print("Mean error across folds: ", np.mean(error_list))

    # train model on all of the training set.

    print("Now training model on the whole training set.")

    X = init_x(wine_df, features)
    W_0 = init_weights(X)
    Y = init_y(wine_df,'quality')
    weights = fit(W_0, X, Y, learning_rate, epochs, stop_criterion)

    test_df = pd.read_csv("winequality-red.test.csv")
    y = predict(weights,test_df, features)
    acc = accuracy(y,test_df, 'quality')
    print("The accuracy on the test set is", acc)
    prec = precision(y,test_df, 'quality')
    print("The precision on the test set is", prec)
    recall = recall(y,test_df, 'quality')
    print("The recall on the test set is", recall)

    #
    #
    #
    #
    # RUNNING LOGISTIC REGRESSION ON BREAST CANCER DATA
    #
    #
    #
    #

    print("running logistic regression on breast cancer data:")

    bcw_df = pd.read_csv("bcw-cleaned.train.csv")
    k_folds = k_fold(bcw_df, k)
    # choose from  "Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"
    features = ["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli"]
    learning_rate = 0.01
    epochs = 700
    stop_criterion = 0.0

    weight_list = []
    prediction_list = []
    error_list = []

    for i in range(0,2*k,2):
        dataf = k_folds[i]
        X = init_x(dataf, features)
        W_0 = init_weights(X)
        Y = init_y(dataf,'Class')
        weights = fit(W_0, X, Y, learning_rate, epochs, stop_criterion)
        weight_list.append(weights)

        dataf = k_folds[i+1]
        y = predict(weights,dataf, features)
        prediction_list.append(y)

        err = error(y, k_folds[i+1], 'Class')
        print("Fold", int(i/2), "error:", err)
        error_list.append(err)

    print("Mean error across folds: ", np.mean(error_list))

    print("Now training model on the whole training set.")

    X = init_x(bcw_df, features)
    W_0 = init_weights(X)
    Y = init_y(bcw_df,'Class')
    weights = fit(W_0, X, Y, learning_rate, epochs, stop_criterion)

    test_df = pd.read_csv("bcw-cleaned.test.csv")
    y = predict(weights,test_df, features)
    acc = accuracy(y,test_df, 'Class')
    print("The accuracy on the test set is", acc)
    prec = precision(y,test_df, 'Class')
    print("The precision on the test set is", prec)
    recall = recall(y,test_df, 'Class')
    print("The recall on the test set is", recall)


    # df = pd.read_csv("test.csv", delimiter = ';')
    # y = predict(weights,df, features)
    # print(y)

    # acc = accuracy(y, "test.csv", 'quality')
    # print(acc)

    # testing LDA:
    # data = np.array(
    #     [[2,0,0,1],
    #      [2,0,4,1],
    #      [4,2,3,0],
    #      [9,1,2,0],
    #      [9,1,4,0],
    #      [9,1,2,0],
    #      [2,1,1,0]]
    # )


    # predicted_odds = np.empty([np.size(data,0),1])
    # for i in range(np.size(data,0)):
    #         predicted_odds[i] = predict_log_odds(train_data , data[i][:-1])
    # print(predicted_odds)import numpy as np
