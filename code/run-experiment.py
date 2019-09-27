import numpy as np
from model import *
from k_fold import *
import time
#from LDA import *

def run_logreg_and_report(k_folds,features,target_label,learning_rate,epochs,stop_criterion):
    # running and reporting on the logistic regression model
    weight_list = []
    prediction_list = []
    error_list = []
    acc_list = []
    prec_list = []
    rec_list = []
    run_times = []

    print(features)

    for i in range(0,2*k,2):
        dataf = k_folds[i]
        X = init_x(dataf, features)
        W_0 = init_weights(X)
        Y = init_y(dataf,target_label)
        start = time.time()

        weights = fit(W_0, X, Y, learning_rate, epochs, stop_criterion)

        end = time.time()

        run_times.append(end-start)
        weight_list.append(weights)

        dataf = k_folds[i+1]
        y = predict(weights,dataf, features)
        prediction_list.append(y)

        err = error(y, k_folds[i+1], target_label)
        print("Fold", int(i/2), "error:          ", err)
        error_list.append(err)
        acc = accuracy(y,k_folds[i+1], target_label)
        acc_list.append(acc)
        print("       accuracy:       ", acc)
        prec = precision(y,k_folds[i+1], target_label)
        prec_list.append(prec)
        print("       precision:      ", prec)
        rec = recall(y,k_folds[i+1], target_label)
        rec_list.append(rec)
        print("       recall:         ", rec)


    print("")
    print("Mean error across folds:    ", np.mean(error_list))
    print("Mean accuracy across folds: ", np.mean(acc_list))
    print("Mean precision across folds:", np.mean(prec_list))
    print("Mean recall across folds:   ", np.mean(rec_list))
    print("Mean run time for fit function across folds:", np.mean(run_times), " seconds.")
    print("---")


if __name__ == "__main__":

    k=5

    print("--> running logistic regression on wine data:")
    wine_df = pd.read_csv("winequality-red.randomized.csv", delimiter= ',')
    # wine_df = wine_df.reindex(np.random.permutation(wine_df.index))
    k_folds = k_fold(wine_df, k)
    features = ['density', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol']
    # features = ['density', 'chlorides', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol']
    # features = ['density', 'volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol']
    # features = ['volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol']
    learning_rate = 0.02
    epochs = 100
    stop_criterion = 0.1

    run_logreg_and_report(k_folds,features,"quality",learning_rate,epochs,stop_criterion)


    print("--> running logistic regression on modified wine data:")
    wine_df = pd.read_csv("winequality-red-modified.csv", delimiter= ';')
    # wine_df = wine_df.reindex(np.random.permutation(wine_df.index))
    k_folds = k_fold(wine_df, k)
    features = ['density2', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol']
    # features = ['density', 'chlorides', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol']
    # features = ['density', 'volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol']
    # features = ['volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol']
    learning_rate = 0.02
    epochs = 100
    stop_criterion = 0.1

    run_logreg_and_report(k_folds,features,"quality",learning_rate,epochs,stop_criterion)



    print("--> running logistic regression on breast cancer data:")
    bcw_df = pd.read_csv("bcw-cleaned.randomized.csv")
    k_folds = k_fold(bcw_df, k)
    # choose from  "Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"
    #features = ["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli"]
    features = ["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]
    learning_rate = 0.01
    epochs = 100
    stop_criterion = 0.1

    run_logreg_and_report(k_folds,features,"Class",learning_rate,epochs,stop_criterion)



    # predicted_odds = np.empty([np.size(data,0),1])
    # for i in range(np.size(data,0)):
    #         predicted_odds[i] = predict_log_odds(train_data , data[i][:-1])
    # print(predicted_odds)import numpy as np
