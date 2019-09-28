import numpy as np
from LDA import fit, predict
import pandas as pd
from model import error, accuracy, precision, recall, count
from k_fold import *
import time
#from LDA2 import accuracy

def run_LDA_and_report(k_folds, features, target_label):
    # running and reporting on the linear discriminate  analysis
    prediction_list = []
    error_list = []
    acc_list = []
    prec_list = []
    rec_list = []
    run_times = []
    class_outputs = []

    for i in range(0,2*k,2):
        trainingfold_df = k_folds[i]
        if (trainingfold_df.columns[-1] == "quality"):
            trainingfold_df.loc[trainingfold_df["quality"] <= 5, ["quality"]] = 0
            trainingfold_df.loc[trainingfold_df["quality"] > 5, ["quality"]] = 1
        if (trainingfold_df.columns[-1] == "Class"):
            trainingfold_df.loc[trainingfold_df ["Class"] == 2, ["Class"]] = 0
            trainingfold_df.loc[trainingfold_df ["Class"] == 4, ["Class"]] = 1
        train_data_ = trainingfold_df[features].copy().to_numpy()
        start = time.time()
        stats_logodds = fit(train_data_)
        end = time.time()

        validation_df = k_folds[i+1]
        train_data = validation_df[features].copy().to_numpy()
        train_data_x = np.delete(train_data, -1, axis=1)

        y = predict( train_data_x, stats_logodds[0], stats_logodds[1],
                           stats_logodds[2], stats_logodds[3], stats_logodds[4])

        prediction_list.append(y)

        # print(init_y(validation_df, target_label))
        err = error(y, validation_df, target_label)
        print("Fold", int(i/2), "error:          ", err)
        error_list.append(err)
        acc = accuracy(y,validation_df, target_label)
        acc_list.append(acc)
        print("       accuracy:       ", acc)
        prec = precision(y,validation_df, target_label)
        prec_list.append(prec)
        print("       precision:      ", prec)
        rec = recall(y,validation_df, target_label)
        rec_list.append(rec)
        print("       recall:         ", rec)
        run_times.append(end-start)
        class_outputs.append(count(y, validation_df, target_label))
        print(class_outputs[int(i/2)])


    m = class_outputs[0]
    for i in range(1,5):
        m = np.add(m, class_outputs[i])
    m = m/5
    print("")
    print("Avg TP:", m[0], " FP:", m[1], " TN:", m[2], "FN:", m[3])

    print("Mean error across folds:    ", np.mean(error_list))
    print("Mean accuracy across folds: ", np.mean(acc_list))
    print("Mean precision across folds:", np.mean(prec_list))
    print("Mean recall across folds:   ", np.mean(rec_list))
    print("Mean run time for fit function across folds:", np.mean(run_times), " seconds.")
    print("---")


if __name__ == "__main__":

    k=5


    print("--------------------- running LDA on the breast cancer dataset ----------------")
    bcw_df = pd.read_csv("bcw-cleaned.randomized.csv")
    k_folds = k_fold(bcw_df, k)
     # choose from  "Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"
    features = ["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli", "Class"]
    run_LDA_and_report(k_folds, features,"Class")

    print("---------------- running  LDA on wine data: -----------------")
    wine_df = pd.read_csv("winequality-red.randomized.csv", delimiter= ',')
    # wine_df = wine_df.reindex(np.random.permutation(wine_df.index))
    k_folds = k_fold(wine_df, k)
    features = ['density', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol', 'quality']
    # features = ['density', 'chlorides', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol','quality']
    # features = ['density', 'volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol','quality']
    # features = ['volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol','quality]
    run_LDA_and_report(k_folds, features,"quality")