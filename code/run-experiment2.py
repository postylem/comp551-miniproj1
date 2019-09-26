import numpy as np
from LDA import fit, predict, errorlda
import pandas as pd
from model import error, accuracy, precision, recall, init_y
from k_fold import *
#from LDA2 import accuracy

def run_LDA_and_report(k_folds, features, target_label):
    # running and reporting on the linear discriminate  analysis 
    prediction_list = []
    error_list = []
    acc_list = []
    prec_list = []
    rec_list = []


    for i in range(0,2*k,2):
        dataf = k_folds[i]
        if (dataf.columns[-1] == "quality"):
            dataf.loc[dataf["quality"] <= 5, ["quality"]] = 0
            dataf.loc[dataf["quality"] > 5, ["quality"]] = 1
        if (dataf.columns[-1] == "Class"):
            dataf.loc[bcw_df ["Class"] == 2, ["Class"]] = 0
            dataf.loc[bcw_df ["Class"] == 4, ["Class"]] = 1
        train_data_ = dataf[features].copy().to_numpy()
        stats_logodds = fit(train_data_)

        dataf = k_folds[i+1]
        train_data = dataf[features].copy().to_numpy()
        train_data_x = np.delete(train_data, -1, axis=1)
        
        y = predict( train_data_x, stats_logodds[0], stats_logodds[1], 
                           stats_logodds[2], stats_logodds[3], stats_logodds[4], 
                           stats_logodds[5])

        prediction_list.append(y)
        
        print(init_y(k_folds[i+1], target_label))
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
    print("---")


if __name__ == "__main__":

    k=5
    
    """print("---------------- running  LDA on wine data: -----------------")
    wine_df = pd.read_csv("winequality-red.csv", delimiter= ';')
    # wine_df = wine_df.reindex(np.random.permutation(wine_df.index))
    k_folds = k_fold(wine_df, k)
    features = ['density', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol', 'quality']
    # features = ['density', 'chlorides', 'volatile acidity', 'total sulfur dioxide','citric acid', 'sulphates', 'alcohol','quality']
    # features = ['density', 'volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol','quality']
    # features = ['volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol','quality]
    run_LDA_and_report(k_folds, features,"quality")"""
    
    
    print("--------------------- running LDA on the breast cancer dataset ----------------")
    bcw_df = pd.read_csv("bcw-cleaned.csv")
    k_folds = k_fold(bcw_df, k)
     # choose from  "Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"
    features = ["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli", "Class"]
    run_LDA_and_report(k_folds, features,"Class")
    







