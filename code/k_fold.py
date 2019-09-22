import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import sys

def k_fold(dataframe, k): #takes in a dataframe, returns list of k dataframes
    
    lines = dataframe.shape[0]
    quotient = int(lines / k)
    rem = lines % k

    folds = [] #contains k distinct dataframes from the training set
    next_row = 0


    for i in range(k):
        
        if i < rem:
            quotient +=1
        
        df = dataframe.iloc[ next_row : next_row + quotient]
        folds.append(df)

        next_row += quotient

        if i < rem:
            quotient -=1
    

    #k_folds contains the cross validation sets in form (k-1) dataframes for training, 1 for test,  (k-1) dataframes for training, 1 for test, etc.
    k_fold = []

    for i in range(k):
        df = folds[i]
        for j in range(1,k-1):
            df = df.append(folds[(j+i)%k], ignore_index = True)
        k_fold.append(df)
        k_fold.append(folds[(i+k-1)%k])

    return k_fold 

def main():

    source = sys.argv[1]
    df = pd.read_csv(source, delimiter = ';')

    k_fold(df, int(sys.argv[2]))

if __name__ == "__main__":
    main()