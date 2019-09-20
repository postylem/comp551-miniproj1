import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import sys

def k_fold(dataframe, k): #takes in a dataframes, returns list of k dataframes
    
    lines = dataframe.shape[0]
    quotient = int(lines / k)
    rem = lines % k

    folds = []
    next_row = 0

    print(quotient)

    for i in range(k):

        if i < rem:
            quotient +=1
        
        df = dataframe.iloc[ next_row : next_row + quotient]
        folds.append(df)

        next_row += quotient

        if i < rem:
            quotient -=1

    #print(folds)

    return folds

        

def main():

    source = sys.argv[1]
    df = pd.read_csv(source, delimiter = ';')

    k_fold(df, int(sys.argv[2]))

if __name__ == "__main__":
    main()


