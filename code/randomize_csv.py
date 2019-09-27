import numpy as np
import pandas as pd
import os
import sys

#SEPARATE DATASET INTO TRAINING/VALIDATION and TEST SETS.

def main(source, delimiter):
    # input source file path, delimiter (',' or ';'), and percent for training
    df = pd.read_csv(source, sep=delimiter)
    shuffled_df = df.reindex(np.random.permutation(df.index))

    # output delimiter will always be ',' the standard
    source_path = os.path.splitext(source)[0]
    df.to_csv(source_path+'.randomized.csv', index=False)

if __name__ == "__main__":

        main(sys.argv[1], sys.argv[2])
