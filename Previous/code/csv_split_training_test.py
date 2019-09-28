import numpy as np
import pandas as pd
import os
import sys

#SEPARATE DATASET INTO TRAINING/VALIDATION and TEST SETS.

def main(source, delimiter, percent):
    # input source file path, delimiter (',' or ';'), and percent for training
    df = pd.read_csv(source, sep=delimiter)
    shuffled_df = df.reindex(np.random.permutation(df.index))
    total = shuffled_df.shape[0]
    fraction_training = int(percent)/100.0

    nb_training = int(total * fraction_training)
    training_set = shuffled_df.head(nb_training)
    test_set = shuffled_df.tail(total-nb_training)

    source_path = os.path.splitext(source)[0]

    # output delimiter will always be ',' the standard
    training_set.to_csv(source_path+'.train.csv', index=False)
    test_set.to_csv(source_path+'.test.csv', index=False)

    print("total",df.shape[0])
    print("train",training_set.shape[0])
    print("test",test_set.shape[0])


if __name__ == "__main__":

    if int(sys.argv[3]) < 0 or int(sys.argv[3]) > 100:
        print("Argument for percentage out of bounds")
        sys.exit()
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
