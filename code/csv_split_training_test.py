import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import sys

#GOAL IS TO SEPARE DATASET INTO TRAINING/VALIDATION and TEST SETS.

def main(source, percent):
    df = pd.read_csv(source, delimiter = ';')
    shuffled_df = df.reindex(np.random.permutation(df.index))
    
    total = shuffled_df.shape[0]

    fraction_training = int(percent)/100.0

    nb_training = int(total * fraction_training)

    training_set = shuffled_df.head(nb_training)

    test_set = shuffled_df.tail(total-nb_training)

    training_set.to_csv('training.csv', sep = ';', index=False)

    test_set.to_csv('test.csv', sep = ';', index=False)

    print(df.shape[0])
    print(training_set.shape[0])
    print(test_set.shape[0])


if __name__ == "__main__":

    if int(sys.argv[2]) < 0 or int(sys.argv[2]) > 100:
        print("Argument for percentage out of bounds")
        sys.exit()
    else:
        main(sys.argv[1], sys.argv[2])
