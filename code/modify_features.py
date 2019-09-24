import numpy as np
import pandas as pd
import math
import os
import sys
from model import *

#SEPARATE DATASET INTO TRAINING/VALIDATION and TEST SETS.

def new_csv():
    # input source file path, delimiter (',' or ';'), and percent for training
    df = pd.read_csv("winequality-red.csv", sep=';')

    #### Make modifications to matrix here

    print(df)

    df['citric_acid'] = df.citric_acid.apply(
               lambda x: ( x ** 4))

    df['quality'] = df.quality.apply(
               lambda x: (1 if x>5 else 0))

    print(df)

    ####

    # source_path = os.path.splitext(source)[0]
    # output delimiter will always be ',' the standard
    df.to_csv('wine_modified_alcohol.csv', index=False, sep = ';')


if __name__ == "__main__":
    new_csv()

