import numpy as np
import pandas as pd
import math
import os
import sys

def new_csv():
    # input source file path, delimiter (',' or ';'), and percent for training
    df = pd.read_csv("winequality-red.randomized.csv", sep=',')

    #### Make modifications to matrix here

    y = df["quality"]

    df = df.drop(['quality'], axis=1)

    df = df.assign(density2
               =lambda x: (1e5*(1 - x['density']) ** 2))

    df['quality'] = y

    ####

    # source_path = os.path.splitext(source)[0]
    # output delimiter will always be ',' the standard
    df.to_csv('winequality-red.randomized-modified.csv', index=False, sep = ',')


if __name__ == "__main__":
    new_csv()
