import numpy as np
import pandas as pd
import math
import os
import sys
from model import *

def new_csv():
    # input source file path, delimiter (',' or ';'), and percent for training
    df = pd.read_csv("winequality-red.csv", sep=';')

    #### Make modifications to matrix here

    print(df)

    df['density2'] = df.density.apply(
               lambda x: (1e5*(1 - x) ** 2))
    df['density3'] = df.density.apply(
               lambda x: (1e7*(1 - x) ** 3))
    df['density4'] = df.density.apply(
               lambda x: (1e9*(1 - x) ** 4))
    #
    # df['quality'] = df.quality.apply(
    #            lambda x: (1 if x>5 else 0))

    print(df)

    ####

    # source_path = os.path.splitext(source)[0]
    # output delimiter will always be ',' the standard
    df.to_csv('winequality-red-modified.csv', index=False, sep = ';')


if __name__ == "__main__":
    new_csv()
