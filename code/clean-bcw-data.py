import numpy as np
import pandas as pd
import os
import sys


def prepare_breast_cancer_data(source, bool_remove_rows):
    # custom data cleaning for breast cancer data
    df = pd.read_csv(source,names=["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shap","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"])
    print("total rows",df.shape[0])
    if bool_remove_rows:
        df = df[pd.to_numeric(df['Bare Nuclei'], errors='coerce').notnull()]
        print("after cleaning",df.shape[0])
        df.to_csv('bcw-cleaned.csv', index=False)
    else:
        df.to_csv('bcw.csv', index=False)

if __name__ == "__main__":

    if sys.argv[2] == 'remove-rows':
        prepare_breast_cancer_data(sys.argv[1], True)
    if sys.argv[2] == 'keep-rows':
        prepare_breast_cancer_data(sys.argv[1], False)
