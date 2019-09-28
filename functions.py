import numpy as np
import pandas as pd
import math
import time

def standardize(vector):
    #takes in a vector of data from one feature, standardizes the data
    min_ = np.amin(vector)
    delta = (np.amax(vector) - min_)
    return (vector - min_)/delta

def init_x(df,features):
    #creates the matrix of training data from .csv file
    #and a list of column names for features e.g. ['pH','alcohol'] and a column of 1`s
    data = []
    for i in features:
        feature = standardize(np.array(df[i])) #standardized feature
        data.append(feature)
    matrix = data[len(features)-1]
    if len(features)-1 == 0:
        matrix = np.column_stack(([1]*len(data[0]),matrix)) #add column of 1
    else:
        for i in reversed(range( len(features)-1 )):
            matrix = np.column_stack((data[i],matrix)) #add i-th feature
            if i == 0:
                matrix = np.column_stack(([1]*len(data[0]),matrix)) #add column of 1
    return matrix

def init_weights(matrix):
    #creates the vector of initial weights
    return [1] * len(matrix[0])

def init_y(df, feature):
    y = np.array(df[feature])
    if feature == 'quality':
        y[y<=5] = 0
        y[y>5] = 1
    if feature == 'Class':
        y[y==2] = 0
        y[y==4] = 1
    return y

def error(predictions, dataframe, target):
    real_y = init_y(dataframe, target)
    count = 0
    for i in range(len(real_y)):
        if real_y[i] != predictions[i]:
            count+=1
        else:
            continue
    error = float(count)/len(real_y)
    return error


def count(predictions, df, target):
    true_values = init_y(df, target)
    m = [0,0,0,0]
    for i in range(len(true_values)):
        if true_values[i] == predictions[i] and true_values[i] == 1:
            m[0] +=1 # True positives
        if true_values[i] != predictions[i] and true_values[i] == 0:
            m[1] +=1 # False positives
        if true_values[i] == predictions[i] and true_values[i] == 0:
            m[2] +=1 # True negatives
        if true_values[i] != predictions[i] and true_values[i] == 1:
            m[3] +=1 # False negatives
    return m


def accuracy(predictions, df, target):
    m = count(predictions, df, target)
    acc = float(m[0]+m[2])/(m[0]+m[1]+m[2]+m[3])
    return acc

def precision(predictions, df, target):
    m = count(predictions, df, target)
    if m[0]==m[1]==0:
        # print("No positives at all (true nor false). Precision undefined.")
        return math.inf
    prec = float(m[0])/(m[0]+m[1])
    return prec

def recall(predictions, df, target):
    m = count(predictions, df, target)
    if m[0]==m[3]==0:
        # print("No true positives nor false negatives. Recall undefined.")
        return math.inf
    rec = float(m[0])/(m[0]+m[3])
    return rec