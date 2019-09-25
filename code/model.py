import numpy as np
import pandas as pd
import math

# here's a pass at implementing the logistic regression model

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

def sigma(a):
    # the logistic squishing function sigma: R->[0,1]
    return 1/(1 + np.exp(-a))

def predicted_probability(w, x):
    # passes weighted sum w.x through logistic function
    return sigma(np.dot(w, x))

def L2_reg(lamb, weights):
    w = 0
    for i in range(len(weights)):
        w += weights[i] ** 2
    return lamb * w


def update_weights(weights, observations, true_labels, learning_rate):
    # function of current weights (vector) and observations (array of vectors),
    # updates by one step of size learning_rate based on grad of cross entropy loss
    if len(observations) == len(true_labels) and len(observations[0]) == len(weights):
        number_of_datapoints = np.size(true_labels)
        loss_vector = np.zeros_like(weights)
        for i in range(number_of_datapoints):
            loss_i = true_labels[i] - predicted_probability(weights, observations[i])
            loss_vector = np.add(loss_vector,observations[i] * loss_i)
        new_weights = np.add(weights, learning_rate * loss_vector)
        return new_weights
    else: print("Error. Size mismatch.")

def step_decay(iteration, lr):
    # step decay for learning rate
    step = 0.5
    resolution = 5 # numer of interations per step
    new_lr = lr * step ** math.floor((1+iteration)/resolution)
    return new_lr

def fit(weights, observations, true_labels, learning_rate,num_iterations, stop_criterion):
    #function of current weights (vector) and observations (array of vectors), true_labels (vector)
    #runs update_weight function num_iterations times, and stops if stop_criterion is reached
    last_weights = np.zeros_like(weights)
    for i in range(num_iterations):
        stepped_lr = step_decay(i,learning_rate)
        weights = update_weights(weights, observations, true_labels, stepped_lr)
        if  np.amax( np.absolute( np.subtract(weights,last_weights))) < stop_criterion: #if max(|weights-last_weights|)<stop_criterion
            print("Stop criterion reached, number of iterations:", i)
            return weights
        last_weights = weights
    return weights

def predict(weights, df, list_predictors):
    #function of weights resulting from fit() and .csv file of features to be used as prediction
    predictors = init_x(df,list_predictors)
    predictions = []
    for i in range(len(predictors)):
        a = np.dot(predictors[i], weights) #compute a
        pred_prob = sigma(a) #compute probability
        if pred_prob > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

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

def count(predictions, true_values):
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
    true_y = init_y(df, target)
    m = count(predictions, true_y)
    acc = float(m[0]+m[2])/(m[0]+m[1]+m[2]+m[3])
    return acc

def precision(predictions, df, target):
    true_y = init_y(df, target)
    m = count(predictions, true_y)
    if m[0]==m[1]==0:
        # print("No positives at all (true nor false). Precision undefined.")
        return math.inf
    prec = float(m[0])/(m[0]+m[1])
    return prec

def recall(predictions, df, target):
    true_y = init_y(df, target)
    m = count(predictions, true_y)
    if m[0]==m[3]==0:
        # print("No true positives nor false negatives. Recall undefined.")
        return math.inf
    rec = float(m[0])/(m[0]+m[3])
    return rec
