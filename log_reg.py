import numpy as np
import pandas as pd
import math
import time
from functions import *

# here's a pass at implementing the logistic regression model

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

def predict(weights, df, list_predictors,decision_tresh):
    #function of weights resulting from fit() and .csv file of features to be used as prediction
    predictors = init_x(df,list_predictors)
    predictions = []
    for i in range(len(predictors)):
        a = np.dot(predictors[i], weights) #compute a
        pred_prob = sigma(a) #compute probability
        if pred_prob > decision_tresh:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


