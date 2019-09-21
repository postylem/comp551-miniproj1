import numpy as np


# here's a pass at implementing the logistic regression model

def standardize(vector):
    min_ = np.amin(vector)
    delta = (np.amax(vector) - min_)
    return (vector - min_)/delta

def sigma(a):
    # the logistic squishing function sigma: R->[0,1]
    return 1/(1 + np.exp(-a))

def predicted_probability(w, x):
    # passes weighted sum w.x through logistic function
    return sigma(np.dot(w, x))

def update_weights(weights, observations, true_labels, learning_rate):
    # function of current weights (vector) and observations (array of vectors),
    # updates by one step of size learning_rate based on grad of cross entropy loss
    if (np.size(observations,0) == true_labels.size and np.size(observations,1) == weights.size):
        number_of_datapoints = np.size(true_labels)
        loss_vector = np.zeros_like(weights)
        for i in range(number_of_datapoints):
            loss_i = true_labels[i] - predicted_probability(weights, observations[i])
            loss_vector = np.add(loss_vector,observations[i] * loss_i)
        new_weights = np.add(weights, learning_rate * loss_vector)
        return new_weights
    else: print("Error. Size mismatch.")

def logistic_regression(weights, observations, true_labels, learning_rate,num_iterations, stop_criterion):
    #function of current weights (vector) and observations (array of vectors), true_labels (vector)
    #runs update_weight function num_iterations times, and stops if stop_criterion is reached
    last_weights = np.zeros_like(weights)
    for i in range(num_iterations):
        weights = update_weights(weights, observations, true_labels, learning_rate)
        if  np.amax( np.absolute( np.substract(weights,last_weights))) < stop_criterion: #if max(|weights-last_weights|)<stop_criterion
            print("Stop criterion reached:", np.substract(weights,last_weights))
            return weights
        last_weights = weights
    return weights
