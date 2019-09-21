import numpy as np


def get_log_odds(data):
    # gets the predicted log odds : log[ P(y=1|x)/P(y=0|x) ]

    # observations - a numpy array of observations
    # true_labels - a list of the true labels for each row of observations

    observations_labeled_0 = data[data[:,-1]==0][:,:-1]
    observations_labeled_1 = data[data[:,-1]==1][:,:-1]
    N_0 = np.size(observations_labeled_0,axis=0)
    N_1 = np.size(observations_labeled_1,axis=0)
    P_0 = N_0 / (N_0 + N_1)
    P_1 = N_1 / (N_0 + N_1)
    mean_0 = np.mean(observations_labeled_0,0)
    mean_1 = np.mean(observations_labeled_1,0)
    covariance_matrix = np.cov(np.transpose(data[:,:-1]))
    invcov = np.linalg.inv(covariance_matrix)

    # covariance matrix calculated on the train data without the last column (the labels)
    covariance_matrix = np.cov(data[:,:-1])

    log_odds = 0
    return log_odds
