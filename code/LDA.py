import numpy as np


def predict_log_odds(train_data, test_datapoint):
    # takes in training data in an array, one row per observation,
    # with true labels being the last row of the data
    # gets the predicted log odds of test_datapoint being label = 1 vs 0
    # that is, log[ P(y=1|x)/P(y=0|x) ] , given the assumpions of LDA

    # make new array, just containing the rows of a given label, while removing that last row
    observations_labeled_0 = train_data[train_data[:,-1]==0][:,:-1]
    observations_labeled_1 = train_data[train_data[:,-1]==1][:,:-1]

    # some useful constants:
    N_0 = np.size(observations_labeled_0,axis=0)
    N_1 = np.size(observations_labeled_1,axis=0)
    if (N_0 + N_1 != np.size(train_data,axis=0)):
        print("Warning! At least one train_data point is not labeled 0 or 1")
    P_0 = N_0 / (N_0 + N_1)
    P_1 = N_1 / (N_0 + N_1)

    # the mean vectors (mean over all train_data points of each label)
    mean_0 = np.mean(observations_labeled_0,0)
    mean_1 = np.mean(observations_labeled_1,0)

    # make the covariance matrix (note we want the covariance matrix over the _columns_ of train_data, hence the transpose)
    cov_matrix = np.cov(np.transpose(train_data[:,:-1]))
    inv_cov = np.linalg.inv(cov_matrix)

    # now to compute the predicted log odds
    log_odds = (
        np.log(P_1/P_0)
        - 0.5 * np.dot(mean_1, np.dot(inv_cov, mean_1))
        + 0.5 * np.dot(mean_0, np.dot(inv_cov, mean_0))
        + np.dot(test_datapoint , np.dot(inv_cov, (mean_0 - mean_1))))

    return log_odds
