import numpy as np
#from model import init_y


# takes in the training data in an array and output the "weights"
def fit(train_data):
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

    stats_logodds = [P_0, P_1, mean_0, mean_1, inv_cov]
    return stats_logodds

 # now to compute the predicted log_odds array for all the data_points using the outputs from fit
def predict(train_data, probs_0, probs_1, mean_0, mean_1, inv_cov):
    predicted_data = []
    # loop through all the feature data_points
    for data_point in range(len(train_data)):
        log_odds = (
            np.log(probs_1/probs_0)
            - 0.5 * np.dot(mean_1, np.dot(inv_cov, mean_1))
            + 0.5 * np.dot(mean_0, np.dot(inv_cov, mean_0))
            + np.dot(train_data[data_point], np.dot(inv_cov, (mean_1 - mean_0))))
        # takes in the log odd and classifies the datapoint to its respected 0 or 1 class - binary classification
       # print(log_odds)
        if (log_odds >= 0):
            predicted_data.append(1)
        else:
            predicted_data.append(0)
    return predicted_data