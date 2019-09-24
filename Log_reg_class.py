import numpy as np
import pandas as pd

# here's a pass at implementing the logistic regression model
class log_reg:
 #initialize the attributes given when creating a class for logistic regression
    def __init__(self, prep_training_set, learning_rate, epochs, stop_criterion):
        self.prep_training_set = prep_training_set
        self.learning_rate = learning_rate 
        self.epochs = epochs
        self.stop_criterion = stop_criterion
       
    # this function will take the initilaized inputs and return the weights after the learning process
    def __new__(self, cls):
        X = log_reg.init_x(self.prep_training_set, self.prep_training_set[:, 0: -1]) # preprocessed features, x's
        Y = log_reg.init_y(self.prep_training_set, self.prep_training_set[:, -1]) # e.g wine quality / malignant or beingn
        W_0 = log_reg.init_weights(X)
        weights = log_reg.fit(W_0, X, Y, self.learning_rate, self.epochs, self.stop_criterion)
        return weights
        
    def standardize(vector):
        #takes in a vector of data from one feature, standardizes the data
        min_ = np.amin(vector)
        delta = (np.amax(vector) - min_)
        return (vector - min_)/delta
    
    def init_x(self, source,features):
        #creates the matrix of training data from .csv file
        #and a list of column names for features e.g. ['pH','alcohol'] and a column of 1`s
        df = pd.read_csv(source, delimiter = ';')
        data = []
        for i in features:
            feature = self.standardize(np.array(df[i])) #standardized feature
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
    
    def init_y(source, feature):
        df = pd.read_csv(source, delimiter = ';')
        y = np.array(df[feature])
        y[y<=5] = 0
        y[y>5] = 1
        return y
    
    def sigma(a):
        # the logistic squishing function sigma: R->[0,1]
        return 1/(1 + np.exp(-a))
    
    def predicted_probability(self, w, x):
        # passes weighted sum w.x through logistic function
        return self.sigma(np.dot(w, x))
    
    def update_weights(self, weights, observations, true_labels, learning_rate):
        # function of current weights (vector) and observations (array of vectors),
        # updates by one step of size learning_rate based on grad of cross entropy loss
        if len(observations) == len(true_labels) and len(observations[0]) == len(weights):
            number_of_datapoints = np.size(true_labels)
            loss_vector = np.zeros_like(weights)
            for i in range(number_of_datapoints):
                loss_i = true_labels[i] - self.predicted_probability(weights, observations[i])
                loss_vector = np.add(loss_vector,observations[i] * loss_i)
            new_weights = np.add(weights, learning_rate * loss_vector)
            return new_weights
        else: print("Error. Size mismatch.")
    
    def fit(self, weights, observations, true_labels, learning_rate,num_iterations, stop_criterion):
        #function of current weights (vector) and observations (array of vectors), true_labels (vector)
        #runs update_weight function num_iterations times, and stops if stop_criterion is reached
        last_weights = np.zeros_like(weights)
        for i in range(num_iterations):
            weights = self.update_weights(weights, observations, true_labels, learning_rate)
            if  np.amax( np.absolute( np.subtract(weights,last_weights))) < stop_criterion: #if max(|weights-last_weights|)<stop_criterion
                print("Stop criterion reached:", np.subtract(weights,last_weights))
                return weights
            last_weights = weights
        return weights
    
    def predict(self, weights, csv_file_predictors, list_predictors):
        #function of weights resulting from fit() and .csv file of features to be used as prediction
        predictors = self.init_x(csv_file_predictors,list_predictors)
        predictions = []
        for i in range(len(predictors)):
            a = np.dot(predictors[i], weights) #compute a
            pred_prob = self.sigma(a) #compute probability
            if pred_prob > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
    
    def accuracy(self, predictions, csv_file_real_y, target):
        real_y = self.init_y(csv_file_real_y, target)
        count = 0
        for i in range(len(real_y)):
            if real_y[i] == predictions[i]:
                count+=1
            else:
                continue
        accuracy = float(count)/len(real_y)
        return accuracy