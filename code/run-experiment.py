import numpy as np
import matplotlib.pyplot as plt

import model

if __name__ == "__main__":

    # some dummy data
    X = np.array([[1,2,3],
                  [1,4,2]])
    Y = np.array([1,0])

    # randomized start weights
    weights = np.random.rand(3)
    # and a fixed learning rate
    learning_rate = 1e-1

    #see how it does over 20 epochs
    for i in range(20):
        weights = update_weights(weights, X, Y, learning_rate)
        print('epoch {}'.format(i))
        for j in range(2):
            print('prob(Y={}) ={}'.format(j,predicted_probability(weights,Xobs[j])))
    print(weights)
