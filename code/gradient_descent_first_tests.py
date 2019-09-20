import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import sys


def sigmoid(W_transposed, X):
    a = np.dot(W_transposed, X)
    return 1/(1+np.exp(-1*a))

def gradient_desc(weights, features, y, lrng_rate):
    
    for i in range(len(y)):
        row = features[i]
        sig = sigmoid(weights,row)
        distance = y[i] - sig
        dirct_deriv = lrng_rate * row * distance

        weights += dirct_deriv
    
    print(weights)


def main():
    source = sys.argv[1]

    #array containing the scores for quality
    quality = []
    df = pd.read_csv(source, delimiter = ';')
    quality=np.array(df['quality'])

    #array containing binary scores fo quality
    y = [0] * len(quality)
    for i in range(len(y)):
        if quality[i] > 5:
            y[i] = 1
        else:
            continue
    
    #array of ones for intercept term
    ones = [1] * len(quality)

    #arrays of features
    feat1 = np.array(df['fixed acidity'])
    feat2 = np.array(df['volatile acidity'])

    #matrix of features
    features = np.column_stack((ones,feat1,feat2))


    #n+1 dimensions where n is number of features
    weights = np.array([1.0,1.0,1.0])


    gradient_desc(weights, features, y, 0.3)


    #PLOT THE MODEL Y
    delta = (max(feat1) - min(feat1)) * 0.3
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(-100, 100)
    #ax.plot(x, 1/(1 + np.exp(-(weights[0] + weights[1]*x))))
    ax.plot(feat1, y, 'ro')
    plt.show()


if __name__ == "__main__":
    main()