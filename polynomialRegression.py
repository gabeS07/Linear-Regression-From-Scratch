# Polynomaial Regression from Scratch
import numpy as np
import matplotlib.pyplot as plt
import random
fig, ax = plt.subplots()
# dataset

X = np.array([
    -5.0, -4.7, -4.4, -4.1, -3.8, -3.5, -3.2, -2.9, -2.6, -2.3,
    -2.0, -1.7, -1.4, -1.1, -0.8, -0.5, -0.2,  0.1,  0.4,  0.7,
     1.0,  1.3,  1.6,  1.9,  2.2,  2.5,  2.8,  3.1,  3.4,  3.7,
     4.0,  4.3,  4.6,  4.9,  5.2,  5.5,  5.8,  6.1,  6.4,  6.7
])




labels = np.array([
    25.4, 23.1, 20.8, 18.9, 16.7, 14.6, 12.5, 10.9,  9.3,  7.8,
     6.5,  5.3,  4.6,  3.9,  3.5,  3.1,  2.9,  2.8,  3.0,  3.4,
     4.0,  4.9,  5.9,  7.2,  8.7, 10.4, 12.3, 14.5, 16.9, 19.6,
    22.5, 25.7, 29.1, 32.8, 36.7, 40.9, 45.3, 50.0, 54.9, 60.1
])


# plot points


# normalize features to prevent weights from getting too large and causing overflow


ax.scatter(X, labels, color='blue')

# compute and store other features (x^2, x^3, x^4, etc)
features = np.array([X, X**2, X**3, X**4, X**5]) 

def square_trick(bias, weights, feature, label, learning_rate):
    y_pred = bias + np.dot(weights, feature)
    bias += learning_rate* (label - y_pred)
    for i in range(len(weights)):
        weights[i] += learning_rate*feature[i]*(label-y_pred) # for each weight, do square trick
        
    return bias, weights

def linear_regression(features, labels, learning_rate, epochs):
    bias = random.random()
    weights = np.zeros(features.shape[0]) # set weights to zero vector of same length as number of features

    # training loop
    for epoch in range(epochs):
        i = np.random.randint(0,len(features[0])) # choose random point
        feature = features[:, i] # gets the i-th column (so X^2, X^3, X, etc) in features, which would be the weights for that point
        label = labels[i]
        bias, weights = square_trick(bias, weights, feature, label, learning_rate)

        # L2 regularization
        for i in range(len(weights)):
            weights[i] *= (1 - (learning_rate*0.1)) 
                     

    
    return bias, weights

bias, weights = linear_regression(features, labels, learning_rate = 0.01, epochs = 100) # train model
print(f'Bias: {bias}, Weights: {weights}')
def predict(feature):
    return bias + np.dot(weights, feature)

ax.plot(X, predict(features), color='red') # plot line of best fit

plt.show()

