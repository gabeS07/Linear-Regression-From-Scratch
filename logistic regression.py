# logistic regression 

# Perceptron Learning Algorithm

import numpy as np
import random
import matplotlib.pyplot as plt

# data

X = np.array([ # features data
    [1.0, 1.0],
    [1.5, 1.2],
    [2.0, 1.0],
    [1.0, 2.0],
    [2.0, 2.0],
    [1.8, 1.5],
    [2.2, 1.8],

    [4.0, 4.0],
    [4.5, 3.8],
    [5.0, 4.2],
    [3.8, 4.5],
    [4.2, 5.0],
    [5.0, 5.0],
    [3.5, 4.0],
]) 


labels = np.array([ # labels
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1
])  

# splitting data into two classes for better visualization
neg_class = X[labels == 0]
pos_class = X[labels == 1]

plt.scatter(neg_class[:,0], neg_class[:,1], color='red')
plt.scatter(pos_class[:,0], pos_class[:,1], color='blue')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(features, weights, bias):
    score = np.dot(features, weights) + bias
    return sigmoid(score)

def log_loss(predicted, label):
    return - (label * np.log(predicted) + (1 - label) * np.log(1 - predicted))

def mean_error(X, labels, weights, bias):
    total = 0
    for i in range(len(X)):
        features = X[i]
        label = labels[i]
        p = predict(features, weights, bias)
        total += log_loss(p, label)
    return total / len(X)


# training

epochs = 10000
weights = np.zeros(X.shape[1])  # initialize weights
bias = 0  # initialize bias
learning_rate = 0.01

for epoch in range(epochs):
    # pick random point
    i = random.randint(0, len(X)-1)
    features = X[i]
    label = labels[i]

    # logistic trick

    weights += (label-predict(features, weights, bias ))*features * learning_rate
    bias += (label-predict(features, weights, bias)) * learning_rate

# plotting the decision boundary


x = np.linspace(0, 6, 14) # generate x values for plotting the line

if weights[1] != 0: # to avoid division by zero
    y = (weights[0]*x + bias) / -weights[1]
else:
    y = bias / -weights[0] # these formulas are derived from the decision boundary equation: w1*x1 + w2*x2 + b = 0

plt.plot(x, y, color = 'green') 
print("Trained weights: ", weights, " Trained bias:", bias, " log loss: ", mean_error(X, labels, weights, bias))
print("Prediction of [1, 1]: ", predict(np.array([1, 1]), weights, bias))
plt.show()