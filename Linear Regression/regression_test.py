# Author: Shibo Tang 
import regression as re
import numpy as np


# Linear regression
print('Regression:')
testData = re.read_csv("regression/test.csv")
trainData = re.read_csv("regression/train.csv")

# Batch gradient descent algorithm
BGD1 = re.bgd(0.25, trainData['x'], trainData['y'])
costTest = re.cost(BGD1['weight'], testData['x'], testData['y'])
print('Batch gradient descent algorithm:')
print('Final cost function value of the training data:', BGD1['cost'][-1])
print('Final cost function value of the test data:', costTest)
print('r:', BGD1['r'])
print('Final weight:', BGD1['weight'])

# Stochastic gradient descent (SGD) algorithm
SGD1 = re.sgd(0.25, trainData['x'], trainData['y'])
costTest = re.cost(SGD1['weight'], testData['x'], testData['y'])
print('Stochastic gradient descent algorithm:')
print('Final cost function value of the training data:', SGD1['cost'][-1])
print('Final cost function value of the test data:', costTest)
print('r:', SGD1['r'])
print('Final weight:', SGD1['weight'])

# optimal weight vector with analytical form
w = re.analytical(trainData['x'], trainData['y'])
cost_train = re.cost(w, trainData['x'], trainData['y'])
cost_test = re.cost(w, testData['x'], testData['y'])
print('The optimal weight vector with an analytical form:', w)
print('The cost function value of the training data:', cost_train)
print('The cost function value of the test data:', cost_test)