import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from decision_tree_classifier import decisiontreeclassifier
from sklearn.metrics import accuracy_score

# n = 1000 (fixed)
X = pd.read_csv("../data/weather_data.csv")
y = X['RainTomorrow']
del X['RainTomorrow']

seed = 27

# Test 1
output = decisiontreeclassifier(X, y, 0.00, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 1 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.9
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.9

# Test 2
output = decisiontreeclassifier(X, y, 0.01, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 2 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.94
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.965

# Test 3
output = decisiontreeclassifier(X, y, 0.005, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 3 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.905
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.94

# Test 4
output = decisiontreeclassifier(X, y, 0.007, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 4 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.93
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.94

# Test 5
output = decisiontreeclassifier(X, y, 0.0085, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 5 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.95
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.95

# Test 6
output = decisiontreeclassifier(X, y, 0.00925, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 6 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.95
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.95

# Test 7
output = decisiontreeclassifier(X, y, 0.00950, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 7 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.95
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.95

# Test 8
output = decisiontreeclassifier(X, y, 0.00958, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 8 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.94
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.965

# Test 9
output = decisiontreeclassifier(X, y, 0.04, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 9 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.945
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.915

'''
    The best accuracy achieved through validation is 0.95 using a min_impurity_decrease
    of around 0.0085-0.00957, however, the best testing accuracy was found with
    a min_impurity_decrease of 0.00958 to 0.00999 to ~0.03
'''
