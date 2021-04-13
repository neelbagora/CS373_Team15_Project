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

seed = 28

# Test 1
output = decisiontreeclassifier(X, y, 0.00, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 1 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.915
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.92

# Test 2
output = decisiontreeclassifier(X, y, 0.01, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 2 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.94
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

# Test 3
output = decisiontreeclassifier(X, y, 0.005, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 3 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.945
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.915

# Test 4
output = decisiontreeclassifier(X, y, 0.007, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 4 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.945
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

# Test 5
output = decisiontreeclassifier(X, y, 0.0085, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 5 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.94
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

# Test 6
output = decisiontreeclassifier(X, y, 0.00775, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 6 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.945
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

# Test 7
output = decisiontreeclassifier(X, y, 0.007875, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 7 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.945
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

# Test 8
output = decisiontreeclassifier(X, y, 0.0079, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 8 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.945
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

# Test 9
output = decisiontreeclassifier(X, y, 0.008, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 9 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.945
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

# Test 9
output = decisiontreeclassifier(X, y, 0.0081, seed)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("TEST 9 Results")
print("Validation Accuracy Score ", validation_accuracy_score) # 0.94
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing)) # 0.925

'''
    Using a min_impurity_decrease of around 0.007-0.008, we achieve the best
    validation accuracy and testing accuracy of 0.94 and 0.925, respectively.
'''
