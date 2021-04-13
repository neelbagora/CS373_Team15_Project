from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from decision_tree_classifier import decisiontreeclassifier
from sklearn.metrics import accuracy_score

# n = 1000 (fixed)
raw_data = pd.read_csv("../data/weather_data.csv")
outputs = raw_data['RainTomorrow']
del raw_data['RainTomorrow']
output = decisiontreeclassifier(raw_data, outputs, 0.01427854)

validation_accuracy_score = output[0]
y_hat_testing = output[1]
y_testing = output[2]

print("Validation Accuracy Score ", validation_accuracy_score)
print("Testing Accuracy Score ", accuracy_score(y_hat_testing, y_testing))
