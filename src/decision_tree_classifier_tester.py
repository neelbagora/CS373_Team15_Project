import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import decision_tree_classifier
import time
import sys
import my_get_accuracy

start = time.perf_counter()

# n = 1000 (fixed)
X = pd.read_csv("../data/weather_data.csv")
y = X['RainTomorrow']
del X['RainTomorrow']

# number of tests to run per hyperparameter
n_tests = int(input("Enter number of tests per Hyperparameter: "))
step = float(input("Enter step value: "))

print(f'Hyperparameter Range being Run: ({0}, {2}), step={step}.')

inputs = np.arange(0, 2, step).tolist()
validation_outputs = []
testing_outputs = []

# test for every hyperparameter in the list
for i in range(len(inputs)):
    validation_accuracy_score = 0
    testing_accuracy_score = 0

    # run n_tests and calculate average accuracy score obtained from
    # specified hyperparameter
    for j in range(n_tests):
        validation_score, y_hat_testing, y_testing = decision_tree_classifier.run(X, y, inputs[i])
        validation_accuracy_score = validation_accuracy_score + validation_score

        accuracy_score = my_get_accuracy.run(y_hat_testing, y_testing, True)
        accuracy_score = accuracy_score / len(y_hat_testing)
        testing_accuracy_score += accuracy_score

    validation_outputs.append(validation_accuracy_score / n_tests)
    testing_outputs.append(testing_accuracy_score / n_tests)

# form pandas dataframe from data
df = pd.DataFrame(list(zip(inputs, validation_outputs, testing_outputs)))
df.columns = ['Min Impurity Decrease', 'Validation Accuracy', 'Testing Accuracy']
print(df)
print(f'Runtime {time.perf_counter() - start} seconds')

compression_opts = dict(method='zip', archive_name=f'dtc_output_data.csv')
df.to_csv(f'../output/dtc_output_data.zip', index=False, compression=compression_opts)
