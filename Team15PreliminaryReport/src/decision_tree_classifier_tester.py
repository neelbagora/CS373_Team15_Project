import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import decision_tree_classifier
import my_get_accuracy
import time
import sys

start = time.perf_counter()

# n = 1000 (fixed)
X = pd.read_csv("../data/weather_data.csv")
y = X['RainTomorrow']
del X['RainTomorrow']

# test_num indicates the range of hyperparameters being tested
# Range of Hyperparameters = (0.1 * test_num - 0.1, test_num * 0.1)
# ex: n_tests = 5, range(0.4, 0.5 (exlucisve))
test_num = int(input("Enter test number: "))

# number of tests to run per hyperparameter
n_tests = int(input("Enter number of tests per Hyperparameter: "))
step = float(input("Enter step value: "))

print(f'Hyperparameter Range being Run: ({(test_num * 0.1) - 0.1}, {test_num * 0.1}), step={step}.')

inputs = np.arange(((test_num * 0.1) - 0.1), test_num * 0.1, step).tolist()
validation_outputs = []
testing_outputs = []

# test for every hyperparameter in the list
for i in range(len(inputs)):
    validation_accuracy_score = 0
    testing_accuracy_score = 0

    # run n_tests and calculate average accuracy score obtained from
    # specified hyperparameter
    for j in range(n_tests):
        validation_accuracy_score, y_hat_testing, y_testing = decision_tree_classifier.run(X, y, inputs[i])
        testing_accuracy_score += my_get_accuracy.run(y_hat_testing, y_testing)

    validation_outputs.append(validation_accuracy_score / n_tests)
    testing_outputs.append(testing_accuracy_score / n_tests)

# form pandas dataframe from data
df = pd.DataFrame(list(zip(inputs, validation_outputs, testing_outputs)))
print(df)
print(f'Runtime {time.perf_counter() - start} seconds')

compression_opts = dict(method='zip', archive_name=f'dtc_output_data_{test_num}.csv')
df.to_csv(f'../output/dtc_output_data_{test_num}.zip', index=False, compression=compression_opts)
