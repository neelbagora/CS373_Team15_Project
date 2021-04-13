import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from decision_tree_classifier import decisiontreeclassifier
from sklearn.metrics import accuracy_score
import time

start = time.perf_counter()

# n = 1000 (fixed)
X = pd.read_csv("../data/weather_data.csv")
y = X['RainTomorrow']
del X['RainTomorrow']

n_tests = 500

test_num = 3
inputs = np.arange(test_num - 1, test_num, 0.01).tolist()
validation_outputs = []
testing_outputs = []

# test for every hyperparameter in the list
for i in range(len(inputs)):
    validation_accuracy_score = 0
    testing_accuracy_score = 0

    # run n_tests and calculate average accuracy score obtained from
    # specified hyperparameter
    for j in range(n_tests):
        output = decisiontreeclassifier(X, y, inputs[i])

        validation_accuracy_score += output[0]
        y_hat_testing = output[1]
        y_testing = output[2]

        testing_accuracy_score += accuracy_score(y_hat_testing, y_testing)
    validation_outputs.append(validation_accuracy_score / n_tests)
    testing_outputs.append(testing_accuracy_score / n_tests)

# form pandas dataframe from data
df = pd.DataFrame(list(zip(inputs, validation_outputs, testing_outputs)))
print(df)
print(time.perf_counter() - start)

compression_opts = dict(method='zip', archive_name=f'output_data_{test_num}.csv')
df.to_csv(f'output_data_{test_num}.zip', index=False, compression=compression_opts)
