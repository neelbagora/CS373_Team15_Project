import naive_bayes_classifier
import pandas as pd
import numpy as np
import my_get_accuracy

df = pd.read_csv('../data/weather_data.csv')

# Split up X and y
y = df['RainTomorrow']
X = df.drop(columns=['RainTomorrow']) # Get rid of prediction

# containers
testing_outputs = []
train_outputs = []

num_test = int(input("Enter number of tests per Hyperparameter: "))
step = float(input("Enter step value: "))

# *step of 0.01 used in experiments conducted*
print(f'Hyperparameter Range being Run: ({0.01}, {0.5}), step={step}.')

# test
alphas = np.arange(0.01, 0.5, step)

# Testing for alpha
for alpha in alphas:
    test_avg = 0
    train_avg = 0
    for i in range(num_test):
        test_score, y_hat_testing, y_testing = naive_bayes_classifier.run(X, y, alph=alpha)
        test_avg += test_score
        train_avg += my_get_accuracy.run(y_hat_testing, y_testing, True)
    test_avg = test_avg / num_test
    train_avg = train_avg / num_test
    testing_outputs.append(test_avg)
    train_outputs.append(train_avg)

# to df
df = pd.DataFrame(list(zip(alphas, train_outputs, testing_outputs)), columns=['Alpha', 'Train Score', 'Testing Score'])
print(df)
compression_opts = dict(method='zip', archive_name=f'bayes_output_data.csv')
df.to_csv(f'../output/bayes_output_data.zip', index=False, compression=compression_opts)
