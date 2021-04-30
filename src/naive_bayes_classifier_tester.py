import naive_bayes_classifier
import pandas as pd
import numpy as np

df = pd.read_csv('../data/weather_data.csv')

# Split up X and y
y = df['RainTomorrow']
X = df.drop(columns=['RainTomorrow']) # Get rid of prediction

# containers
testing_outputs = []
train_outputs = []

num_test = int(input("Enter number of tests per Hyperparameter: "))
step = float(input("Enter step value: "))

print(f'Hyperparameter Range being Run: ({0}, {0.5}), step={step}.')

# test
alphas = np.arange(0, 0.5, step)

# Testing for alpha
for alpha in alphas:
    test_avg = 0
    train_avg = 0
    for i in range(num_test):
        y_pred, test_score, train_score = naive_bayes_classifier.run(X, y, alph=alpha)
        test_avg += test_score
        train_avg += train_score
    test_avg = test_avg / num_test
    train_avg = train_avg / num_test
    testing_outputs.append(test_avg)
    train_outputs.append(train_avg)

# to df
df = pd.DataFrame(list(zip(alphas, train_outputs, testing_outputs)))
print(df)
compression_opts = dict(method='zip', archive_name=f'bayes_output_data.csv')
df.to_csv(f'../output/bayes_output_data.zip', index=False, compression=compression_opts)
