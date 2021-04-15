import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import bayes_functions

df = pd.read_csv('../data/weather_data.csv')

# Convert values of columns to Integers
df['RainToday'] = df['RainToday'].astype(int)
df['RainTomorrow'] = df['RainTomorrow'].astype(int)
df['WindGustDir'] = df['WindGustDir'].astype(int)

# Split up X and y
y = df['RainTomorrow']
X = df.drop(columns=['RainTomorrow']) # Get rid of prediction

# containers
testing_outputs = []
train_outputs = []

# test
alphas = np.arange(0.0, 1.01, 0.01)
alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
folds = 3

# Testing for alpha
for alpha in alphas:
  test_avg = 0
  train_avg = 0
  for i in range(100):
    y_pred, test_score, train_score = bayes_functions.run(X, y, alph=alpha)
    test_avg += test_score
    train_avg += train_score
  test_avg = test_avg / 100
  train_avg = train_avg / 100
  testing_outputs.append(test_avg)
  train_outputs.append(train_avg)

# to df
df = pd.DataFrame(list(zip(alphas, train_outputs, testing_outputs)))
print(df)
compression_opts = dict(method='zip', archive_name=f'bayes_output_data.csv')
df.to_csv(f'../output/bayes_output_data.csv.zip', index=False, compression=compression_opts)