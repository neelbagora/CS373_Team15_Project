import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def run(X, y , alph, np_seed=None):
  if np_seed:
    np.random.seed(np_seed)

  # Randomly split up the dataset for training, validation and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, train_size=0.2)
  y_test = y_test.to_numpy()

  # Training
  clf = BernoulliNB(alpha=alph)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  # Accuracy Output
  #print('Test Accuracy : %.4f' % clf.score(X_test, y_test))
  #print('Training Accuracy : %.3f' % clf.score(X_train, y_train))


  return y_pred, clf.score(X_test, y_test), clf.score(X_train, y_train)

df = pd.read_csv('../data/weather_data.csv')

# Convert values of columns to Integers
df['RainToday'] = df['RainToday']
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
num_test = 100

# Testing for alpha
for alpha in alphas:
  test_avg = 0
  train_avg = 0
  for i in range(num_test):
    y_pred, test_score, train_score = run(X, y, alph=alpha)
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
df.to_csv(f'../output/bayes_output_data.csv.zip', index=False, compression=compression_opts)