import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
import my_get_accuracy

def run(X, y , alph, np_seed=None):
  if np_seed:
    np.random.seed(np_seed)

  # Randomly split up the dataset for training, validation and testing
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, train_size=0.2)
  testing_indices = list(np.random.choice(len(X), len(X), replace=False))

  X_training = X.iloc[testing_indices[0:600]]
  y_training = y.iloc[testing_indices[0:600]].tolist()

  X_validation = X.iloc[testing_indices[600:800]]
  y_validation = y.iloc[testing_indices[600:800]].tolist()

  X_testing = X.iloc[testing_indices[800:1000]]
  y_testing = y.iloc[testing_indices[800:1000]].tolist()

  #y_test = y_test.to_numpy()

  # Training
  clf = BernoulliNB(alpha=alph)
  clf.fit(X_training, y_training)
  y_pred = clf.predict(X_testing)

  y_hat_validation = clf.predict(X_validation)
  y_hat_testing = clf.predict(X_testing)

  return my_get_accuracy.run(y_hat_validation, y_validation, True), y_hat_testing, y_testing
