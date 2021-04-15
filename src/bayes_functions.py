import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def bayesClassifier(X, y ,alpha=None, np_seed=None):
  if np_seed:
    np.random.seed(np_seed)
  
  # Randomly split up the dataset for training, validation and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, train_size=0.2)
  y_test = y_test.to_numpy()

  # Training
  clf = BernoulliNB() 
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  # Accuracy Output
  print('Test Accuracy : %.4f' % clf.score(X_test, y_test))
  print('Training Accuracy : %.3f' % clf.score(X_train, y_train))

  return y_pred