import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB

X = pd.read_csv('weatherAUS_cleaned.csv', sep=',')
y = pd.read_csv('weatherAUS_output.csv', sep=',').astype('int')
clf = BernoulliNB()
clf.fit(X, y)
test = pd.read_csv('weatherAUS_test.csv', sep=',')
print(clf.predict(test))
