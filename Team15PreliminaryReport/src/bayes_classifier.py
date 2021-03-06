import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import my_get_accuracy

df = pd.read_csv("../data/weather_data.csv", sep=',')
# df = pd.read_csv('')

# Convert values of columns to Integers
df['RainToday'] = df['RainToday'].astype(int)
df['RainTomorrow'] = df['RainTomorrow'].astype(int)
df['WindGustDir'] = df['WindGustDir'].astype(int)

# Split up X and y
y = df['RainTomorrow']
X = df.drop(columns=['RainTomorrow']) # Get rid of prediction

# Split the Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_test = y_test.to_numpy()

# Train data
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy Output
accuracy = my_get_accuracy.run(y_test, y_pred)
print('Test Accuracy : %.4f' % clf.score(X_test, y_test))
print('Training Accuracy : %.3f' % clf.score(X_train, y_train))

# Gridsearch for best alpha
# will implement own version
from sklearn.model_selection import GridSearchCV
params = {'alpha': [0.01, 0.1, 0.5, 1.0],}

bernoulli_nb_grid = GridSearchCV(BernoulliNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5)
bernoulli_nb_grid.fit(X,y)

print('Train Accuracy : %.3f' % bernoulli_nb_grid.best_estimator_.score(X_train, y_train))
print('Test Accuracy : %.3f' % bernoulli_nb_grid.best_estimator_.score(X_test, y_test))
print('Best Accuracy Through Grid Search : %.3f' % bernoulli_nb_grid.best_score_)
print('Best Parameters : ',bernoulli_nb_grid.best_params_)

# Test with adjusted alpha
# Get X and y
y = df.RainTomorrow
X = df.drop(columns=['RainTomorrow'])

# Split the Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_test = y_test.to_numpy()

clf = BernoulliNB(alpha=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


accuracy = my_get_accuracy.run(y_test, y_pred)

print('Test Accuracy : %.4f' % clf.score(X_test, y_test))
print('Training Accuracy : %.3f' % clf.score(X_train, y_train))