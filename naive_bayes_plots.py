import pandas as pd
import numpy as np
import naive_bayes_classifier
from plotnine import *

df = pd.read_csv('../data/weather_data.csv')

# Split up X and y
y = df['RainTomorrow']
X = df.drop(columns=['RainTomorrow']) # Get rid of prediction

y_pred, test_score, train_score = naive_bayes_classifier.run(X, y, alph=0.5)
print(y_pred)
print('something')