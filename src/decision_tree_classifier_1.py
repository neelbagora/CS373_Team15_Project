from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# n = 1000 (fixed)
raw_data = pd.read_csv("../data/weather_data.csv")
outputs = raw_data['RainTomorrow']
del raw_data['RainTomorrow']

# randomly select indices to be used for training, validation, and testing
testing_indices = list(np.random.choice(len(raw_data), len(raw_data), replace=False))

# training subset will utilize 60% of the data
training_data = raw_data.iloc[testing_indices[0 : 600]]
training_data_output = outputs.iloc[testing_indices[0 : 600]]
# print(training_data)

# validation subset will utilize 20% of the remaining data
validation_data = raw_data.iloc[testing_indices[600 : 800]]
validation_data_output = outputs.iloc[testing_indices[600 : 800]]
# print(validation_data)

# testing subset will utilize the remaining 20% of data
testing_data = raw_data.iloc[testing_indices[800 : 1000]]
# print(testing_data)

'''
clf = DecisionTreeClassifier(min_impurity_decrease=0.5)
fit_clf = clf.fit(training_data, training_data_output)
print(fit_clf.predict(validation_data))'''
