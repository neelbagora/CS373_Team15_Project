import sklearn.tree.DecisionTreeClassifier
import pandas as pd
import numpy as np

# n = 1000 (fixed)
data = pd.read_csv("weather_data.csv")

# randomly select indices to be used for training, validation, and testing
testing_indices = list(np.random.choice(len(data), len(data), replace=False))


# training subset will utilize 60% of the data
training_data = data.iloc[testing_indices[0 : 600]]
print(training_data)

# validation subset will utilize 20% of the remaining data
validation_data = data.iloc[testing_indices[600 : 800]]
print(validation_data)

# testing subset will utilize the remaining 20% of data
testing_data = data.iloc[testing_indices[800 : 1000]]
print(testing_data)
