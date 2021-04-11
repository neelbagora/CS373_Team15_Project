from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from decision_tree_classifier import decisiontreeclassifier

# n = 1000 (fixed)
raw_data = pd.read_csv("../data/weather_data.csv")
print(decisiontreeclassifier(raw_data, 0.01427854, 26))
