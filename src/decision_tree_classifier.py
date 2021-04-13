from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Input: Pandas Dataframe X
#           dataframe containing the data to be used to form the decision
#           tree model
#        Pandas Dataframe y
#           dataframe containing the outputs corresponding to the input_data
#           provided.
#        float impurity_decrease
#           float specifying the minimum impurity decrease required to yield
#           a split in the decision tree
#        int np_seed
#           integer specifying if numpy random seed is necessary (for testing
#           and analysis of algorithm)
# Output: float accuracy_score of the validation subset
#         numpy vector y-hat (using testing data), of size 200
#         numpy vector y (using testing data), of size 200
def decisiontreeclassifier(X, y, impurity_decrease, np_seed=None):
    if np_seed:
        np.random.seed(np_seed)

    # randomly select indices to be used for training, validation, and testing
    testing_indices = list(np.random.choice(len(X), len(X), replace=False))

    # training subset will utilize 60% of the data
    training_data = X.iloc[testing_indices[0 : 600]]
    training_data_output = y.iloc[testing_indices[0 : 600]]

    # validation subset will utilize 20% of the remaining data
    validation_data = X.iloc[testing_indices[600 : 800]]
    validation_data_output = y.iloc[testing_indices[600 : 800]]

    # testing subset will utilize the remaining 20% of data
    testing_data = X.iloc[testing_indices[800 : 1000]]
    # print(testing_data)

    clf = DecisionTreeClassifier(min_impurity_decrease=impurity_decrease)
    clf = clf.fit(training_data, training_data_output)
    clf_testing_predict = clf.predict(testing_data)
    testing_output = np.array(y.iloc[testing_indices[800:]])
    return (accuracy_score(clf.predict(validation_data), validation_data_output), clf_testing_predict, testing_output)
