from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Input: Pandas Dataframe input_data X
#           dataframe containing the data to be used to form the decision
#           tree model
#        Pandas Dataframe outputs y
#           dataframe containing the outputs corresponding to the input_data
#           provided.
#        float impurity_decrease
#           float specifying the minimum impurity decrease required to yield
#           a split in the decision tree
#        int np_seed
#           integer specifying if numpy random seed is necessary (for testing
#           and analysis of algorithm)
# Output: numpy vector y-hat, of size 200
#         float accuracy_score of the validation subset
def decisiontreeclassifier(input_data, outputs, impurity_decrease, np_seed=None):
    if np_seed:
        np.random.seed(np_seed)

    # randomly select indices to be used for training, validation, and testing
    testing_indices = list(np.random.choice(len(input_data), len(input_data), replace=False))

    # training subset will utilize 60% of the data
    training_data = input_data.iloc[testing_indices[0 : 600]]
    training_data_output = outputs.iloc[testing_indices[0 : 600]]

    # validation subset will utilize 20% of the remaining data
    validation_data = input_data.iloc[testing_indices[600 : 800]]
    validation_data_output = outputs.iloc[testing_indices[600 : 800]]

    # testing subset will utilize the remaining 20% of data
    testing_data = input_data.iloc[testing_indices[800 : 1000]]
    # print(testing_data)

    clf = DecisionTreeClassifier(min_impurity_decrease=impurity_decrease)
    clf = clf.fit(training_data, training_data_output)
    clf_testing_predict = clf.predict(testing_data)
    testing_output = np.array(outputs.iloc[testing_indices[800:]])
    return (clf_testing_predict, testing_output, accuracy_score(clf.predict(validation_data), validation_data_output))
