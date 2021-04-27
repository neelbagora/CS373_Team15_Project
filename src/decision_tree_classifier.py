from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Input:  Pandas Dataframe X
#           dataframe containing the data to be used to form the decision
#           tree model
#         Pandas Dataframe y
#           dataframe containing the outputs corresponding to the input_data
#           provided.
#         float impurity_decrease
#           float specifying the minimum impurity decrease required to yield
#           a split in the decision tree
#         int np_seed
#           integer specifying if numpy random seed is necessary (for testing
#           and analysis of algorithm)
# Output: float accuracy_score of the validation subset
#         numpy vector y-hat (using testing data), of size 200
#         numpy vector y (using testing data), of size 200
def run(X, y, impurity_decrease, np_seed=None):
    if np_seed:
        np.random.seed(np_seed)

    # randomly select indices to be used for training, validation, and testing
    testing_indices = list(np.random.choice(len(X), len(X), replace=False))

    # training subset will utilize 60% of the data
    X_training = X.iloc[testing_indices[0 : 600]]
    y_training = y.iloc[testing_indices[0 : 600]].tolist()

    # validation subset will utilize 20% of the remaining data
    X_validation = X.iloc[testing_indices[600 : 800]]
    y_validation = y.iloc[testing_indices[600 : 800]].tolist()

    # testing subset will utilize the remaining 20% of data
    X_testing = X.iloc[testing_indices[800 : 1000]]
    y_testing = y.iloc[testing_indices[800:]].tolist()
    # print(testing_data)

    clf = DecisionTreeClassifier(min_impurity_decrease=impurity_decrease)

    clf = clf.fit(X_training, y_training)
    clf_testing_predict = clf.predict(X_testing)

    clf_validation_predict = clf.predict(X_validation)

    validation_accuracy_score = 0
    for i in range(len(clf_validation_predict)):
        if (clf_validation_predict[i] == y_validation[i]):
            validation_accuracy_score = validation_accuracy_score + 1
    validation_accuracy_score = validation_accuracy_score / len(clf_validation_predict)

    return (validation_accuracy_score, clf_testing_predict, y_testing)
