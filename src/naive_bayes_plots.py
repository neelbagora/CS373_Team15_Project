import pandas as pd
import numpy as np
import naive_bayes_classifier
import my_get_accuracy
from plotnine import *


# BernoulliNB classifier function for plots. 
def brun(X, y , alph, np_seed=None):
    if np_seed:
        np.random.seed(np_seed)

  # Randomly split up the dataset for training, validation and testing
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, train_size=0.2)
    testing_indices = list(np.random.choice(len(X), len(X), replace=False))

    X_training = X.iloc[testing_indices[0:600]]
    y_training = y.iloc[testing_indices[0:600]]

    X_validation = X.iloc[testing_indices[600:800]]
    y_validation = y.iloc[testing_indices[600:800]]

    X_testing = X.iloc[testing_indices[800:1000]]
    y_testing = y.iloc[testing_indices[800:1000]]

  #y_test = y_test.to_numpy()

  # Training
    clf = BernoulliNB(alpha=alph)
    clf.fit(X_training, y_training)
    y_pred = clf.predict(X_testing)

    return y_pred, y_testing, y_training

def get_accuracy_graph():
  df = pd.read_csv('../data/weather_data.csv')
  # Split up X and y
  y = df['RainTomorrow']
  X = df.drop(columns=['RainTomorrow']) # Get rid of prediction

  # test
  alphas = np.arange(0.01, 0.5, 0.01).tolist()

  # containers
  y_scores = []
  y_trains = []
  alp = []

  for alpha in alphas:
      alp.append(alpha)
      temp_test = 0
      temp_trains = 0
      for i in range(10):
          y_pred, test_score, train_score = naive_bayes_classifier.run(X, y, alph=alpha)
          temp_test += test_score
          temp_trains += train_score
      y_scores.append(temp_test / 10)
      y_trains.append(temp_trains / 10)
  ###

  test_df = pd.DataFrame(list(zip(alp, y_scores, y_trains)), columns=['alphas', 'train_score', 'test_score'])
  test_df
  p = ggplot(test_df) + geom_line(aes(x='alphas', y='train_score'), color='blue') \
                + geom_line(aes(x='alphas', y='test_score'), color='red') \
                + labs(y='Accuracy', x='Parameter Value') \
                + ggtitle('Accuracy vs Alpha') 
  print(p)

def get_roc_curve():
  df = pd.read_csv('../data/weather_data.csv')

  # Split up X and y
  y = df['RainTomorrow']
  X = df.drop(columns=['RainTomorrow']) # Get rid of prediction

  true_positive_rate = 0
  false_positive_rate = 0

  sensitivity_list = [] # false neg
  specificity_list = [] # false pos

  sensitivity_list.append(1)
  specificity_list.append(0)

  # testing -> actula values
  # yhat -> pred
  y_hat_testing, y_testing,y_training =  brun(X, y, alph=0.6)
  y_testing = y_testing.to_numpy()


  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0

  # for training
  for x in range(len(y_hat_testing)):
      # y-hat = 1
      if (y_hat_testing[x] == 1):
          # y = 1 (True Positive)
          if (y_testing[x] == 1):
              true_positive = true_positive + 1
          # y = 0 (False Positive)
          else:
              false_positive = false_positive + 1
      # y-hat = 0
      else:
          # y = 1 (False Negative)
          if (y_testing[x] == 1):
              false_negative = false_negative + 1
          # y = 0 (True Negative)
          else:
              true_negative = true_negative + 1
              
  # calculate sensitivity and specificity
  sensitivity = true_positive / (true_positive + false_negative)
  specificity = true_negative / (true_negative + false_positive)

  sensitivity_list.append(sensitivity)
  specificity_list.append(specificity)

  sensitivity_list.append(0)
  specificity_list.append(1)
  
  # plot results
  df = pd.DataFrame(list(zip(specificity_list, sensitivity_list)), columns =['Specificity', 'Sensitivity'])

  return ggplot(df) + geom_line(aes(x='Specificity', y="Sensitivity"), color='red') + xlim(0, 1) + ylim(0, 1) + geom_abline(slope=-1, intercept=1, linetype='dotted') + ggtitle("ROC Curve of Naive Bayes Classifier")
