import pandas as pd
import numpy as np
import naive_bayes_classifier
import my_get_accuracy
from plotnine import *
from sklearn.naive_bayes import BernoulliNB

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
          test_score, y_hat_testing, y_testing = naive_bayes_classifier.run(X, y, alph=alpha)
          temp_test += test_score
          temp_trains += my_get_accuracy.run(y_hat_testing, y_testing, True)
      y_scores.append(temp_test / 10)
      y_trains.append(temp_trains / 10)
  ###

  test_df = pd.DataFrame(list(zip(alp, y_scores, y_trains)), columns=['alphas', 'train_score', 'test_score'])
  test_df
  p = ggplot(test_df) + geom_line(aes(x='alphas', y='train_score'), color='blue') \
                + geom_line(aes(x='alphas', y='test_score'), color='red') \
                + labs(y='Accuracy', x='Parameter Value') \
                + ggtitle('Alpha vs. Accuracy')
  return p

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
  validation_accuracy_score, y_hat_testing, y_testing = naive_bayes_classifier.run(X, y, alph=0.6)


  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0

  n_tests = 35
  specificity = 0
  sensitivity = 0
  for i in range(n_tests):
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
      sensitivity = sensitivity + (true_positive / (true_positive + false_negative))
      specificity = specificity + (true_negative / (true_negative + false_positive))

  sensitivity_list.append(sensitivity / n_tests)
  specificity_list.append(specificity / n_tests)

  sensitivity_list.append(0)
  specificity_list.append(1)

  # plot results
  df = pd.DataFrame(list(zip(specificity_list, sensitivity_list)), columns =['Specificity', 'Sensitivity'])

  return ggplot(df) + geom_line(aes(x='Specificity', y="Sensitivity"), color='red') + xlim(0, 1) + ylim(0, 1) + geom_abline(slope=-1, intercept=1, linetype='dotted') + ggtitle("ROC Curve of Naive Bayes Classifier")

testnum = int(input('Input 0 for the Accuracy Curve, Input 1 for the ROC Curve: '))

if testnum == 0:
    print(get_accuracy_graph())
else:
    print(get_roc_curve())
