import pandas as pd
import decision_tree_classifier
from plotnine import *

X = pd.read_csv("../data/weather_data.csv")
y = X['RainTomorrow']
del X['RainTomorrow']

true_positive_rate = 0
false_positive_rate = 0

sensitivity_list = []
specificity_list = []

sensitivity_list.append(1)
specificity_list.append(0)

# The range where the highest accuracy occurs

# min_impurity_decrease of 0.02 is our optimal value
validation_score, y_hat_testing, y_testing = decision_tree_classifier.run(X, y, 0.02)
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# calculate tp, tn, fp, fn
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

(ggplot(df) + geom_line(aes(x='Specificity', y="Sensitivity"), color='red') + xlim(0, 1) + ylim(0, 1) + geom_abline(slope=-1, intercept=1) + ggtitle("ROC Curve of Decision Tree Classifier"))
