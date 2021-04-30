import pandas as pd
from plotnine import *

data = pd.read_csv("../output/dtc_output_data.csv")
print(ggplot(data, aes(x = 'Min Impurity Decrease')) + geom_line(aes(y='Validation Accuracy'), color='red') + geom_line(aes(y='Testing Accuracy'), color='blue') + labs(y = "Accuracy") + ggtitle("Min Impurity Decrease Versus Accuracy"))
