import pandas as pd
from sklearn.preprocessing import LabelEncoder

# For Documentation Purposes
'''
    Cleans raw data by removing unnecessary features, converts categorical
    variables to numerical values, and writes new data frame into
    'weather_data.csv', and compressing into 'weather_data.zip'
'''
data = pd.read_csv("../data/weatherAUSData.csv")

# drop unnecessary data columns
del data['Date']
del data['Location']
del data['Evaporation']
del data['Sunshine']

# convert categorical variables to numerical variables
le = LabelEncoder()
winds_label = data['WindGustDir']
winds_label.append(data['WindDir9am'])
winds_label.append(data['WindDir3pm'])
winds_label = list(set(winds_label))

# remove NaN values
winds_label = [x for x in winds_label if str(x) != 'nan']

le_fit = le.fit(winds_label)
le_transform = le.transform(winds_label)

# form dictionary mapping wind labels to numerical labels
winds_dict = dict(zip(winds_label, le_transform.tolist()))

# re-do process for Yes and No variables
yes_no = ['Yes', 'No']
le_fit = le.fit(yes_no)
le_transform  = le.transform(yes_no)
yes_no_dict = dict(zip(yes_no, le_transform.tolist()))

# convert categorical data to numerical data
data = data.replace({'WindGustDir': winds_dict})
data = data.replace({'WindDir9am': winds_dict})
data = data.replace({'WindDir3pm': winds_dict})
data = data.replace({"RainToday": yes_no_dict})
data = data.replace({"RainTomorrow": yes_no_dict})

# remove NaN values
data.fillna(data.mean(), inplace=True)

# compress data into a zip file to be used for analysis
compression_opts = dict(method='zip', archive_name='weather_data.csv')
data.to_csv('weather_data.zip', index=False, compression=compression_opts)
