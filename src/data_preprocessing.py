import pandas as pd
from sklearn.preprocessing import LabelEncoder

# For Documentation Purposes
'''
    Cleans raw data by removing unnecessary features, converts literal
    strings to numerical values, and writes new data frame into
    'weather_data.csv', and compressing into 'out.zip'
'''
data = pd.read_csv("../data/weatherAUSData.csv")
del data['Date']
del data['Location']
del data['Evaporation']
del data['Sunshine']

le = LabelEncoder()
winds_label = data['WindGustDir']
winds_label.append(data['WindDir9am'])
winds_label.append(data['WindDir3pm'])
winds_label = list(set(winds_label))
winds_label = [x for x in winds_label if str(x) != 'nan']

le_fit = le.fit(winds_label)
le_transform = le.transform(winds_label)
winds_dict = dict(zip(winds_label, le_transform.tolist()))

yes_no = ['Yes', 'No']
le_fit = le.fit(yes_no)
le_transform  = le.transform(yes_no)
yes_no_dict = dict(zip(yes_no, le_transform.tolist()))

data = data.replace({'WindGustDir': winds_dict})
data = data.replace({'WindDir9am': winds_dict})
data = data.replace({'WindDir3pm': winds_dict})
data = data.replace({"RainToday": yes_no_dict})
data = data.replace({"RainTomorrow": yes_no_dict})
data = data.fillna(0)

compression_opts = dict(method='zip', archive_name='weather_data.csv')
data.to_csv('out.zip', index=False, compression=compression_opts)
