import pandas as pd

# For Documentation Purposes
'''
    Cleans raw data by removing unnecessary features and
    writing new data frame into 'weather_data.csv' and compressing
    into 'out.zip'
'''
data = pd.read_csv("weatherAUSData.csv")
del data['Date']
del data['Location']
del data['Evaporation']
del data['Sunshine']
compression_opts = dict(method='zip', archive_name='weather_data.csv')
data.to_csv('out.zip', index=False, compression=compression_opts)

print(data)
