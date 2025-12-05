
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('AIR.csv')

# Drop unneeded columns
del df['StationId']
del df['Datetime']

# Label encode AQI_Bucket
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var = ['AQI_Bucket']

for i in var:
    df[i] = le.fit_transform(df[i]).astype(int)

# Check for nulls and drop them
df = df.dropna()

# Print the unique AQI_Bucket labels and basic info
print(df['AQI_Bucket'].unique())
print(df.describe())
print(df.corr())
print(df.info())

# Crosstab and groupby examples
print(pd.crosstab(df["PM2.5"], df["PM10"]))
print(df.groupby(["NO","NO2"]).groups)

# Value counts and categorical description
print(df["AQI_Bucket"].value_counts())
print(pd.Categorical(df["NOx"]).describe())

# Check for duplicates and drop
print(sum(df.duplicated()))
df = df.drop_duplicates()
print(sum(df.duplicated()))
