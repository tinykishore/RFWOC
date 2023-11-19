import pandas as pd
from sklearn.model_selection import train_test_split
import RandomForestWithCluster as rfc

dataframe = pd.read_csv('dataset/graduation.csv')
dataframe.fillna(dataframe.mean(), inplace=True)
train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)
model = rfc.RandomForestWithCluster()
model.fit(train_df, 'Target')
model.predict(test_df, 'Target')
