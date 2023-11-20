import pandas as pd
from sklearn.model_selection import train_test_split
from rfwoc.model import RandomForestWithCluster
from rfwoc.compare import compare_roc_curve, compare_metrics

dataframe = pd.read_csv('dataset/graduation.csv')
dataframe.fillna(dataframe.mean(), inplace=True)
train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)

model = RandomForestWithCluster()

model.fit(train_df, 'Target')
predictions = model.predict(test_df, 'Target')


compare_roc_curve('dataset/graduation.csv', 'Target')
compare_metrics('dataset/graduation.csv', 'Target')