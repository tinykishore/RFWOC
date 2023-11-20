import pandas as pd
from sklearn.model_selection import train_test_split
from rfwoc.model import RandomForestWithCluster
from rfwoc.compare import compare_roc_curve, compare_metrics, compare_prauc_curve

data = pd.read_csv('dataset/asthma.csv')
# dataframe.fillna(dataframe.mean(), inplace=True)
# train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)

# model = RandomForestWithCluster()
#
# model.fit(train_df, 'target')
# predictions = model.predict(test_df, 'target')

# Assuming the dataset has a column named 'class' indicating the class label (1 or 0)
class_1_data = data[data['target'] == 1].sample(n=1500, random_state=42)
class_0_data = data[data['target'] == 0].sample(n=1000, random_state=42)

# Combined sampled data
sampled_data = pd.concat([class_1_data, class_0_data], axis=0)
sampled_data.to_csv('dataset/asthma_data_sampled.csv', index=False)


# compare_roc_curve('dataset/graduation.csv', 'Target')
# compare_metrics('dataset/graduation.csv', 'Target')
compare_prauc_curve('dataset/asthma_data_sampled.csv','target')

