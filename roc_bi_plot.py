import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import RandomForestWithCluster as rfc

data_path = 'dataset/darwin.csv'
target = 'class'

dataframe = pd.read_csv(data_path)

print(dataframe.head())
# Fill missing values with avg
dataframe.fillna(dataframe.mean(), inplace=True)

train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)

model = rfc.RandomForestWithCluster(terminal_debug=True)

model.fit(train_df, target)

model.predict(test_df, target)

y_probs, y_test = model.predict_proba(test_df, target)


# Compute ROC curve and ROC area under the curve (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

##############################################################################################################
# Read the dataset
df = pd.read_csv(data_path)

# Fill missing values with avg
df.fillna(df.mean(), inplace=True)
# Separate features and target
X = df.drop(target, axis=1)
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Get predicted probabilities for positive class
y_probs2 = rf_classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area under the curve (AUC)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_probs2)
roc_auc2 = auc(fpr2, tpr2)

# # Plot both ROC curves in the same plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve RFWOC (AUC = {:.2f})'.format(roc_auc))
plt.plot(fpr2, tpr2, color='green', lw=2, label='ROC curve RF (AUC = {:.2f})'.format(roc_auc2))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

