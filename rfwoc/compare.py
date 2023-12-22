from random import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from rfwoc.model import RandomForestWithCluster


def compare_roc_curve(
        data_path,
        target_column
):
    dataframe = pd.read_csv(data_path)

    # Fill missing values with avg
    dataframe.fillna(dataframe.mean(), inplace=True)

    # Create Two copies of the dataframe
    dataframe_rfwoc = dataframe.copy()
    dataframe_rf = dataframe.copy()

    # For RFWOC ###############################################################
    # Split the data into training and testing sets
    rfwoc_train_df, rfwoc_test_df = train_test_split(
        dataframe_rfwoc,
        test_size=0.2,
        random_state=42
    )

    # Create a RandomForest classifier
    rfwoc_model = RandomForestWithCluster()

    # Train the classifier
    rfwoc_model.fit(rfwoc_train_df, target_column)

    # Make predictions probabilities
    rfwoc_y_probs, rfwoc_y_test = rfwoc_model.predict_proba(rfwoc_test_df, target_column)

    # Compute ROC curve and ROC area under the curve (AUC)
    rfwoc_fpr, rfwoc_tpr, thresholds = roc_curve(rfwoc_y_test, rfwoc_y_probs)
    rfwoc_roc_auc = auc(rfwoc_fpr, rfwoc_tpr)

    # For Normal RandomForest ################################################
    # Separate features and target
    x = dataframe_rf.drop(target_column, axis=1)
    y = dataframe_rf[target_column]

    # Set Data Randomly here ...
    # Actually set random data for just normal rf model
    # This random data will be used for seed in train_test_division

    # Split the data into training and testing sets
    rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create a RandomForest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Train the classifier
    rf_classifier.fit(rf_x_train, rf_y_train)

    # Make predictions probabilities
    rf_y_probs = rf_classifier.predict_proba(rf_x_test)[:, 1]

    # Compute ROC curve and ROC area under the curve (AUC)
    rf_fpr, rf_tpr, thresholds = roc_curve(rf_y_test, rf_y_probs)
    rf_roc_auc = auc(rf_fpr, rf_tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        rfwoc_fpr,
        rfwoc_tpr,
        color='darkorange',
        lw=2,
        label='RFWOC ROC curve (AUC = {:.2f})'.format(rfwoc_roc_auc)
    )
    plt.plot(
        rf_fpr,
        rf_tpr,
        color='green',
        lw=2,
        label='RF ROC curve (AUC = {:.2f})'.format(rf_roc_auc)
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def compare_metrics(
        data_path,
        target_column
):
    pass


def compare_prauc_curve(data_path, target_column):
    dataframe = pd.read_csv(data_path)

    # Fill missing values with avg
    dataframe.fillna(dataframe.mean(), inplace=True)

    # Create Two copies of the dataframe
    dataframe_rfwoc = dataframe.copy()
    dataframe_rf = dataframe.copy()

    rfwoc_train_df, rfwoc_test_df = train_test_split(
        dataframe_rfwoc,
        test_size=0.2,
        random_state=42
    )

    # Create a RandomForest classifier
    rfwoc_model = RandomForestWithCluster()

    # Train the classifier
    rfwoc_model.fit(rfwoc_train_df, target_column)

    # Make predictions probabilities
    rfwoc_y_probs, rfwoc_y_test = rfwoc_model.predict_proba(rfwoc_test_df, target_column)

    # Compute Precision-Recall curve and Precision-Recall AUC
    rfwoc_precision, rfwoc_recall, _ = precision_recall_curve(rfwoc_y_test, rfwoc_y_probs)
    rfwoc_pr_auc = auc(rfwoc_recall, rfwoc_precision)

    x = dataframe_rf.drop(target_column, axis=1)
    y = dataframe_rf[target_column]

    # Split the data into training and testing sets
    rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create a RandomForest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Train the classifier
    rf_classifier.fit(rf_x_train, rf_y_train)

    # Make predictions probabilities
    rf_y_probs = rf_classifier.predict_proba(rf_x_test)[:, 1]

    rf_precision, rf_recall, _ = precision_recall_curve(rf_y_test, rf_y_probs)
    rf_pr_auc = auc(rf_recall, rf_precision)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))

    plt.plot(
        rfwoc_recall,
        rfwoc_precision,
        color='darkorange',
        lw=2,
        label='RFWOC PRAUC (AUC = {:.2f})'.format(rfwoc_pr_auc)
    )
    plt.plot(
        rf_recall,
        rf_precision,
        color='green',
        lw=2,
        label='RF PRAUC (AUC = {:.2f})'.format(rf_pr_auc)
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


