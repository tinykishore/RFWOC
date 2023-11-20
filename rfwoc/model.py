import copy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RandomForestWithCluster:
    def __init__(self,
                 plot=False,
                 cluster_override=0,
                 rf_split_random_state=42,
                 rf_random_state=50,
                 rf_n_estimators=100,
                 verbose=False):
        self.__debug_message = verbose
        self.__plot = plot
        self.__cluster_override = cluster_override
        self.__rf_split_random_state = rf_split_random_state
        self.__rf_random_state = rf_random_state
        self.__rf_n_estimators = rf_n_estimators

        # Clustering Model
        self.__kmeans_cluster_model = None
        # Random Forest Classifier Models
        self.__rfc_models = []
        self.__debug("\033[92m" + "DEBUG: RandomForestWithCluster object created." + "\033[0m")

    def fit(self, train_dataframe, target_column, accuracy_threshold=0.70):
        # For train dataset find optimal k value using elbow method
        optimal_clusters = self.__find_optimal_k(train_dataframe, target_column)

        # User defined optimal k value by looking at the elbow __plot
        if self.__cluster_override > 0:
            optimal_clusters = self.__cluster_override

        # Perform K-means clustering on the training set
        clustered_train_dataset, self.__kmeans_cluster_model = self.__perform_kmeans_clustering(
            train_dataframe,
            optimal_clusters
        )

        # Divide the clustered training set into clusters
        clusters = self.__divide_clusters(clustered_train_dataset, optimal_clusters)

        # For each cluster, train a Random Forest Classifier
        for cluster in clusters:
            work_cluster = cluster.copy()
            work_cluster.drop('Cluster', axis=1, inplace=True)
            while True:
                # randomize the cluster
                work_cluster = work_cluster.sample(frac=1).reset_index(drop=True)

                x_train, x_test, y_train, y_test = train_test_split(
                    work_cluster.drop(target_column, axis=1),
                    work_cluster[target_column],
                    test_size=0.2,
                    random_state=self.__rf_split_random_state
                )

                rfc = RandomForestClassifier(
                    n_estimators=self.__rf_n_estimators,
                    random_state=self.__rf_random_state
                )

                rfc.fit(x_train, y_train)
                y_pred = rfc.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)

                if accuracy > accuracy_threshold:
                    self.__rfc_models.append(rfc)
                    break
                else:
                    continue

    def predict(self, test_dataframe, target_column):
        if self.__kmeans_cluster_model is None or len(self.__rfc_models) == 0:
            print("ERROR: You must call fit() before calling predict().")
            return

        # Perform K-means clustering on the test set using the same model as the training set
        clustered_test_dataset = self.__kmeans_cluster_model.predict(test_dataframe)
        optimal_clusters = len(self.__rfc_models)

        test_dataset = copy.deepcopy(test_dataframe)

        # add the cluster column to the test data
        test_dataset['Cluster'] = clustered_test_dataset

        test_clusters = []
        for i in range(optimal_clusters):
            test_clusters.append(test_dataset.loc[test_dataset['Cluster'] == i])

        predictions = []
        for i in range(optimal_clusters):
            cluster = test_clusters[i]
            cluster = cluster.drop('Cluster', axis=1)
            cluster = cluster.drop(target_column, axis=1)
            predictions.append(self.__rfc_models[i].predict(cluster))

        avg_accuracy = 0
        # calculate accuracy for each cluster
        for i in range(optimal_clusters):
            y_test = test_clusters[i][target_column]
            y_pred = predictions[i]
            accuracy = accuracy_score(y_test, y_pred)
            avg_accuracy += accuracy
            print("Accuracy: " + str(accuracy))

        avg_accuracy = avg_accuracy / optimal_clusters

        print("Average Accuracy: " + str(avg_accuracy))

        return np.concatenate(predictions)

    def predict_proba(self, test_dataframe, target_column):
        if self.__kmeans_cluster_model is None or len(self.__rfc_models) == 0:
            print("ERROR: You must call fit() before calling predict_proba().")
            return

        # Perform K-means clustering on the test set using the same model as the training set
        clustered_test_dataset = self.__kmeans_cluster_model.predict(test_dataframe)
        optimal_clusters = len(self.__rfc_models)

        test_dataset = copy.deepcopy(test_dataframe)

        # Add the cluster column to the test data
        test_dataset['Cluster'] = clustered_test_dataset

        test_clusters = []
        for i in range(optimal_clusters):
            test_clusters.append(test_dataset.loc[test_dataset['Cluster'] == i])

        predictions_proba = []
        for i in range(optimal_clusters):
            cluster = test_clusters[i]
            cluster = cluster.drop('Cluster', axis=1)
            cluster = cluster.drop(target_column, axis=1)
            predictions_proba.append(self.__rfc_models[i].predict_proba(cluster)[:, 1])

        predictions_proba = np.concatenate(predictions_proba)

        y_true = []
        for i in range(len(test_clusters)):
            y_true.append(test_clusters[i][target_column])

        y_true = np.concatenate(y_true)

        return predictions_proba, y_true

    def __divide_clusters(self, clustered_dataframe, optimal_clusters):
        clusters = []
        # Iterate through clusters
        for i in range(optimal_clusters):
            clusters.append(clustered_dataframe.loc[clustered_dataframe['Cluster'] == i])

        total_instances = len(clustered_dataframe)
        total_observed_instances = 0
        # Check how many instances are in each cluster
        for i in range(optimal_clusters):
            total_observed_instances += len(clusters[i])
            self.__debug(
                "\033[92m" + "DEBUG: Number of instances in cluster " + str(i) + ": " + str(
                    len(clusters[i])) + "\033[0m")

        if total_instances == total_observed_instances:
            self.__debug("\033[92m" + "DEBUG: Total number of instances in all clusters: " + str(
                total_observed_instances) + "\033[0m")

        return clusters

    def __find_optimal_k(self, dataframe, target_column):
        self.__debug(
            "\033[92m"
            + "DEBUG: Finding the optimal number of clusters (k) based on the elbow point [find_optimal_k()]..."
            + "\033[0m")

        # Assume train_dataset is your training data
        train_dataset = copy.deepcopy(dataframe)
        train_dataset.drop(target_column, axis=1, inplace=True)

        # Create a list to store WCSS values
        wcss_values = []

        self.__debug("\033[92m" + "DEBUG: Defined clustering K range -> [2, 11]" + "\033[0m")
        # Define a range for k
        k_range = range(2, 11)

        self.__debug("\033[92m" + "DEBUG: Starting K-means clustering..." + "\033[0m")
        # Create a for-loop
        for k in k_range:
            # Instantiate a k-means clustering object
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

            # Fit the model to the training set
            kmeans.fit(train_dataset)

            # Append the WCSS value to the list
            wcss_values.append(kmeans.inertia_)

        # Plot the WCSS values if __plot is True
        if self.__plot:
            self.__debug("\033[92m" + "DEBUG: Plotting the WCSS curve for different values of k..." + "\033[0m")
            plt.plot(k_range, wcss_values, marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("WCSS (Within-Cluster-Sum-of-Squares)")
            plt.title("WCSS Curve for Different Values of k")
            plt.show()

        self.__debug(
            "\033[92m" + "DEBUG: Finding the optimal number of clusters (k) based on the elbow point..." + "\033[0m")
        # Find the optimal number of clusters (k) based on the elbow point
        optimal_k = 1  # Default to 1 cluster if no clear elbow is observed
        if len(wcss_values) > 1:
            # Calculate the second derivative to find the elbow
            second_derivative = [wcss_values[i - 2] - 2 * wcss_values[i - 1] + wcss_values[i] for i in
                                 range(2, len(wcss_values))]
            optimal_k_index = second_derivative.index(max(second_derivative)) + 1  # Add 1 for the 0-based index
            optimal_k = k_range[optimal_k_index]

        self.__debug(
            "\033[92m" + "DEBUG: Found the optimal number of clusters (k) based on the elbow point." + "\033[0m")
        self.__debug("\033[92m" + "DEBUG: The optimal number of clusters (k) is: " + str(optimal_k) + "\033[0m")

        return optimal_k

    def __perform_kmeans_clustering(self, dataframe, optimal_clusters):
        self.__debug("\033[92m" + "DEBUG: Performing K-means clustering on the training set..." + "\033[0m")
        self.__debug("\033[92m" + "DEBUG: The optimal number of clusters (k) is: " + str(optimal_clusters) + "\033[0m")
        # Instantiate the KMeans clustering object with the optimal number of clusters
        kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)

        self.__debug("\033[92m" + "DEBUG: Fitting the model to the entire dataset..." + "\033[0m")
        # Fit the model to the entire dataset
        kmeans_model.fit(dataframe)

        self.__debug("\033[92m" + "DEBUG: Getting the cluster labels assigned to each data point..." + "\033[0m")
        # Get the cluster labels assigned to each data point
        cluster_labels = kmeans_model.labels_

        self.__debug("\033[92m" + "DEBUG: The cluster labels are: " + str(cluster_labels) + "\033[0m")
        self.__debug("\033[92m" + "DEBUG: Adding the cluster labels to the DataFrame..." + "\033[0m")
        # Add the cluster labels to the DataFrame
        dataframe_with_clusters = dataframe.copy()  # Create a copy to avoid modifying the original DataFrame

        dataframe_with_clusters['Cluster'] = cluster_labels

        self.__debug("\033[92m" + "DEBUG: K-means clustering on the training set is done." + "\033[0m")
        self.__debug("\033[92m" + "DEBUG: Returning the DataFrame and model..." + "\033[0m")

        return dataframe_with_clusters, kmeans_model

    def __debug(self, message):
        if self.__debug_message:
            print(message)
