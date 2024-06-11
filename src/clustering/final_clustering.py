import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import _tree
from sklearn.model_selection import cross_val_score

# --------------------------------------------------------------------------------------------------------- #
# Data Cleaning
# --------------------------------------------------------------------------------------------------------- #
class DataCleaner:
    """
    Class to perform data cleaning on the given data.
    """

    def __init__(self, df):
        self.df = df

    def remove_duplicates(self):
        """
        Remove duplicate rows from the DataFrame.
        """
        print(f"Initial shape before removing duplicates: {self.df.shape}")
        self.df = self.df.drop_duplicates()
        print(f"Shape after removing duplicates: {self.df.shape}")

    def drop_invalid_rows(self):
        """
        Drop rows with any 'Sin respuesta' or 'Sin fecha' values.
        """
        print(f"Initial shape before dropping invalid rows: {self.df.shape}")
        self.df = self.df.dropna()
        print(f"Shape after dropping invalid rows: {self.df.shape}")

    '''
    def normalize_numerical_columns(self):
        """
        Normalize numerical columns to have values between 0 and 1.
        """
        print(f"Initial shape before normalizing numerical columns: {self.df.shape}")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numerical_cols] = self.df[numerical_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        print(f"Shape after normalizing numerical columns: {self.df.shape}")
    '''

    def handle_remaining_nans(self):
        """
        Handle remaining NaNs by imputing or dropping.
        """
        print(f"Initial shape before handling remaining NaNs: {self.df.shape}")
        # Check if there are still NaNs
        if self.df.isna().any().any():
            # Impute numerical NaNs with mean and categorical NaNs with mode
            for col in self.df.columns:
                if self.df[col].dtype in [np.float64, np.int64]:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        print(f"Shape after handling remaining NaNs: {self.df.shape}")

    def perform_cleaning(self):
        """
        Perform all cleaning steps.
        """
        self.remove_duplicates()
        self.drop_invalid_rows()
        # self.normalize_numerical_columns()
        self.handle_remaining_nans()
        return self.df


# --------------------------------------------------------------------------------------------------------- #
# Data Preprocessing
# --------------------------------------------------------------------------------------------------------- #
class DataPreprocessor:
    """
    Class to perform data preprocessing on the given data.
    """

    def __init__(self, data_path):
        """
        Initialize the DataPreprocessor class with a given data path.
        :param data_path: str - Path to the data file in CSV format
        """
        self.data_path = data_path  # Path to the data file in CSV format
        self.df = self.read_data()  # DataFrame containing the data
        self.original_df = self.df.copy()  # Copy of the original DataFrame
        self.encoders = {}  # Dictionary to store encoders for categorical columns
        self.mapping_dicts = (
            {}
        )  # Dictionary to store mapping dictionaries for categorical columns
        self.categorical_columns_hot_encoded = (
            []
        )  # List to store categorical columns that were one-hot encoded
        self.categorical_columns = (
            []
        )  # List to store categorical columns that were label encoded
        self.numerical_columns = []  # List to store numerical columns

    def read_data(self):
        """
        Read the data from the given data path and return as a DataFrame.
        :return: DataFrame - Read data as DataFrame
        """
        return pd.read_csv(self.data_path)

    @staticmethod
    def print_statistics(initial_count, final_count):
        """
        Print statistics such as initial and final record count after data encoding.
        :param initial_count: int - Initial count of records before encoding
        :param final_count: int - Final count of records after encoding
        """
        print(f"Initial record count: {initial_count}")
        print(f"Final record count after encoding: {final_count}")
        if initial_count != final_count:
            print(f"Records lost: {initial_count - final_count}")
        print()

    def label_encode_column(self, col):
        """
        Perform label encoding on a given column.
        :param col: str - Name of the column to label encode
        """
        le = LabelEncoder()  # Initialize the LabelEncoder
        self.df[col] = le.fit_transform(self.df[col])  # Fit and transform the column
        self.encoders[col] = le  # Store the encoder
        self.mapping_dicts[col] = dict(
            enumerate(le.classes_)
        )  # Store the mapping dictionary

    def one_hot_encode_column(self, col):
        """
        Perform one-hot encoding on a given column.
        :param col: str - Name of the column to one-hot encode
        """
        self.categorical_columns_hot_encoded.append(col)  # Add the column to the list
        ohe = OneHotEncoder(sparse_output=False)  # Initialize the OneHotEncoder
        encoded_features = ohe.fit_transform(
            self.df[[col]]
        )  # Fit and transform the column
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=ohe.get_feature_names_out([col]),
        )  # Create a DataFrame from the encoded features
        self.df = self.drop_and_concat(
            self.df, encoded_df, col
        )  # Drop and concatenate the new columns
        self.encoders[col] = ohe  # Store the encoder

    @staticmethod
    def drop_and_concat(original_df, new_cols_df, col_to_drop):
        """
        Drop a column from the original DataFrame and concatenate new columns generated from encoding.
        :param original_df: pd.DataFrame - Original DataFrame
        :param new_cols_df: pd.DataFrame - DataFrame containing new columns generated from encoding
        :param col_to_drop: str - Name of the column to drop
        :return: DataFrame - Updated DataFrame after dropping and concatenating
        """
        original_df = original_df.drop([col_to_drop], axis=1)  # Drop the column
        return pd.concat(
            [original_df, new_cols_df], axis=1
        )  # Concatenate the new columns

    def encode_categorical(self):
        """
        Perform encoding on all categorical columns in the DataFrame.
        """
        initial_count = self.df.shape[0]
        # Loop over all columns in the DataFrame
        for col in self.df.columns:
            if self.df[col].dtype == "object":  # Check if the column is of object type
                unique_values = self.df[
                    col
                ].nunique()  # Get the number of unique values in the column

                if (
                    unique_values >= 2
                ):  # If the number of unique values is greater than 3, perform one-hot encoding
                    self.label_encode_column(col)
                else:  # Else, perform label encoding
                    self.one_hot_encode_column(col)
                self.categorical_columns += [col]  # Add the column to the list

        final_count = self.df.shape[0]
        self.print_statistics(initial_count, final_count)

    def prepare_final_df(self):
        """
        Prepare the final DataFrame for machine learning algorithms by dropping unnecessary columns.
        :return: pd.DataFrame - DataFrame ready for machine learning algorithms
        """
        return self.df.drop(
            ["userid"], axis=1
        ).copy()  # Drop the userid column and return the DataFrame


# --------------------------------------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------------------------------------- #
class Clusterer:
    """
    Class to perform clustering on the given data.
    """

    def __init__(self, df):
        """
        Initialize the Clusterer class with a given DataFrame.
        :param df: DataFrame - DataFrame containing features to be clustered
        """
        self.color_palette_dict = (
            None  # Dictionary to store color palettes for each cluster
        )
        self.df = df.drop(
            "userid", axis=1
        )  # DataFrame containing features to be clustered
        self.moodle_ids = df["userid"].values  # List of moodle ids
        self.optimal_clusters = None  # Optimal number of clusters
        self.kmeans = None  # KMeans model
        self.final_df = None  # Final DataFrame containing features and cluster labels

        # Print message to indicate start of clustering
        print("-- Clustering --\n")

    def find_optimal_clusters(self, max_k=10):
        """
        Find the optimal number of clusters using silhouette scores.
        :param max_k: int - Maximum number of clusters to consider
        """
        iters = range(2, max_k + 1, 1)  # Range of clusters to consider
        silhouette_scores = []  # List to store silhouette scores for each cluster
        # Loop over all clusters
        for k in iters:
            kmeans_model = KMeans(
                n_clusters=k, init="k-means++", n_init=10, random_state=0
            ).fit(
                self.df
            )  # Fit the KMeans model to the data
            silhouette_scores.append(
                silhouette_score(self.df, kmeans_model.labels_, metric="euclidean")
            )  # Calculate the silhouette score and append to the list
        n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        if n_clusters == 2:
            n_clusters = 3
        self.optimal_clusters = n_clusters  # Store the optimal number of clusters

    def apply_kmeans_clustering(self):
        """
        Perform KMeans clustering using the optimal number of clusters.
        """
        self.kmeans = KMeans(
            n_clusters=self.optimal_clusters,
            init="k-means++",
            n_init=10,
            random_state=0,
        ).fit(
            self.df
        )  # Fit the KMeans model to the data
        self.final_df = self.df.copy()  # Create a copy of the DataFrame
        self.final_df[
            "cluster"
        ] = self.kmeans.labels_  # Add the cluster labels to the DataFrame
        self.final_df["userid"] = self.moodle_ids  # Add the moodle ids to the DataFrame

    def print_statistics(self, initial_count, final_count):
        """
        Print various statistics related to clustering.
        :param initial_count: int - Initial number of records before clustering
        :param final_count: int - Final number of records after clustering
        """
        print(f"Initial record count: {initial_count}\n")

        print("% of users in each cluster:\n")
        cluster_counts = self.final_df["cluster"].value_counts(normalize=True)
        for i in range(self.optimal_clusters):
            print(f"\tCluster {i}: {round(cluster_counts.get(i, 0) * 100, 2)}%")
        print()

        print("Number of records in each cluster:\n")
        cluster_counts = self.final_df["cluster"].value_counts()
        for i in range(self.optimal_clusters):
            print(f"\tCluster {i}: {cluster_counts.get(i, 0)}")
        print()

        print(f"Final record count after clustering: {final_count}\n")
        if initial_count != final_count:
            print(f"Records lost: {initial_count - final_count}")

    def perform_clustering(self):
        """
        Perform clustering and print various statistics.
        """
        initial_count = self.df.shape[0]
        self.apply_kmeans_clustering()
        final_count = self.final_df.shape[0]
        self.print_statistics(initial_count, final_count)


# --------------------------------------------------------------------------------------------------------- #
# Feature Importance
# --------------------------------------------------------------------------------------------------------- #
class FeatureIdentifier:
    """
    Class to identify the most important features for each cluster.
    """

    def __init__(self, df, optimal_clusters):
        """
        Initialize the FeatureImportanceIdentifier class.
        :param df: DataFrame - DataFrame containing features and cluster labels
        :param optimal_clusters: int - Optimal number of clusters
        """
        self.df = df  # DataFrame containing features and cluster labels
        self.optimal_clusters = optimal_clusters  # Optimal number of clusters
        self.feature_importance_dict = (
            {}
        )  # Dictionary to store feature importances for each cluster
        self.f1_score_dict = {}  # Dictionary to store F1 scores for each cluster
        self.accuracy_dict = {}  # Dictionary to store accuracy scores for each cluster
        self.feature_importance_df = (
            None  # DataFrame to store feature importances for each cluster
        )
        self.final_df = (
            pd.DataFrame()
        )  # Final DataFrame with features and cluster labels

        # Print message to indicate start of feature importance identification
        print("-- Feature importance --\n")

    def prepare_data(self, i):
        """
        Prepare the data for classification.
        :param i: int - Current cluster index
        :return: Tuple - Training and testing sets
        """
        target_col = f"Cluster_{i}"  # Name of the target column
        classification_df = self.df.copy()  # Create a copy of the DataFrame
        classification_df = classification_df.drop("userid", axis=1)  # Drop the userid
        classification_df[target_col] = (classification_df["cluster"] == i).astype(
            int
        )  # Create the target column
        X = classification_df.drop(
            ["cluster", target_col], axis=1
        )  # Drop the cluster and target columns
        y = classification_df[target_col]  # Get the target column
        return train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )  # Split the data into train and test

    @staticmethod
    def train_and_evaluate_model(X_train, y_train, X, y):
        """
        Train a RandomForestClassifier and get metrics.
        :param X_train: DataFrame - Training feature set
        :param y_train: Series - Training target set
        :param X: DataFrame - Full feature set
        :param y: Series - Full target set
        :return: Tuple - F1 score and feature importances
        """
        model = RandomForestClassifier(
            random_state=0
        )  # Initialize the RandomForestClassifier
        model.fit(X_train, y_train)  # Fit the model to the training data
        cv_f1_scores = cross_val_score(
            model, X, y, cv=10, scoring="f1_macro"
        )  # Get the F1 scores
        cv_accuracy_scores = cross_val_score(
            model, X, y, cv=10, scoring="accuracy"
        )  # Get the accuracy scores
        return (
            cv_f1_scores.mean(),
            cv_accuracy_scores.mean(),
            model.feature_importances_,
        )  # Return the F1 score and feature importances

    def identify_importance(self):
        """
        Identify feature importances across all clusters.
        """
        initial_count = self.df.shape[0]
        print(f"Initial record count: {initial_count}\n")

        # Loop over all clusters
        for i in range(self.optimal_clusters):
            X_train, X_test, y_train, y_test = self.prepare_data(
                i
            )  # Prepare the data for classification
            f1_score, accuracy, feature_importances = self.train_and_evaluate_model(
                X_train,
                y_train,
                pd.concat([X_train, X_test], ignore_index=True),
                pd.concat([y_train, y_test], ignore_index=True),
            )  # Train and evaluate the model
            feature_names_list = (
                X_train.columns.tolist()
            )  # Get the list of feature names

            self.f1_score_dict[f"cluster_{i}"] = f1_score  # Store the F1 score
            self.accuracy_dict[f"cluster_{i}"] = accuracy  # Store the accuracy
            self.feature_importance_dict[f"cluster_{i}"] = dict(
                zip(feature_names_list, feature_importances)
            )  # Store the feature importances

    def filter_dataset(self, threshold=0.05):
        """
        Filter the dataset based on the given threshold.
        :param threshold: float - Threshold value
        :return: None
        """
        self.final_df = self.df.copy()
        features_to_keep_set = set()

        for i in range(self.optimal_clusters):
            feature_importances = self.feature_importance_dict.get(f"cluster_{i}", {})
            cluster_features = {
                feature
                for feature, importance in feature_importances.items()
                if importance > threshold
            }
            features_to_keep_set.update(cluster_features)

        features_to_keep_list = list(features_to_keep_set)
        self.final_df = self.final_df[["cluster", "userid"] + features_to_keep_list]

    def display_results(self):
        """
        Display the computed results.
        """
        self.display_f1_scores()
        self.display_accuracy_scores()
        self.display_feature_importance()

    def display_f1_scores(self):
        """
        Display the F1 score for each cluster.
        """
        print("F1 score for each cluster, sorted by score:\n")
        # Sort the F1 scores in descending order
        for cluster, f1 in sorted(
            self.f1_score_dict.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"\t\t{cluster}: {round(f1, 4)}")

    def display_accuracy_scores(self):
        """
        Display the accuracy score for each cluster.
        """
        print("\nAccuracy score for each cluster, sorted by score:\n")
        # Sort the accuracy scores in descending order
        for cluster, accuracy in sorted(
            self.accuracy_dict.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"\t\t{cluster}: {round(accuracy, 4)}")

    def display_feature_importance(self):
        """
        Display the feature importances for each cluster.
        """
        print("\nFeature Importance for each cluster, sorted by importance:\n")
        # Sort the feature importances in descending order
        for cluster, importance in self.feature_importance_dict.items():
            print(f"\t{cluster}:")
            sorted_importance = {
                k: v
                for k, v in sorted(
                    importance.items(), key=lambda item: item[1], reverse=True
                )
            }
            for feature, imp in sorted_importance.items():
                print(f"\t\t{feature}: {round(imp, 4)}")
            print()

    def save_feature_importance_to_csv(self, data_preprocessor, save_path):
        """
        Save the feature importances to a CSV file.
        """
        feature_importance_df = pd.DataFrame()
        # Loop over all clusters
        for cluster, importance in self.feature_importance_dict.items():
            temp_df = pd.DataFrame(
                list(importance.items()), columns=["Feature", "Importance"]
            )  # Create a DataFrame
            temp_df["Cluster"] = cluster  # Add the cluster column
            feature_importance_df = pd.concat(
                [feature_importance_df, temp_df], ignore_index=True
            )  # Concatenate

        # Mean of hot encoded categorical variables
        feature_importance_df_ = feature_importance_df.copy()
        feature_importance_df_["Cluster"] = feature_importance_df_["Cluster"].apply(
            lambda x: int(x.split("_")[1])
        )  # Cluster column to integer

        # Calculate the mean of the feature importances for hot encoded categorical variables
        # First, create a new column in the DataFrame to store the top-level feature names.
        feature_importance_df_["Top_Level_Feature"] = (
            feature_importance_df_["Feature"].str.split("_").str[0]
        )

        # Group by 'Cluster' and 'Top_Level_Feature' and then calculate the mean importance.
        mean_feature_importance_df = (
            feature_importance_df_.groupby(["Cluster", "Top_Level_Feature"])[
                "Importance"
            ]
            .mean()
            .reset_index()
        )

        # If you want to rename 'Top_Level_Feature' back to 'Feature', you can do it like this:
        mean_feature_importance_df.rename(
            columns={"Top_Level_Feature": "Feature"}, inplace=True
        )

        # Filter mean feature importances for hot encoded categorical variables
        mean_feature_importance_df = mean_feature_importance_df[
            mean_feature_importance_df["Feature"].isin(
                data_preprocessor.categorical_columns_hot_encoded
            )
        ]

        # Add the mean feature importances to the DataFrame
        feature_importance_df_ = pd.concat(
            [feature_importance_df_, mean_feature_importance_df], ignore_index=True
        )
        # Sort the DataFrame by cluster and importance
        feature_importance_df_ = feature_importance_df_.sort_values(
            ["Cluster", "Importance"], ascending=[True, False]
        )
        # Cluster as integer
        feature_importance_df_["Cluster"] = feature_importance_df_["Cluster"].astype(
            int
        )
        feature_importance_df_.drop("Top_Level_Feature", axis=1, inplace=True)
        feature_importance_df_.to_csv(save_path, index=False)  # Save to a CSV file
        self.feature_importance_df = feature_importance_df


# --------------------------------------------------------------------------------------------------------- #
# Decision Tree
# --------------------------------------------------------------------------------------------------------- #
class DecisionTree:
    """
    Class to perform decision tree classification on the clusters and extract rules.
    """

    def __init__(self, df, mapping_dicts):
        """
        Inicializa la clase DecisionTreeRefactored.
        :param df: DataFrame - DataFrame original
        :param mapping_dicts: dict - Diccionario de mapeo para variables categóricas
        """
        self.cluster_all_rules = {}  # Dictionary to store all rules for each cluster
        self.all_rules = {}  # Dictionary to store all rules
        self.final_df = (
            None  # Final DataFrame containing features, cluster labels, and rules
        )
        self.cluster_rules_map = {}  # Dictionary to map clusters to rules
        self.leaf_cluster_map = {}  # Dictionary to map leaf nodes to clusters
        self.model = None  # DecisionTreeClassifier model
        self.df = df.drop(
            "userid", axis=1
        )  # DataFrame containing features, cluster labels, and rules
        self.mapping_dicts = mapping_dicts  # Dictionary containing mapping dictionaries for categorical columns

        # Print message to indicate start of decision tree classification
        print("-- Decision Tree Classification --\n")

        # Print columns to use in the decision tree
        print("Columns to use in the decision tree:")
        print(self.df.columns, "\n")

    def prepare_data(self):
        """
        Prepare the data for training the model.
        :return: Tuple - Training and testing sets
        """
        X = self.df.drop(["cluster"], axis=1)  # Drop the cluster column
        y = self.df["cluster"]  # Get the cluster column
        return train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )  # Split the data into train and test

    def train_model(self):
        """
        Train a DecisionTreeClassifier model.
        """
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.prepare_data()  # Prepare the data for training
        self.model = DecisionTreeClassifier(
            random_state=0
        )  # Initialize the DecisionTreeClassifier
        self.model.fit(X_train, y_train)  # Fit the model to the training data
        self.calculate_f1_score(
            X_train, y_train
        )  # Calculate the F1 score for the training data

    def calculate_f1_score(self, X, y):
        """
        Calculate the F1 score for the model.
        :param X: pd.DataFrame - Feature set
        :param y: pd.Series - Target set
        """
        cv_f1_scores = cross_val_score(
            self.model, X, y, cv=10, scoring="f1_macro"
        )  # Get the F1 scores
        cv_accuracy_scores = cross_val_score(
            self.model, X, y, cv=10, scoring="accuracy"
        )  # Get the accuracy scores
        avg_f1_score = cv_f1_scores.mean()  # Calculate the average F1 score
        avg_accuracy_score = (
            cv_accuracy_scores.mean()
        )  # Calculate the average accuracy score
        print(f"Accuracy: {round(avg_accuracy_score, 4)}")
        print(f"F1 score: {round(avg_f1_score, 4)}")

    def plot_tree(self, save_path):
        """
        Plot the decision tree and save it to a file.
        :param save_path: str - Path to save the decision tree
        :return: None
        """
        fig, ax = plt.subplots(figsize=(40, 20))  # Initialize the figure and axes
        plot_tree(
            self.model,
            filled=True,
            feature_names=self.df.drop("cluster", axis=1).columns.tolist(),
            class_names=[str(i) for i in sorted(self.model.classes_)],
            rounded=True,
            proportion=True,
            fontsize=10,
            ax=ax,
        )  # Plot the decision tree

        ax.text(
            0,
            -0.1,
            "Disclaimer: The color coding used to represent classes in this decision tree "
            "visualization is independent of the color scheme applied in the PCA scatter plot.\n\n"
            "It is imperative to note that the numerical identifiers for leaf nodes in this "
            "decision tree directly map to the cluster indices, but the associated colors do "
            "not maintain consistency across different visual representations.",
            style="italic",
            fontsize=10,
            transform=ax.transAxes,
        )  # Note for the reader

        # Title, label, and save
        ax.set_title("Decision Tree for Clustering", fontsize=16)
        ax.set_xlabel("Features", fontsize=14)
        ax.set_ylabel("Clusters", fontsize=14)
        plt.savefig(save_path, format="svg")

    def recurse_extract_rules(self, node, rules):
        """
        Recursively extract the rules from the decision tree.
        :param node: int - Current node
        :param rules: list - List of rules
        """
        tree_ = self.model.tree_  # Get the decision tree
        feature_names = self.df.drop(
            "cluster", axis=1
        ).columns.tolist()  # Get the feature names

        # Check if the current node is a leaf node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]  # Get the feature name
            threshold = tree_.threshold[node]  # Get the threshold value
            self.recurse_extract_rules(
                tree_.children_left[node], rules + [f"{name} <= {threshold}"]
            )  # Recurse
            self.recurse_extract_rules(
                tree_.children_right[node], rules + [f"{name} > {threshold}"]
            )  # Recurse
        # If the current node is a leaf node, extract the rules
        else:
            leaf_cluster = tree_.value[node].argmax()  # Get the leaf cluster
            self.leaf_cluster_map[
                node
            ] = leaf_cluster  # Map the leaf node to the leaf cluster
            self.all_rules[node] = rules  # Store the rules
            self.cluster_all_rules.setdefault(leaf_cluster, []).append(
                rules
            )  # Store the rules for the leaf cluster

    def extract_rules(self):
        """
        Extract the rules from the decision tree.
        """
        self.recurse_extract_rules(0, [])  # Recursively extract the rules

    def print_rules(self):
        """
        Print the rules for each cluster.
        """
        print("\nRules for each cluster:\n")
        # Loop over all clusters
        for cluster, rules in self.cluster_all_rules.items():
            print(f"\tCluster {cluster}:")
            for rule in rules:
                print(f"\t\t{rule}")  # Print the rules
            print()

    def apply_rules_and_get_values(self, df, rules, target_column):
        """
        Apply the rules to the DataFrame and get the unique values for the target column.
        :param df: pd.DataFrame - DataFrame to apply the rules to
        :param rules: list - List of rules
        :param target_column: str - Name of the target column
        :return: list - List of unique values for the target column
        """
        temp_df = df.copy()
        # Loop over all rules
        for rule in rules:
            if isinstance(rule, str):  # Check if 'rule' is a string
                feature, operator, value = rule.split(" ")  # Split the rule
                if operator == "<=":  # Check the operator
                    temp_df = temp_df[
                        temp_df[feature] <= float(value)
                    ]  # Apply the rule
                else:
                    temp_df = temp_df[temp_df[feature] > float(value)]  # Apply the rule
        return [
            self.mapping_dicts.get(target_column, {}).get(
                code, code
            )  # Get the mapped value
            for code in temp_df[target_column].unique()  # Get the unique values
        ]

    @staticmethod
    def evaluate_rule(rules, row):
        """
        Evaluate if a given set of rules applies to a specific row.
        :param rules: list - List of rules
        :param row: pd.Series - Single row from DataFrame
        :return: bool - True if rules apply, False otherwise
        """
        for rule in rules:
            feature, operator, value = rule.split(" ")
            if operator == "<=":
                if row[feature] > float(value):
                    return False
            else:
                if row[feature] <= float(value):
                    return False
        return True

    def add_rules_to_df(self, clusterer):
        """
        Add the rule indices to the DataFrame.
        :param clusterer: Clusterer - Clusterer object
        :return: None
        """
        self.df["rule_index"] = None  # Initialize the 'rule_index' column

        # Apply the rules to each row of the DataFrame
        def apply_rules(row):
            cluster = row["cluster"]  # Get the cluster
            # Get the rules for this cluster
            rules_list = self.cluster_all_rules.get(cluster, [])
            if rules_list:  # Check if rules exist
                for idx, rules in enumerate(rules_list):
                    if self.evaluate_rule(rules, row):  # Call to evaluate_rule
                        row["rule_index"] = "Rule " + str(idx)  # Add the rule index
                        break  # Exit the loop once a matching rule is found
            return row

        self.final_df = clusterer.final_df.apply(apply_rules, axis=1)

    def save_rules_to_csv(self, save_path):
        """
        Save the rules and corresponding clusters to a CSV file.
        :param save_path: str - Path to save the CSV file
        :return: None
        """
        # Initialize an empty list to store the rules and clusters
        rule_cluster_data = []

        # Iterate through the dictionary of cluster rules
        for cluster, rules_list in self.cluster_all_rules.items():
            for idx, rules in enumerate(rules_list):
                # Convert the list of rules to a single string for easier storage
                rule_str = ", ".join(rules)

                # Append the rule and its corresponding cluster to the list
                rule_cluster_data.append(
                    {"Cluster": cluster, "Rule": "Rule " + str(idx), "Rules": rule_str}
                )

        # Create a DataFrame from the list of rules and clusters
        rule_cluster_df = pd.DataFrame(rule_cluster_data)

        # Save the DataFrame to a CSV file
        rule_cluster_df.to_csv(save_path, index=False)

    def save_cluster_variables(self, data_preprocessor, feature_identifier, save_path):
        """
        Save the cluster variables to a JSON file.
        :param data_preprocessor: DataPreprocessor - DataPreprocessor object
        :param feature_identifier: FeatureIdentifier - FeatureIdentifier object
        :param save_path: str - Path to save the JSON file
        :return: None
        """
        # Initialize a dictionary to store data for each cluster
        cluster_data = {
            "cluster": [],  # Cluster index
            "n_users": [],  # Number of users in the cluster
            "feature_importance": [],  # Feature importance
            "rules": [],  # Rules
            "categorical_variables": [],  # Categorical variables
            "numerical_variables": [],  # Numerical variables
        }

        numerical_variables_info = {
            "cluster": [],
            "feature": [],
            "mean": [],
            "std": [],
        }
        # Loop over all clusters
        for cluster, rules in self.cluster_all_rules.items():
            # Initialize a dictionary to store the row data
            row_data = {
                "cluster": cluster,
                "n_users": self.df[self.df["cluster"] == cluster].shape[
                    0
                ],  # Get the number of users in the cluster
                "feature_importance": {},
                "rules": {},
                "categorical_variables": {},
                "numerical_variables": {},
            }

            # Add the rules to the row data
            for i, rule in enumerate(rules):
                row_data["rules"][f"rule_{i}"] = rule  # Add the rule

            # Loop over all categorical variables
            for cat_var, mapping_dict in self.mapping_dicts.items():
                unique_values = [
                    self.mapping_dicts.get(cat_var, {}).get(
                        code, code
                    )  # Get the mapped value
                    for code in self.apply_rules_and_get_values(
                        self.df, rules, cat_var
                    )  # Get the unique values
                ]
                row_data["categorical_variables"][cat_var] = unique_values

            # Loop over categorical variables that were one-hot encoded
            for cat_var in data_preprocessor.categorical_columns_hot_encoded:
                # Check if any columns match the regex pattern
                matching_columns = [
                    col
                    for col in self.df.columns
                    if re.match(
                        f"{re.escape(cat_var)}_.+",
                        col,  # Check if the column matches the regex pattern
                    )
                    and (
                        self.df[self.df["cluster"] == cluster][col]
                        != 0  # Check if the column is non-zero
                    ).any()
                ]

                if matching_columns:
                    unique_values = [
                        col.split("_")[1]
                        for col in matching_columns
                        if col.startswith(cat_var)
                    ]  # Get the unique values
                    row_data["categorical_variables"][
                        cat_var
                    ] = unique_values  # Add the unique values

            # Calculate the hot encoded categorical variables for this cluster
            matching_columns = pd.Index([])  # Initialize an empty Index
            for cat_var in data_preprocessor.categorical_columns_hot_encoded:
                new_columns = pd.Index(
                    [
                        col
                        for col in self.df.columns
                        if re.match(f"{re.escape(cat_var)}_.+", col)
                    ]
                )  # Get the columns that match the regex pattern
                matching_columns = matching_columns.append(
                    new_columns
                )  # Append the columns to the Index

            # Loop over all numerical variables
            non_cat_vars = (
                set(self.df.columns)
                - set(self.mapping_dicts.keys())
                - set(matching_columns)
                - {"cluster", "rule_index"}
            )  # Get the numerical variables
            for non_cat_var in non_cat_vars:
                # Calculate descriptive statistics for this non-categorical variable
                mean_value = np.mean(
                    self.df[self.df["cluster"] == cluster][non_cat_var]
                )
                std_value = np.std(self.df[self.df["cluster"] == cluster][non_cat_var])
                row_data["numerical_variables"][f"{non_cat_var}_mean"] = mean_value
                row_data["numerical_variables"][f"{non_cat_var}_std"] = std_value
                numerical_variables_info["cluster"].append(cluster)
                numerical_variables_info["feature"].append(f"{non_cat_var}_mean")
                numerical_variables_info["mean"].append(mean_value)
                numerical_variables_info["std"].append(std_value)

            # Save the numerical variables to a CSV file
            numerical_variables_df = pd.DataFrame(numerical_variables_info)

            # Add the feature importance to the row data
            for feature, importance in sorted(
                feature_identifier.feature_importance_dict[
                    f"cluster_{cluster}"
                ].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                row_data["feature_importance"][feature] = importance

            # Append the row data to the cluster_data dictionary
            for key, value in row_data.items():  # Loop over all keys in the row data
                cluster_data[key].append(
                    value
                )  # Append the value to the cluster_data dictionary

        # Convert cluster_data dictionary to a DataFrame
        cluster_variables_df = pd.DataFrame(cluster_data)

        # Make sure the 'cluster' column is integer type
        cluster_variables_df["cluster"] = cluster_variables_df["cluster"].astype(int)

        # Save DataFrame to JSON, tabulating it in a more organized way
        cluster_variables_df.to_json(save_path, orient="records", indent=4)


# --------------------------------------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------------------------------------- #
class Visualizer:
    """
    Class to visualize the clusters using PCA.
    """

    def __init__(self, df, cluster_column="cluster"):
        if "userid" in df.columns:
            self.df = df.drop("userid", axis=1)
            self.moodle_ids = df["userid"].values
        else:
            self.df = df
            self.moodle_ids = None
        self.cluster_column = cluster_column

    def plot_pca(self, save_path):
        pca = PCA(n_components=2)  # Initialize the PCA model
        principal_components = pca.fit_transform(
            self.df.drop(self.cluster_column, axis=1)
        )  # Fit the PCA model to the data and transform the data
        principal_df = pd.DataFrame(
            data=principal_components,
            columns=["Principal Component 1", "Principal Component 2"],
        )  # Create a DataFrame from the principal components
        principal_df[self.cluster_column] = self.df[
            self.cluster_column
        ]  # Add the cluster column

        # Plot the PCA
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x="Principal Component 1",
            y="Principal Component 2",
            hue=self.cluster_column,
            data=principal_df,
            palette=sns.color_palette(
                "hls", n_colors=len(principal_df[self.cluster_column].unique())
            ),
            alpha=0.6,
            s=40,
        )

        # Title, label, and save
        plt.title("2-Component PCA for Clustering", fontsize=16, pad=20)
        plt.xlabel("Principal Component 1", fontsize=14, labelpad=10)
        plt.ylabel("Principal Component 2", fontsize=14, labelpad=10)
        plt.savefig(save_path, format="svg")

        # Principal dataframe with userid, cluster, and principal components
        principal_df["userid"] = self.moodle_ids
        principal_df = principal_df[
            ["userid", "cluster", "Principal Component 1", "Principal Component 2"]
        ]

        # Save PCA data to CSV
        principal_df.to_csv(Path.cwd() / "data/clustering/pca.csv", index=False)


# --------------------------------------------------------------------------------------------------------- #
# Save Data
# --------------------------------------------------------------------------------------------------------- #
class SaveData:
    """
    Class to save the data after clustering.
    """

    def __init__(self, df):
        """
        Initialize the SaveData class with a given DataFrame.
        :param df: pd.DataFrame - DataFrame containing features and cluster labels
        """
        self.df = df

    def merge_original_data(self, data_preprocessor, decision_tree):
        """
        Merge the original data with the cluster labels.
        :param data_preprocessor: DataPreprocessor - DataPreprocessor object
        :param decision_tree: DecisionTree - DecisionTree object
        :return: None
        """
        # Merge the original data with the cluster labels
        self.df = pd.merge(
            data_preprocessor.original_df,
            decision_tree.final_df[["userid", "cluster", "rules"]],
            on="userid",
            how="right",
        )

    def save_data(self, save_path):
        """
        Save the DataFrame to a CSV file.
        :param save_path: str - Path to save the CSV file
        :return:
        """
        # Save the DataFrame to a CSV file
        self.df.to_csv(save_path, index=False)


# --------------------------------------------------------------------------------------------------------- #
# Data Handler
# --------------------------------------------------------------------------------------------------------- #
class DataHandler:
    """
    Class to handle the data.
    """

    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def merge_dataframes(original_df, final_df, on="userid", how="right"):
        return pd.merge(
            original_df, final_df[["userid", "cluster", "rule_index"]], on=on, how=how
        )

    @staticmethod
    def save_to_csv(df, save_path):
        df.to_csv(save_path, index=False)

    def run_data_preprocessing(self):
        data_preprocessor = DataPreprocessor(self.data_path)
        data_preprocessor.encode_categorical()
        return data_preprocessor

    def run_data_cleaning(self, df):
        data_cleaner = DataCleaner(df)
        cleaned_df = data_cleaner.perform_cleaning()
        if cleaned_df.shape[0] == 0:
            raise ValueError("DataFrame está vacío después de la limpieza.")
        return cleaned_df

    @staticmethod
    def run_clustering(final_df_with_ids):
        clusterer = Clusterer(final_df_with_ids)
        clusterer.find_optimal_clusters()
        clusterer.perform_clustering()
        return clusterer

    @staticmethod
    def run_feature_identification(clusterer, data_preprocessor):
        feature_identifier = FeatureIdentifier(
            clusterer.final_df, clusterer.optimal_clusters
        )
        feature_identifier.identify_importance()
        feature_identifier.filter_dataset(threshold=0.0001)
        feature_identifier.display_results()
        feature_identifier.save_feature_importance_to_csv(
            data_preprocessor,
            Path.cwd() / "data/clustering/feature_importance.csv",
        )
        return feature_identifier

    @staticmethod
    def run_decision_tree(
        final_df,
        mapping_dicts,
        data_preprocessor,
        feature_identifier,
        clusterer,
        clustering_data_path=Path.cwd() / "data/clustering",
    ):
        decision_tree = DecisionTree(final_df, mapping_dicts)
        decision_tree.train_model()
        decision_tree.extract_rules()
        decision_tree.print_rules()
        decision_tree.add_rules_to_df(clusterer)
        decision_tree.save_cluster_variables(
            data_preprocessor,
            feature_identifier,
            clustering_data_path / "cluster_variables.json",
        )
        decision_tree.save_rules_to_csv(clustering_data_path / "rules.csv")
        decision_tree.plot_tree(clustering_data_path / "images/decision_tree.svg")
        return decision_tree

    @staticmethod
    def run_visualization(
        final_df, clustering_data_path=Path.cwd() / "data/clustering"
    ):
        visualizer = Visualizer(final_df)
        visualizer.plot_pca(clustering_data_path / "images/pca.svg")
        return visualizer


# --------------------------------------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------------------------------------- #
def main():
    """
    Main function to perform clustering on the given data.
    :return: None
    """
    # Print start message
    print("\n-- Starting --\n")

    # Set the data path
    base_data_path = Path.cwd() / "data"
    moodle_data_path = base_data_path / "moodle_data"
    working_data_path = base_data_path / "working_data"
    clustering_data_path = base_data_path / "clustering"

    # Run the pipeline
    handler = DataHandler(clustering_data_path / "clustering_dataset.csv")
    data_preprocessor = handler.run_data_preprocessing()

    # Adding a checkpoint to print the DataFrame before cleaning
    print(f"DataFrame before cleaning: {data_preprocessor.df.shape}")
    print(data_preprocessor.df.head())

    cleaned_df = handler.run_data_cleaning(data_preprocessor.df)

    # Verificación del DataFrame limpiado
    print(f"Shape of cleaned_df: {cleaned_df.shape}")
    print(f"Cleaned DataFrame Head:\n{cleaned_df.head()}")
    
    # Check for NaNs
    if cleaned_df.isna().any().any():
        print("DataFrame contains NaNs after cleaning!")
        print(cleaned_df[cleaned_df.isna().any(axis=1)])
        raise ValueError("No data available for clustering after cleaning.")

    clusterer = handler.run_clustering(cleaned_df)
    feature_identifier = handler.run_feature_identification(
        clusterer, data_preprocessor
    )
    decision_tree = handler.run_decision_tree(
        feature_identifier.final_df,
        data_preprocessor.mapping_dicts,
        data_preprocessor,
        feature_identifier,
        clusterer,
    )
    handler.run_visualization(clusterer.final_df)

    # Merge original data with clustering results
    merged_df = handler.merge_dataframes(
        data_preprocessor.original_df, decision_tree.final_df
    )
    merged_df["cluster"] = merged_df["cluster"].astype(int)

    # Read CSV users.csv
    users = pd.read_csv(working_data_path / "cleaned_data_users.csv")
    users = users[["id", "city", "country"]]
    users.rename(columns={"id": "userid"}, inplace=True)
    merged_df = pd.merge(merged_df, users, on="userid", how="left")
    handler.save_to_csv(merged_df, clustering_data_path / "dataset_with_results2.csv")

    # Print end message
    print("-- Finished --\n")

# --------------------------------------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------- #
# End of file
