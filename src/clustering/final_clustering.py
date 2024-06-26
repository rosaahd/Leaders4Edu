import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
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
                    unique_values >= 3
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
        self.gmm = None  # Gaussian Mixture Model
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
            gmm_model = GaussianMixture(
                n_components=k, covariance_type='full', random_state=0
            ).fit(
                self.df
            )  # Fit the GMM model to the data
            cluster_labels = gmm_model.predict(self.df)
            silhouette_scores.append(
                silhouette_score(self.df, cluster_labels, metric="euclidean")
            )  # Calculate the silhouette score and append to the list
        n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        if n_clusters == 2:
            n_clusters = 3
        self.optimal_clusters = n_clusters  # Store the optimal number of clusters

    def apply_gmm_clustering(self):
        """
        Perform GMM clustering using the optimal number of clusters.
        """
        self.gmm = GaussianMixture(
            n_components=self.optimal_clusters,
            covariance_type='full',
            random_state=0,
        ).fit(
            self.df
        )  # Fit the GMM model to the data
        self.final_df = self.df.copy()  # Create a copy of the DataFrame
        self.final_df[
            "cluster"
        ] = self.gmm.predict(self.df)  # Add the cluster labels to the DataFrame
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
        self.apply_gmm_clustering()
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
        Train an SVC model and get metrics.
        :param X_train: DataFrame - Training feature set
        :param y_train: Series - Training target set
        :param X: DataFrame - Full feature set
        :param y: Series - Full target set
        :return: Tuple - F1 score and feature importances
        """
        model = SVC(kernel='linear', random_state=0)  # Initialize the SVC model
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
            model.coef_[0],
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

    def merge_original_data(self, data_preprocessor, feature_identifier):
        """
        Merge the original data with the cluster labels.
        :param data_preprocessor: DataPreprocessor - DataPreprocessor object
        :param feature_identifier: FeatureIdentifier - FeatureIdentifier object
        :return: None
        """
        # Merge the original data with the cluster labels
        self.df = pd.merge(
            data_preprocessor.original_df,
            feature_identifier.final_df[["userid", "cluster"]],
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
            original_df, final_df[["userid", "cluster"]], on=on, how=how
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
    handler.run_visualization(clusterer.final_df)

    # Merge original data with clustering results
    merged_df = handler.merge_dataframes(
        data_preprocessor.original_df, feature_identifier.final_df
    )
    merged_df["cluster"] = merged_df["cluster"].astype(int)

    # Read CSV users.csv
    users = pd.read_csv(working_data_path / "cleaned_data_users.csv")
    users = users[["userid", "city", "country"]]
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
