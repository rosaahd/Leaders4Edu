import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# --------------------------------------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------------------------------------- #
class Visualizer:
    """
    Class to visualize the clusters using PCA.
    """
    def __init__(self, df, cluster_column="new_cluster"):
        self.df = df.drop(columns=["userid"], axis=1)
        self.moodle_ids = df["userid"].values
        self.cluster_column = cluster_column

    def plot_pca(self, save_path):
        pca = PCA(n_components=2)  # Initialize the PCA model
        features = self.df.drop(columns=[self.cluster_column])
        principal_components = pca.fit_transform(features)  # Fit the PCA model to the data and transform the data
        principal_df = pd.DataFrame(
            data=principal_components,
            columns=["Principal Component 1", "Principal Component 2"],
        )  # Create a DataFrame from the principal components
        principal_df[self.cluster_column] = self.df[self.cluster_column]  # Add the cluster column

        # Plot the PCA
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x="Principal Component 1",
            y="Principal Component 2",
            hue=self.cluster_column,
            data=principal_df,
            palette=sns.color_palette("hls", n_colors=principal_df[self.cluster_column].nunique()),
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
        principal_df = principal_df[["userid", self.cluster_column, "Principal Component 1", "Principal Component 2"]]

        # Save PCA data to CSV
        principal_df.to_csv(Path.cwd() / "data/clustering/pca.csv", index=False)
        print("PCA results saved to pca.csv and pca.svg")

def load_cluster_data(file_path, cluster_column_name):
    """
    Load cluster data from a CSV file.
    :param file_path: Path - path to the CSV file
    :param cluster_column_name: str - name of the cluster column to be renamed
    :return: pd.DataFrame - DataFrame with userid and cluster assignment
    """
    df = pd.read_csv(file_path)
    df = df.rename(columns={'cluster': cluster_column_name})
    return df[['userid', cluster_column_name]]

def load_rules_data(file_path):
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ['Cluster', 'Rules']):
        raise KeyError(f"File {file_path} does not contain required columns 'Cluster' and 'Rules'")
    return df[['Cluster', 'Rules']]

def merge_cluster_rules(cluster1, cluster2, rules_df1, rules_df2):
    rule1 = rules_df1[rules_df1['Cluster'] == cluster1]['Rules'].values[0] if not rules_df1[rules_df1['Cluster'] == cluster1].empty else ""
    rule2 = rules_df2[rules_df2['Cluster'] == cluster2]['Rules'].values[0] if not rules_df2[rules_df2['Cluster'] == cluster2].empty else ""
    return f"{rule1} AND {rule2}"

def save_new_clusters_and_rules(combined_df, base_data_path):
    combined_df['new_cluster'] = 'A' + (combined_df['cluster1'].astype(str) + combined_df['cluster2'].astype(str)).astype('category').cat.codes.astype(str)
    rules_df1 = load_rules_data(base_data_path / "rules1.csv")
    rules_df2 = load_rules_data(base_data_path / "rules2.csv")
    
    rules_combined_df = pd.DataFrame(columns=['new_cluster', 'rule'])

    for new_cluster in combined_df['new_cluster'].unique():
        cluster_comb = combined_df[combined_df['new_cluster'] == new_cluster].iloc[0]
        cluster1, cluster2 = cluster_comb['cluster1'], cluster_comb['cluster2']
        rule = merge_cluster_rules(int(cluster1), int(cluster2), rules_df1, rules_df2)
        rules_combined_df = rules_combined_df._append({'new_cluster': new_cluster, 'rule': rule}, ignore_index=True)
    
    combined_df.to_csv(base_data_path / "combined_clusters.csv", index=False)
    rules_combined_df.to_csv(base_data_path / "rules.csv", index=False)
    print("Combined clusters and rules saved to combined_clusters.csv and rules.csv")

def perform_pca_and_save(df, base_data_path):
    visualizer = Visualizer(df)
    visualizer.plot_pca(base_data_path / "images/pca.svg")

def generate_dataset_with_results(combined_df, base_data_path, working_data_path):
    # Cargar datos originales de usuarios
    users = pd.read_csv(working_data_path / "cleaned_data_users.csv")
    users = users[["userid", "city", "country"]]
    
    # Combinar con los resultados de los clusters
    combined_data = pd.merge(combined_df, users, on='userid', how='left')
    combined_data.to_csv(base_data_path / 'dataset_with_results.csv', index=False)
    print("Dataset with results saved to dataset_with_results.csv")

def main():
    base_data_path = Path("data/clustering")
    working_data_path = Path("data/working_data")

    cluster_df1 = load_cluster_data(base_data_path / "pca1.csv", 'cluster1')
    cluster_df2 = load_cluster_data(base_data_path / "pca2.csv", 'cluster2')
    
    combined_df = pd.merge(cluster_df1, cluster_df2, on='userid', how='outer')

    combined_df['cluster1'].fillna(20, inplace=True)
    combined_df['cluster2'].fillna(20, inplace=True)
    combined_df['cluster1'] = combined_df['cluster1'].astype(int)
    combined_df['cluster2'] = combined_df['cluster2'].astype(int)

    save_new_clusters_and_rules(combined_df, base_data_path)
    perform_pca_and_save(combined_df, base_data_path)
    generate_dataset_with_results(combined_df, base_data_path, working_data_path)

if __name__ == "__main__":
    main()
