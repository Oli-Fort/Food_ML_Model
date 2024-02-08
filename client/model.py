import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans
import os

def load_dataset(file_path):
    """Loads the dataset from a given file path.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        DataFrame: A pandas DataFrame containing the loaded dataset.
    """
    return pd.read_csv(file_path)

def perform_kmeans(X, n_clusters=5, init='k-means++', random_state=0):
    """Performs K-Means clustering on the dataset.

    Args:
        X (array-like): The input data for clustering.
        n_clusters (int, optional): The number of clusters to form. Defaults to 5.
        init (str, optional): Method for initialization. Defaults to 'k-means++'.
        random_state (int, optional): Determines random number generation for centroid initialization. Defaults to 0.

    Returns:
        array, KMeans: The cluster labels for each point in the dataset and the trained KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=random_state)
    return kmeans.fit_predict(X), kmeans

def save_model(model, file_path):
    """Saves the trained model to a file.

    Args:
        model (KMeans): The trained KMeans model.
        file_path (str): The path to save the model file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def save_cluster_data(y, y_kmeans, file_name):
    """Saves the cluster data to a CSV file.

    Args:
        y (array-like): The original labels or data associated with the dataset's rows.
        y_kmeans (array-like): The cluster labels assigned by the KMeans algorithm.
        file_name (str): The name of the file to save the cluster data.
    """
    combined_data = pd.DataFrame(np.stack((y, y_kmeans), axis=1))
    combined_data.to_csv(file_name, index=False, header=False)

def plot_clusters(X, y_kmeans):
    """Plots the clusters in a 3D scatter plot.

    Args:
        X (array-like): The input data for clustering.
        y_kmeans (array-like): The cluster labels assigned by the KMeans algorithm.
    """
    ax = plt.axes(projection="3d")
    n_clusters = np.unique(y_kmeans).size
    for i in range(n_clusters):
        ax.scatter3D(X[y_kmeans == i, 1], X[y_kmeans == i, 2], X[y_kmeans == i, 3], s=10, label=f'Cluster {i+1}')
    plt.title('Food Products Clusters')
    ax.set_xlabel('Total Fat')
    ax.set_ylabel('Protein')
    ax.set_zlabel('Carbohydrates')
    plt.legend()
    plt.show()

# Main workflow
file_dir = os.path.dirname(__file__) # Directory of the current file.
dataset_path = os.path.join(file_dir, '../data/nutrition_cleaned.csv') # Path to the dataset.
model_path = os.path.join(file_dir, 'model_food.pk1') # Path to save the trained model.
output_file = 'foodXclusters.csv' # Name of the output file for cluster data.

# Load and prepare the dataset.
dataset = load_dataset(dataset_path)
X = dataset.iloc[:, 3:7].values # Input features for clustering.
y = dataset.iloc[:, 1] # Original labels or data.

# Perform K-Means clustering.
y_kmeans, kmeans_model = perform_kmeans(X)

# Save the trained model and cluster data.
save_model(kmeans_model, model_path)
save_cluster_data(y, y_kmeans, output_file)

# Plot the clusters in a 3D scatter plot.
plot_clusters(X, y_kmeans)
