!pip install kneed

import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from kneed import KneeLocator
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input

def select_two_images_per_subdir(source_dir):
    """
    Selects the first two images from each subdirectory within a given source directory.
    
    Arguments:
    source_dir (str): Path to the source directory to search through.
    
    Returns:
    list: A list of selected image filenames.
    """
    selected_images = []
    for subdir, _, files in os.walk(source_dir):
        if files:
            files.sort()
            selected_images.extend(files[:2])
            for file in files[:2]:
                full_path = os.path.join(subdir, file)
                print(full_path)
    return selected_images

def load_model_and_extract_features(directory, batch_size=10, target_size=(224, 224)):
    """
    Loads a pre-trained ResNet50 model and extracts features for images in a specified directory.
    
    Arguments:
    directory (str): The directory containing images for feature extraction.
    batch_size (int): Number of images to process in a batch.
    target_size (tuple): Desired size of the images.
    
    Returns:
    tuple: A tuple containing the extracted features and filenames.
    """
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(directory, target_size=target_size, batch_size=batch_size, class_mode=None, shuffle=False)
    features = model.predict(generator, steps=np.ceil(generator.samples / batch_size))
    filenames = generator.filenames
    return features, filenames

def apply_dimensionality_reduction(features, n_components=50):
    """
    Applies PCA to reduce the dimensionality of extracted features.
    
    Arguments:
    features (np.array): High-dimensional feature array.
    n_components (int): Number of components to keep.
    
    Returns:
    np.array: Reduced feature array.
    """
    features_flattened = features.reshape(features.shape[0], -1)
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features_flattened)
    return reduced_features

def find_optimal_clusters(data, range_start=1, range_end=15):
    """
    Determines the optimal number of clusters using the elbow method.
    
    Arguments:
    data (np.array): The dataset to cluster.
    range_start (int): Starting range for testing cluster numbers.
    range_end (int): Ending range for testing cluster numbers.
    
    Returns:
    int: Optimal number of clusters.
    """
    sse = []
    for k in range(range_start, range_end + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        sse.append(kmeans.inertia_)
    knee_locator = KneeLocator(range(range_start, range_end + 1), sse, curve='convex', direction='decreasing')
    optimal_clusters = knee_locator.elbow
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(range_start, range_end + 1), sse, marker='o')
    plt.vlines(optimal_clusters, ymin=min(sse), ymax=max(sse), linestyles='dashed')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.show()
    
    return optimal_clusters

def cluster_and_visualize(data, cluster_labels):
    """
    Applies t-SNE for visualization and clusters data points.
    
    Arguments:
    data (np.array): The dataset to visualize and cluster.
    cluster_labels (np.array): Cluster labels for each data point.
    
    Returns:
    None: This function plots the t-SNE visualization.
    """
    tsne = TSNE(n_components=2, perplexity=50, n_iter=3000)
    tsne_results = tsne.fit_transform(data)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=cluster_labels, palette="viridis")
    plt.title('t-SNE visualization of product images')
    plt.show()

def save_cluster_mapping(filenames, labels, filepath='/content/aaaaaaa.csv'):
    """
    Saves the mapping of filenames to their respective cluster labels to a CSV file.
    
    Arguments:
    filenames (list): List of image filenames.
    labels (np.array): Cluster labels for each filename.
    filepath (str): Path to save the CSV file.
    
    Returns:
    None: This function saves the mapping to a CSV file.
    """
    image_cluster_mapping = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
    image_cluster_mapping_sorted = image_cluster_mapping.sort_values(by=['Cluster', 'Filename'])
    image_cluster_mapping_sorted.reset_index(drop=True, inplace=True)
    image_cluster_mapping_sorted.to_csv(filepath, index=False)

# Example usage
source_dir = "/content/selected_images"
selected_images = select_two_images_per_subdir(source_dir)

features, filenames = load_model_and_extract_features('/content', batch_size=10)
reduced_features = apply_dimensionality_reduction(features)
optimal_clusters = find_optimal_clusters(reduced_features)
kmeans_final = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans_final.fit_predict(reduced_features)

print(optimal_clusters)
cluster_and_visualize(reduced_features, cluster_labels)
save_cluster_mapping(filenames, cluster_labels)

