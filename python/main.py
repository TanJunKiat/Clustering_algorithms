import numpy as np
from PIL import Image

from include.k_mean_clustering import kmeans
from include.plotter import plot_clusters

# Load the image
image = Image.open("python/examples/example_2.png")

# Find the coordinates of black pixels as make them the data points
data = np.argwhere(np.all(np.array(image) == [0, 0, 0], axis=-1))

# Create a KMeans object with the desired number of clusters
n_clusters = 2

# Fit the data to the KMeans model
labels, centers = kmeans(data, n_clusters)

plot_clusters(data, labels, centers)