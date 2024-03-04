import numpy as np
from PIL import Image

def kmeans(X, n_clusters):
    # Initialize cluster centers randomly
    centers = X[np.random.choice(range(len(X)), n_clusters, replace=False)]

    # Iterate until convergence
    while True:
        # Assign each data point to the nearest cluster center
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=-1), axis=-1)

        # Update cluster centers
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Check for convergence
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return labels, centers

# Load the image
image = Image.open("python/examples/example_3.png")

# Convert the image to a numpy array
X = np.array(image)

# Find the coordinates of black pixels
black_pixels = np.argwhere(np.all(X == [0, 0, 0], axis=-1))

# Create a KMeans object with the desired number of clusters
n_clusters = 3

# Fit the data to the KMeans model
labels, centers = kmeans(black_pixels, n_clusters)

# Print the cluster labels and centers
print("Cluster Labels:")
print(labels)
print("\nCluster Centers:")
print(centers)
import matplotlib.pyplot as plt

# Plot the data points with different colors for each cluster
for i in range(n_clusters):
    plt.scatter(black_pixels[labels == i, 0], black_pixels[labels == i, 1], label=f'Cluster {i+1}')

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', label='Cluster Centers')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering')
plt.legend()

# Show the plot
plt.show()