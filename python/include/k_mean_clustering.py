import numpy as np

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