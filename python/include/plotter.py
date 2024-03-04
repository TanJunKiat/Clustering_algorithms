import matplotlib.pyplot as plt

def plot_clusters(data, labels, centers):
    # Plot the data points with different colors for each cluster
    for i in range(len(centers)):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')

    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', label='Cluster Centers')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means Clustering')
    plt.legend()

    # Show the plot
    plt.show()

