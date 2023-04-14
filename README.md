# 5P30-Final-Project: Graph Neural Network-based Clustering Enhancement in VANET for Cooperative Driving
This project aims to cluster the vehicles in the HighD dataset using three different algorithms, namely k-means, spectral clustering, GraphSage convolution, and graph autoencoder-kmeans. The dataset used in this project contains a collection of traffic data captured from a highway in Germany. The data consists of a set of trajectories, which include the positions, velocities, and other information of the vehicles in the scene.
## Requirements
The following libraries are required to run this project:

Python (>=3.6)
numpy
pandas
scikit-learn
networkx
pytorch
torch-geometric
opencv-python
To install these libraries, you can use pip package manager by running the following command:

pip install numpy pandas scikit-learn networkx pytorch torch-geometric opencv-python

## data
The data used in this project is the HighD dataset, which is a collection of traffic data captured from a highway in Germany. The dataset can be downloaded from the following link:

https://www.highd-dataset.com/

The data frame used in this project is a 34 rows x 8 columns pandas DataFrame. Each row represents a car and has the following features: x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration.

## Graph Construction
To create a graph for the vehicle clustering, each row in the dataset is represented as a node in a NetworkX graph. Each node has the same features as the original DataFrame: x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration.

To connect the neighbors, the x and y positions are used to determine which nodes are near each other. The distances between cars are calculated, and each car is connected to the three nearest cars using undirected edges.

Next, random weights are assigned to the edges between 1 and 5 which shows whether the cars are moving in one path or not. This graph with nodes and their features, edges and their weights forms the basis for the clustering algorithms.

![alt text](https://github.com/nazaninmehregan/5P30-Final-Project/blob/master/graphs/graph_construction.png)

## Clustering Algorithms
Three different clustering algorithms are used in this project:

K-Means Clustering
Spectral Clustering
GraphSage convolution
Graph autoencoder-kmeans
### K-Means Clustering
The K-Means clustering algorithm is used to cluster the vehicles in the HighD dataset. The algorithm partitions the nodes into K clusters based on the similarity of their features. The number of clusters K is chosen based on an elbow method, where the optimal K value is chosen based on the point where the decrease in the sum of squared distances between the data points and their cluster centroids starts to level off.

![alt text](https://github.com/nazaninmehregan/5P30-Final-Project/blob/master/Graphs/kmeans_output.png)

### Spectral Clustering
The Spectral Clustering algorithm is used to cluster the vehicles in the HighD dataset. The algorithm uses the graph constructed earlier to create a Laplacian matrix, which is then used to perform eigenvalue decomposition. The resulting eigenvectors are used to project the nodes into a lower-dimensional space, where they can be clustered using K-Means clustering.

![alt text](https://github.com/nazaninmehregan/5P30-Final-Project/blob/master/Graphs/spectral_output.png)

### GraphSage convolution
The GCN - Sage Convolution algorithm is used to cluster the vehicles in the HighD dataset. The algorithm is a Graph Convolutional Network (GCN) that uses the graph constructed earlier as input. The GCN applies a series of graph convolution layers to the graph, where each layer aggregates information from the node's neighbors and updates the node's feature representation. The final node representations are then clustered using K-Means clustering.

![alt text](https://github.com/nazaninmehregan/5P30-Final-Project/blob/master/Graphs/graphsage_output.png)


### Graph Autoencoder and K-Means Clustering
We built a graph dataset using the PyTorch library. This data variable has features and edge indexes that are crucial for us to separate the testing and training sub-nodes. We have converted node features and edges to PyTorch tensors. We have defined our encoder and applied kmeans on the embedding output in the latent space.

![alt text](https://github.com/nazaninmehregan/5P30-Final-Project/blob/master/Graphs/GAE&kmeans_output#1.png)

![alt text](https://github.com/nazaninmehregan/5P30-Final-Project/blob/master/Graphs/GAE&kmeans_output.png)

## Conclusion
This project demonstrates the use of four different clustering algorithms to cluster the vehicles in the HighD dataset. By comparing the results of each algorithm, we can gain insights into the strengths and weaknesses of each method and choose the most suitable one for the task at hand. The combination of graph construction, feature extraction, and graph convolutional network is a powerful technique for clustering data points, especially in the case of high-dimensional data like the vehicle features in the HighD dataset.

