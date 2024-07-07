import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # for plotting on a map
import cartopy.feature as cfeat


# For preprocessing/loading the data.
def preprocess(file_path):
    data = pd.read_csv(file_path)
    filtered_data = data[(data['StateName'] == 'TELANGANA')]  # This isn't sufficient because of the inaccuracies in the dataset
    filtered_data = filtered_data.astype({'Longitude': 'float', 'Latitude': 'float'})
    filtered_data.dropna(subset=['Longitude', 'Latitude'], inplace=True)  # Handling the NaNs
    filtered_data.drop_duplicates()
    filtered_data = filtered_data[
        (filtered_data['Longitude'] >= 77) & (filtered_data['Longitude'] <= 82.5) &
        (filtered_data['Latitude'] >= 15) & (filtered_data['Latitude'] <= 20.5)
    ]
    return filtered_data[['Longitude', 'Latitude']].values


# Returns an array of distances from each point in the data with respect to the point.
def get_dists(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


# Not a very smart initialization, picks k distinct centroids from the dataset randomly.
def init_centroids(X, k):
    n = X.shape[0]
    rand_indices = np.random.choice(n, k, replace=False)
    centroids = X[rand_indices]
    return np.array(centroids)


# Smart initialization, uses k-means++ probabilistic algorithm.
def init_centroids_kpp(X, k):
    n = X.shape[0]
    centroids = []
    centroid_idx = np.random.choice(n)
    centroids.append(X[centroid_idx])

    for _ in range(1, k):
        distances = np.min([np.sum((X - centroid) ** 2, axis=1) for centroid in centroids], axis=0)
        probs = distances / np.sum(distances)
        centroid_idx = np.random.choice(n, p=probs)
        centroids.append(X[centroid_idx])

    return np.array(centroids)


def kmeans(X, k, max_iters=20, init_using='kpp'):
    if init_using == 'kpp':
        centroids = init_centroids_kpp(X, k)
    else:
        centroids = init_centroids(X, k)
    
    for it in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = get_dists(point, centroids)
            centroid_idx = np.argmin(distances)
            clusters[centroid_idx].append(point)

        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(centroids[len(new_centroids)])  # Handle empty clusters

        new_centroids = np.array(new_centroids)
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    clusters = [np.array(cluster) for cluster in clusters]
    return clusters, centroids


# Calculating spread, sum of squares of distances from the centroid to the data points in a cluster.
def get_spread(X, centroids, clusters):
    spread = 0
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            distances = get_dists(centroids[i], cluster)
            spread += np.sum(distances ** 2)
    return spread


# Plotting the elbow graph, rate of decrease in variance/spread with an increase in k.
def plot_spread(X, k_range, init_using, ax):
    spreads = []
    for k in k_range:
        clusters, centroids = kmeans(X, k, init_using=init_using)
        spread = get_spread(X, centroids, clusters)
        spreads.append(spread)

    ax.plot(k_range, spreads, marker='o', linestyle='-', markersize=8, label=f'Init: {init_using}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title(f'Elbow Method ({init_using})', fontsize=14)
    ax.grid(True)
    ax.legend()


def plot_clusters(X, clusters, centroids, init_using, ax):
    ax.set_extent([77, 82.5, 15, 20.5])  # Showing Telangana.
    ax.add_feature(cfeat.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeat.BORDERS, linewidth=0.5)
    ax.add_feature(cfeat.STATES, linewidth=0.5)

    colors = ['blue', 'orange', 'green', 'red', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            ax.scatter(cluster[:, 0], cluster[:, 1], marker='o', s=40, color=colors[i], label=f'Cluster {i+1}', alpha=0.7, edgecolors='black', linewidth=0.5, transform=ccrs.PlateCarree())
    #(used GPT for styling the graph here).
    ax.set_title(f'Clusters ({init_using})', fontsize=14)
    ax.legend(loc='upper right')  # Add the legend


def main():
    data = 'clustering_data.csv'
    Train_data = preprocess(data)

    # Range of k in the graph shown.
    k_range = range(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_spread(Train_data, k_range, init_using='random', ax=axes[0])
    plot_spread(Train_data, k_range, init_using='kpp', ax=axes[1])
    plt.tight_layout()
    plt.show()

    chosen_k = 3  # Adjust based on the graph or choice.
    clusters_random, centroids_random = kmeans(Train_data, chosen_k, init_using='random')
    clusters_kpp, centroids_kpp = kmeans(Train_data, chosen_k, init_using='kpp')
    _, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_clusters(Train_data, clusters_random, centroids_random, init_using='random', ax=axes[0])
    plot_clusters(Train_data, clusters_kpp, centroids_kpp, init_using='kpp', ax=axes[1]) #Graphs side-by-side for easier comparision.
    plt.tight_layout()
    plt.show()


main()


'''
Insights/Observations:
1. Without filtering the latitudes & longitudes some clusters are going completly outside Telanagana even if there are only
25-30 data points with StateName = 'TELANGANA' outside telangana, they are so far away that they are able to make that impact
on the centriods of clusters.

2. The difference in the way kpp and normal intialialization clustering, seems to be more apparent as k increases. kpp seems 
to be doing better at higher values of k.

3. I wanted to see if the algorithm is actually converging or were we just running out of max_iterations. So, i ran the algorithm 
5 times and put a print statement before break and here are the results:

Algorithm converged at run : 18 with random initialization.
Algorithm converged at run : 18 with kpp initialization.
----------------------------------------------
Algorithm converged at run : 13 with random initialization.
Algorithm converged at run : 15 with kpp initialization.
----------------------------------------------
Algorithm converged at run : 11 with random initialization.
Algorithm converged at run : 10 with kpp initialization.
----------------------------------------------
Algorithm converged at run : 13 with random initialization.
Algorithm converged at run : 14 with kpp initialization.
----------------------------------------------
Algorithm converged at run : 15 with random initialization.
Algorithm converged at run : 10 with kpp initialization.

(its nice how its a tie between both on who finished faster).
I ran a few more tests and its almost always converging before 20 iterations.

4. I wanted to know how random this model is, how much is it varying so i ran a few tests (with k=3 & kpp)
   
Size of Cluster 1: 1838 Centriod: [78.79407076 18.68603304]
Size of Cluster 2: 926 Centriod: [78.23295127 16.8971514 ]
Size of Cluster 3: 1446 Centriod: [79.96949569 17.45416172]
--------------------------------------------------
Size of Cluster 1: 1837 Centriod: [78.79403886 18.68657762]
Size of Cluster 2: 930 Centriod: [78.23644569 16.89882954]
Size of Cluster 3: 1443 Centriod: [79.97128331 17.45478463]
--------------------------------------------------
Size of Cluster 1: 930 Centriod: [78.23644569 16.89882954]
Size of Cluster 2: 1837 Centriod: [78.79403886 18.68657762]
Size of Cluster 3: 1443 Centriod: [79.97128331 17.45478463]

So it seems to have a decent precison.


5. Looking at the data it seems like the given data points correspond to different postal offices.
Densely packed clusters can help us identify regions with high postal offices, more postal offices can mean that
the region is more developed. Can help us group under-developed regions.

6. One interesting real life usage can be to find the shortest (or most cost efficient) delivery route from 
one office to another. These clusters help identify which offices are closer to our office and hence help in optimizing 
the devilery routes. (ref : https://www.jmess.org/wp-content/uploads/2021/11/JMESSP13420802.pdf.)
'''