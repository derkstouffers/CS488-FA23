# Deric Shaffer
# CS488 - Assignment 5
# Due Date - Nov. 1st, 2023

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# data
data = pd.read_csv('google_review_ratings.csv')

data.dropna(inplace = True)
X = data.drop(['User'], axis = 1)

kmeans_avg_silhouette = []
gmm_avg_silhouette = []
spectral_avg_silhouette = []

for k in range(2,21):
    kmeans_silhouette = 0
    gmm_silhouette = 0
    spectral_silhouette = 0

    for _ in range(5):
        
        # k-means
        k_means = KMeans(n_clusters = k, n_init = 10, random_state = 19)
        kmeans_labels = k_means.fit_predict(X)
        kmeans_silhouette += silhouette_score(X, kmeans_labels)

        # gauss
        gmm = GaussianMixture(n_components = k, n_init = 10, random_state = 19)
        gmm_labels = gmm.fit_predict(X)
        gmm_silhouette += silhouette_score(X, gmm_labels)
        
        # spectral
        spec = SpectralClustering(n_clusters = k, n_init = 10, affinity='nearest_neighbors', random_state = 19)
        spectral_labels = spec.fit_predict(X)
        spectral_silhouette += silhouette_score(X, spectral_labels)
      
    # calculate avg
    kmeans_silhouette /= 5
    gmm_silhouette /= 5
    spectral_silhouette /= 5

    kmeans_avg_silhouette.append(kmeans_silhouette)
    gmm_avg_silhouette.append(gmm_silhouette)
    spectral_avg_silhouette.append(spectral_silhouette)


# determine best k
best_k_means = np.argmax(kmeans_avg_silhouette)
best_k_gmm = np.argmax(gmm_avg_silhouette)
best_k_spec = np.argmax(spectral_avg_silhouette)



# report centroids
k_means = KMeans(n_clusters = best_k_means, n_init = 10)
labels = k_means.fit_predict(X)

centroids = k_means.cluster_centers_


 
# k-means pca
pca = PCA(n_components = 2)
pca_kmeans = pca.fit_transform(X)
kmeans_cluster = k_means.labels_

plt.figure(figsize = (10,6))
sns.scatterplot(x=pca_kmeans[:, 0], y=pca_kmeans[:, 1], hue=kmeans_cluster, palette='deep')
plt.title('PCA Projection of Clusters (k-Means)')
plt.legend()
plt.show()


# gmm pca
best_k_gmm = GaussianMixture(n_components=best_k_gmm, n_init=10, random_state=19)
pca = PCA(n_components = 2)
pca_gmm = pca.fit_transform(X)
gmm_cluster = best_k_gmm.fit_predict(X)

plt.figure(figsize = (10,6))
sns.scatterplot(x=pca_gmm[:, 0], y=pca_gmm[:, 1], hue=gmm_cluster, palette='deep')
plt.title('PCA Projection of Clusters (GMM)')
plt.legend()
plt.show()



# spectral pca
best_k_spec = SpectralClustering(n_clusters=best_k_spec, n_init=10, affinity='nearest_neighbors', random_state=19)
pca = PCA(n_components = 2)
pca_spec = pca.fit_transform(X)
spectral_cluster = best_k_spec.fit_predict(X)

plt.figure(figsize = (10,6))
sns.scatterplot(x=pca_spec[:, 0], y=pca_spec[:, 1], hue=spectral_cluster, palette='deep')
plt.title('PCA Projection of Clusters (Spectral)')
plt.legend()
plt.show()



# graph all 3 together
plt.figure(figsize=(10, 6))
plt.plot(range(2, 21), kmeans_avg_silhouette, marker = 'o', linestyle = '-', label = 'K-Means Model', color = 'blue')
plt.plot(range(2, 21), gmm_avg_silhouette, marker = 'o', linestyle = '-', label = 'Gaussian Mixture Model', color = 'red')
plt.plot(range(2, 21), spectral_avg_silhouette, marker = 'o', linestyle = '-', label = 'Spectral Clustering', color = 'green')
plt.title('Average Silhouette Coefficient vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Coefficient')
plt.grid(True)
plt.legend()
plt.show()



# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
dbscan_labels = dbscan.labels_
dbscan_silhouette = silhouette_score(X, dbscan_labels)


# print all non-graphical outputs to text file
with open('output.txt', 'w') as file:
    file.write('Best K (k-Means) = ' + str(best_k_means) + '\n')
    file.write('Best K (GMM) = ' + str(best_k_gmm) + '\n')
    file.write('Best K (Spectral) = ' + str(best_k_spec) + '\n')

    file.write("Centroids of Clusters: \n")
    file.write(str(centroids) + '\n')

    file.write('DBScan Silhouette = ' + str(dbscan_silhouette) + '\n')
