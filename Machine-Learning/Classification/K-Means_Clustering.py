'''
K-Means Clustering
    - Use scikit-learn's K-Means Clustering to cluster data
'''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 

from IPython.display import display
import requests

# K-MEANS ON A RANDOMLY GENERATED DATASET
np.random.seed(0) # Create a random seed
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2,-3], [1,1]],
                  cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
#plt.show()

# SETTING UP K-MEANS
k_means = KMeans(init='k-means++', n_clusters=4, n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_ # Grab the labels for each point in the model
k_means_cluster_centers = k_means.cluster_centers_ # Get the coordinates of the 
                                                   # cluster centers

# CREATING THE VISUAL PLOT
# Initialize the plot wit the specified dimensions
fig = plt.figure(figsize=(6,4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
#plt.show()


# CUSTOMER SEGMENTATION WITH K-MEANS
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'\
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/'\
    'Cust_Segmentation.csv'
r = requests.get(path, allow_redirects=True)
open('Classification/Cust_Segmentation.csv', 'wb').write(r.content)

df = pd.read_csv('Classification/Cust_Segmentation.csv')
display(df.head(10))

# Address is a categorical variable, we need to drope it
df = df.drop('Address', axis=1)

# Normalize over the standard deviation
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

# MODELING
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

df["Clus_km"] = labels
df.head(5)
df.groupby('Clus_km').mean()

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))
plt.show()