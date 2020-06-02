# kmeans clustering
# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_customers.csv")
data = dataset.iloc[:,[3,4]].values
data.shape

##Visualizing the dataset
#plt.scatter(data[:,0],data[:,1], s = 10, c='black')
#plt.show()


# Elbow method for determining number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++' , max_iter = 300, n_init = 10)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
