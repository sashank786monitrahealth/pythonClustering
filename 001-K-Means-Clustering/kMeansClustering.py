# kmeans clustering
# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_customers.csv")
data = dataset.iloc[:,[3,4]].values
data.shape

##Visualizing the dataset
plt.scatter(data[:,0],data[:,1], s = 10, c='black')

