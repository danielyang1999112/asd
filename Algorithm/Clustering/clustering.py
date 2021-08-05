from sklearn.cluster import KMeans
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

m = pd.read_csv('DirectMarketingSubset.csv')[['Salary', 'AmountSpent']].to_numpy()
#Rescaling
minn = m.min(axis=0)
maxx = m.max(axis=0)
m = (m - minn) / (maxx - minn)
#Define the number of clusters k = 3; Fit data; Assign to the closest cent; Coordinates of cents
k = KMeans(n_clusters=3)
k.fit(m)
labels = k.labels_
c = k.cluster_centers_
#Plot
plt.scatter(m[:,0], m[:,1], s = 20, label = 'Data', cmap = 'rainbow', c = labels)
plt.scatter(c[:,0], c[:,1], s = 100, label = 'Centroids', color = 'gold', edgecolors = 'black')
plt.legend()
plt.xlabel('Salary')
plt.ylabel('Amount Spent')
plt.title('AmountSpent vs Salary')
plt.savefig('asd')
