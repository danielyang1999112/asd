import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
data = pd.read_csv('data.csv').to_numpy()
# Rescale the x and y values between 0 and 1
maxx = data.max(axis=0)
minn = data.min(axis=0)
data = (data - minn) / (maxx - minn)
#Initialise centroids
np.random.seed(1)
k = 2
cent = np.random.rand(k,2)
#Labels
n = data.shape[0]
labels = np.zeros(n)
new_labels = np.zeros(n)
dists = np.zeros(k)
#-------------------------------------------------------
converged = False
itera = 0
while converged == False:
    #Update labels
    for p in range(n):
        for c in range(k):
            dists[c] = np.linalg.norm(data[p,:] - cent[c,:])
        new_labels[p] = np.argmin(dists)
    #Update cent
    for c in range(k):
        cent[c] = np.mean(data[new_labels == c,:], axis=0)
    if np.array_equal(labels, new_labels):
        converged = True
    labels = new_labels
    itera += 1
print('Iteration: ', itera)
# Plot
plt.scatter(data[:,0], data[:,1], s = 20, label = 'Data', cmap = 'rainbow', c = new_labels)
plt.scatter(cent[:,0], cent[:,1], s=100, color='gold', edgecolors='black', label='Centroids')
# Label points with letters (this is extension, you don't need to know how to do this)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
for i in range(data.shape[0]):
    plt.text(data[i, 0] - 0.05, data[i, 1] - 0.04, letters[i], fontsize=12)

plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.axis('scaled'); plt.savefig('plot')
