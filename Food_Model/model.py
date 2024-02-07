import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

dataset = pd.read_csv("nutrition_cleaned.csv")
X = dataset.iloc[:, 4:7].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 5, init= 'k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

filename = '../.venv/model_food.pk1'
pickle.dump(kmeans, open(filename, 'wb'))


y = dataset.iloc[:, 1]
val = np.stack((y, y_kmeans), axis=1)
a = pd.DataFrame(val)
filename = 'foodXclusters.csv'
a.to_csv(filename, index=False, header=False)


ax = plt.axes(projection= "3d")
ax.scatter3D(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s = 10, c = 'red', label= 'Cluster 1')
ax.scatter3D(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s = 10, c = 'blue', label= 'Cluster 2')
ax.scatter3D(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s = 10, c = 'green', label= 'Cluster 3')
ax.scatter3D(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], X[y_kmeans == 3, 2], s = 10, c = 'cyan', label= 'Cluster 4')
ax.scatter3D(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], X[y_kmeans == 4, 2], s = 10, c = 'magenta', label= 'Cluster 5')
plt.title('Food Products Clusters')
ax.set_xlabel('Total Fat')
ax.set_ylabel('Protein')
ax.set_zlabel('Carbohydrates')
plt.show()
