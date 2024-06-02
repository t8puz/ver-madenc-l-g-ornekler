import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Örnek veri oluşturma
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# KMeans modelini oluşturma
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Kümeleri tahmin etme
y_kmeans = kmeans.predict(X)

# Küme merkezlerini alma
centers = kmeans.cluster_centers_

# Sonuçları görselleştirme
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.show()
