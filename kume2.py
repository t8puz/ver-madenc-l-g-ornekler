from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Örnek veri seti
X = [
    [1, 2],
    [5, 8],
    [1.5, 1.8],
    [8, 8],
    [1, 0.6],
    [9, 11]
]

# KMeans modelini oluşturma
kmeans = KMeans(n_clusters=2)

# Modeli eğitme
kmeans.fit(X)

# Küme merkezlerini almak
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Küme merkezlerini ve noktaları çizdirme
colors = ["g.", "r."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()
