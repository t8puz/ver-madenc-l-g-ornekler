import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def minkowski_distance(p1, p2, p):
    return np.power(np.sum(np.abs(p1 - p2)**p), 1/p)

# Örnek noktalar
p1 = np.array([1, 2])
p2 = np.array([4, 6])

# Mesafelerin hesaplanması
euclidean = euclidean_distance(p1, p2)
manhattan = manhattan_distance(p1, p2)
minkowski_p3 = minkowski_distance(p1, p2, 3)

# Görselleştirme
plt.figure(figsize=(10, 5))

# Noktaların çizimi
plt.scatter(*p1, c='blue', label='P1 (1, 2)')
plt.scatter(*p2, c='red', label='P2 (4, 6)')

# Euclidean mesafesi çizimi
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], label=f'Euclidean (d={euclidean:.2f})')

# Manhattan mesafesi çizimi
plt.plot([p1[0], p1[0]], [p1[1], p2[1]], 'g--')
plt.plot([p1[0], p2[0]], [p2[1], p2[1]], 'g--', label=f'Manhattan (d={manhattan:.2f})')

# Minkowski mesafesi çizimi
plt.plot([p1[0], p1[0]], [p1[1], p2[1]], 'k-.')
plt.plot([p1[0], p2[0]], [p2[1], p2[1]], 'k-.', label=f'Minkowski (p=3, d={minkowski_p3:.2f})')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Euclidean, Manhattan, and Minkowski Distances')
plt.grid(True)
plt.show()
