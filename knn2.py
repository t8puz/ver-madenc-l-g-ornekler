from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Veri setini yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modelini oluşturma
knn = KNeighborsClassifier(n_neighbors=3)

# Modeli eğitme
knn.fit(X_train, y_train)

# Modeli test etme
y_pred = knn.predict(X_test)

# Doğruluk değerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
print("Test seti doğruluğu:", accuracy)
