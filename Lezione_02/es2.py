import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (per il 3D)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns

# 1) Caricamento dati
data_path = Path("Lezione_02") / "Wholesale customers data.csv"
df = pd.read_csv(data_path)
# 1.1) Delte colonna 'Channel' e 'Region'
#df = df.drop(columns=['Channel', 'Region'])

print("Colonne:", df.columns.tolist())

# 2) Standardizzazione
X = df.values
X_scaled = StandardScaler().fit_transform(X)



# 3) k-distance plot (per la scelta di eps)
plt.figure(figsize=(6, 4))
for k in (3, 5, 8):
    nn = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    plt.plot(k_distances, label=f"k={k}")

plt.xlabel("Punti ordinati")
plt.ylabel("Distanza")
plt.title("k-distance plot")
plt.legend()
plt.tight_layout()
plt.show()

# 4) DBSCAN (parametri scelti dal k-distance plot)
dbscan = DBSCAN(eps=1.4, min_samples=3)
labels = dbscan.fit_predict(X_scaled)

# 5) Metriche/contatori
n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #-1 non va considerato come cluster perché i rumori
n_noise = int(np.sum(labels == -1))

if n_clusters >= 2:
    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {sil:.3f}")
else:
    print("Silhouette non calcolabile (meno di 2 cluster).")

print(f"Numero di cluster: {n_clusters}")
print(f"Punti rumorosi (noise): {n_noise}")

unique_labels = set(labels)
palette = sns.color_palette("Set2", len(unique_labels))
color_map = {
    label: palette[i] if label != -1 else (0.6, 0.6, 0.6)  # grigio per outlier
    for i, label in enumerate(sorted(unique_labels))
}
colors_dbscan = [color_map[label] for label in labels]


# 7) PCA: 2D e 3D 
pca2 = PCA(n_components=2,svd_solver='full')
X_pca2 = pca2.fit_transform(X_scaled)
var2 = pca2.explained_variance_ratio_.sum()

pca3 = PCA(n_components=3,svd_solver='full')
X_pca3 = pca3.fit_transform(X_scaled)
var3 = pca3.explained_variance_ratio_.sum()

print(f"Varianza spiegata PCA 2D: {var2:.2%}  ")
print(f"Varianza spiegata PCA 3D: {var3:.2%}  ")

# 8) Plot: confronto PCA 2D vs PCA 3D (stessi label DBSCAN)
fig = plt.figure(figsize=(12, 5))

# 2D
ax2d = fig.add_subplot(1, 2, 1)
ax2d.scatter(X_pca2[:, 0], X_pca2[:, 1], c=colors_dbscan, s=40, alpha=0.8, edgecolor="none")
ax2d.set_xlabel("PC1")
ax2d.set_ylabel("PC2")
ax2d.set_title(f"DBSCAN su PCA 2D — Var. spiegata: {var2:.1%}")

# 3D
ax3d = fig.add_subplot(1, 2, 2, projection="3d")
ax3d.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=colors_dbscan, s=30, alpha=0.8)
ax3d.set_xlabel("PC1")
ax3d.set_ylabel("PC2")
ax3d.set_zlabel("PC3")
ax3d.set_title(f"DBSCAN su PCA 3D — Var. spiegata: {var3:.1%}")

plt.tight_layout()
plt.show()
