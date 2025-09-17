import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("Lezione_02/Mall_Customers.csv")

# Select features Annual Income (k$) and Spending Score (1-100)
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].copy()

# Standardize the dataframe z-score
std = StandardScaler()
X_standardized = pd.DataFrame(std.fit_transform(X), columns=X.columns, index=X.index)
# X_standardized = (X - X.mean()) / X.std()

# Test K= 2...10 and calculate elbow (inertia) and silhouette score
inertia = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_standardized)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_standardized, kmeans.labels_))


# Plot elbow and silhouette score
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia")
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker="o", color="orange")
plt.title("Silhouette Score For Optimal k")
plt.xlabel("Number of clusters k")
plt.ylabel("Silhouette Score")
plt.show()

# From the plots, choose the optimal k (for example, k=5)
optimal_k = 5
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(X_standardized)
# Add cluster labels to the original dataframe
cluster_labels = kmeans_optimal.labels_
centroids = kmeans_optimal.cluster_centers_

X_plot = X_standardized.copy()
X_plot["Cluster"] = cluster_labels

# Plot the clusters - SPAZIO STANDARDIZZATO
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=X_plot,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100,
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    s=200,
    alpha=0.75,
    marker="X",
    label="Centroids",
)
plt.title("KMeans Clustering Results (Standardized Space)")
plt.xlabel("Annual Income (standardized)")
plt.ylabel("Spending Score (standardized)")
plt.legend(title="Cluster")
plt.show()

# Guardando i cluster standardizzati, possiamo notare che sulla X (Annual Income) , mentre sulla Y (Spending Score)

# Basso reditto e Alto spending score in alto a sinistra cluster 2 possiamo classificarli come clienti impulsivi
# Basso reddito e basso spending score in basso a sinistra cluster 4 possiamo classificarli come clienti che economici
# Alto reddito e alto spending score in alto a destra cluster 1 possiamo classificarli come clienti premium
# Alto reddito e basso spending score in basso a destra cluster 3 possiamo classificarli come clienti parsimoniosi

# Facciamo i nostri calcoli sul dataframe non standaridizzato
X["Cluster"] = cluster_labels

# Calcolo: media reddito, media spending, % clienti per cluster
cluster_summary = (
    X.groupby("Cluster")
    .agg(
        mean_income=("Annual Income (k$)", "mean"),
        mean_spending=("Spending Score (1-100)", "mean"),
        count=("Cluster", "size"),  # conta righe per cluster
    )
    .assign(perc_clienti=lambda x: x["count"] / x["count"].sum() * 100)
    .reset_index()
    .sort_values("perc_clienti", ascending=False)
)

print(cluster_summary)
