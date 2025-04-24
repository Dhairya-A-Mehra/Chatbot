import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster  
from sklearn.metrics import silhouette_score
import json

# Load data
cooc_matrix = np.load("data/symptom_cooc.npy")
with open("data/doctors.json", "r") as f:
    doc_data = json.load(f)

# Get optimal cluster count
all_doctor_types = set()
for doctors in doc_data["mappings"].values():
    all_doctor_types.update(doctors)
n_clusters = len(all_doctor_types) or 5

# --- K-means Clustering ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(cooc_matrix)
kmeans_silhouette = silhouette_score(cooc_matrix, kmeans_labels)

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
cooc_2d = pca.fit_transform(cooc_matrix)

# Plot K-means clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    cooc_2d[:, 0], cooc_2d[:, 1],
    c=kmeans_labels, cmap="viridis", alpha=0.6
)
plt.colorbar(scatter, label="Cluster ID")
plt.title("K-means Clustering of Symptoms (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig("data/kmeans_clusters.png")

# --- Hierarchical Clustering ---
Z = linkage(cooc_matrix, method="ward")
hierarchical_labels = fcluster(Z, t=n_clusters, criterion='maxclust')  # Now properly defined
hierarchical_silhouette = silhouette_score(cooc_matrix, hierarchical_labels)

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=12)
plt.title("Hierarchical Clustering Dendrogram")
plt.savefig("data/hierarchical_dendrogram.png")

# Plot comparison
plt.figure(figsize=(10, 5))
plt.bar(['K-means', 'Hierarchical'], 
        [kmeans_silhouette, hierarchical_silhouette],
        color=['blue', 'orange'])
plt.title("Clustering Algorithm Comparison (Silhouette Score)")
plt.ylabel("Score (Higher is better)")
plt.savefig("data/clustering_comparison.png")

print("Visualizations saved to /data/ directory!")