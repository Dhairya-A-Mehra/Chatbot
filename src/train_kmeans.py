import numpy as np
from sklearn.cluster import KMeans
import joblib
import json

# Load co-occurrence matrix
cooc_matrix = np.load("data/symptom_cooc.npy")

# Load doctor data to determine optimal number of clusters
with open("data/doctors.json", "r") as f:
    doc_data = json.load(f)

# Get all unique doctor types to determine number of clusters
all_doctor_types = set()
for doctors in doc_data["mappings"].values():
    all_doctor_types.update(doctors)
n_clusters = len(all_doctor_types) or 5  # Fallback to 5 if empty

# Train K-means with optimal number of clusters
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10,
    init='k-means++',  # Better initialization than random
    max_iter=300  # Increased iterations for better convergence
).fit(cooc_matrix)

# Save the full model
joblib.dump(kmeans, "data/kmeans_model.joblib")
print(f"K-means model with {n_clusters} clusters trained and saved to data/kmeans_model.joblib")