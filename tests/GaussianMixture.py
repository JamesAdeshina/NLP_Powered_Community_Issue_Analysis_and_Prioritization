from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample dataset: Age and Experience
existing_data = np.array([
    [22, 1], [25, 2], [30, 5], [40, 10],
    [45, 15], [50, 20], [55, 25], [60, 30]
])
new_data = np.array([[50, 15]])  # Older with minimal experience (edge case)


# Cluster 0: "Junior Applicants" (young, less experienced).
# Cluster 1: "Senior Applicants" (older, more experienced).


# Step 1: Scale the data (important for clustering)
scaler = StandardScaler()
existing_data_scaled = scaler.fit_transform(existing_data)
new_data_scaled = scaler.transform(new_data)

# Step 2: Fit Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, random_state=42)  # Two clusters: Junior and Senior
gmm.fit(existing_data_scaled)

# Step 3: Predict probabilities for cluster membership
probabilities = gmm.predict_proba(new_data_scaled)
print("Cluster Probabilities:", probabilities)

# Step 4: Determine the most likely cluster
most_likely_cluster = gmm.predict(new_data_scaled)
print("Assigned Cluster:", most_likely_cluster)
