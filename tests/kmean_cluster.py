from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# Example dataset
existing_data = np.array([ [22, 1], [25, 2], [30, 5], [40, 10], [45, 15], [50, 20], [55, 25], [60, 30] ])  # Features: [Age, Experience]
new_data = np.array([[52, 30]])  # New applicant

# Cluster 0: "Junior Applicants" (young, less experienced).
# Cluster 1: "Senior Applicants" (older, more experienced).


# Step 1: Preprocess the data
scaler = StandardScaler()
existing_data_scaled = scaler.fit_transform(existing_data)
new_data_scaled = scaler.transform(new_data)

# Step 2: Fit K-Means clustering model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(existing_data_scaled)

# Step 3: Classify the new data
cluster_label = kmeans.predict(new_data_scaled)
print("Assigned Cluster:", cluster_label)

# print("Cluster Centers:", kmeans.cluster_centers_)


plt.scatter(existing_data_scaled[:, 0], existing_data_scaled[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x')  # Centroids
plt.title("Clusters and Centroids")
plt.xlabel("Feature 1 (Age)")
plt.ylabel("Feature 2 (Experience)")
plt.show()