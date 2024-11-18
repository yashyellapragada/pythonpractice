import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Values.csv')

citation_columns = ['numbers_of_citations_in_2017', 'numbers_of_citations_in_2018',
                    'numbers_of_citations_in_2019', 'numbers_of_citations_in_2020',
                    'numbers_of_citations_in_2021', 'numbers_of_citations_in_2022']

X = data[citation_columns]
y = data['numbers_of_citations_in_2023']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


#Found that 3 clusters is a good number for Kmeans fitting
# Fitting KMeans
opt_clusters = 3
kmeans = KMeans(n_clusters=opt_clusters, random_state=42)
kmeans.fit(X_train_scaled)

# Predicting the nearest cluster for the test data
test_clusters = kmeans.predict(X_test_scaled)
cluster_centers = kmeans.cluster_centers_

# Finding the nearest neighbor for each test point from the training set
nearest_neighbors_idx, _ = pairwise_distances_argmin_min(X_test_scaled, X_train_scaled)

# Prepare the predictions
predictions = {
    'Nearest Neighbor': y_train.iloc[nearest_neighbors_idx].values,  # nearest neighbor
    'Nearest Centroid': np.array([y_train.iloc[pairwise_distances_argmin_min([test], cluster_centers)[0][0]]
                                  for test in X_test_scaled]),  # nearest cluster centroid
    'Cluster Average': np.array([y_train.iloc[kmeans.labels_ == cluster].mean() for cluster in test_clusters])  # Average in the same cluster
}

# actual and predicted values
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted_Nearest_Neighbor': predictions['Nearest Neighbor'],
    'Predicted_Nearest_Centroid': predictions['Nearest Centroid'],
    'Predicted_Cluster_Average': predictions['Cluster Average']
})

# Calculating the average difference magnitude between predicted and actual values
results['Diff_Nearest_Neighbor'] = abs(results['Actual'] - results['Predicted_Nearest_Neighbor'])
results['Diff_Nearest_Centroid'] = abs(results['Actual'] - results['Predicted_Nearest_Centroid'])
results['Diff_Cluster_Average'] = abs(results['Actual'] - results['Predicted_Cluster_Average'])

# Calculate average difference magnitudes
averdiff_neighbor = results['Diff_Nearest_Neighbor'].mean()
averdiff_centroid = results['Diff_Nearest_Centroid'].mean()
averdiff_cluster = results['Diff_Cluster_Average'].mean()

# Display results
print("Average Difference Magnitude:")
print(f"Nearest Neighbor: {averdiff_neighbor.round(2)}")
print(f"Nearest Cluster Centroid: {averdiff_centroid.round(2)}")
print(f"Cluster Average: {averdiff_cluster.round(2)}")

# Saving the rounded results to a CSV file
output_file_path = 'HW4_Predictions.csv'
results.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
