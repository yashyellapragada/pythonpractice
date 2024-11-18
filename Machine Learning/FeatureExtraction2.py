import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the CSV file
file_path = 'Values.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Step 2: Separate the features and the target variable (2023 citation numbers)
X = data.drop(columns=['numbers_of_citations_in_2023', 'h-index', 'i-10index', 'first_initial', 'last initial'])  # Use only 2017-2022 citations and rank
y = data['numbers_of_citations_in_2023']  # Target variable (2023 citation numbers)

# Step 3: Standardize the features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Perform PCA
pca = PCA(n_components=2)  # We'll reduce it to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Step 5: Split the 2023 citation numbers into 4 color-coded groups (percentiles)
percentiles = np.percentile(y, [25, 50, 75])  # 26th, 50th, and 75th percentiles

def color_coding(citation_2023):
    if citation_2023 < percentiles[0]:
        return 'red'  # < 26th percentile
    elif citation_2023 < percentiles[1]:
        return 'orange'  # 26th-50th percentile
    elif citation_2023 < percentiles[2]:
        return 'green'  # 51st-75th percentile
    else:
        return 'blue'  # 76th-100th percentile

colors = y.apply(color_coding)

# Step 6: Plot the results of PCA
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of University Rank and Citations (2017-2022)')
plt.show()

# Step 7: Analyze the results
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by the first two components: {explained_variance}')
