import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Values.csv'  
data = pd.read_csv(file_path)

# Seperating the essential features and target variable
X = data.drop(columns=['h-index', 'i-10index', 'first_initial', 'last initial']) 
y = data['h-index'] 

# Standardising
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perfoming PCA
pca = PCA(n_components=2)  
X_pca = pca.fit_transform(X_scaled)

# Splitting into 4 color-codes
percentiles = np.percentile(y, [25, 50, 75]) 

def color_coding(h_index):
    if h_index < percentiles[0]:
        return 'red'  
    elif h_index < percentiles[1]:
        return 'orange'  
    elif h_index < percentiles[2]:
        return 'green'  
    else:
        return 'blue'  

colours_code = y.apply(color_coding)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colours_code, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of University Rank and Citations (2017-2023)')
plt.show()
