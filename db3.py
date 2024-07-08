# import scipy.io
# import pandas as pd

# # Load the .mat file
# mat = scipy.io.loadmat('db3.mat')

# # Replace 'your_data.mat' with the actual file path.

# Convert the loaded data to a DataFrame (assuming it's a dictionary with keys)
# data_dict = mat['db3']  # Replace 'your_variable_name' with the actual variable name in the .mat file
# df = pd.DataFrame(data_dict)

# # Save the DataFrame to a CSV file
# df.to_csv('output_data.csv', index=False)

# # Replace 'output_data.csv' with the desired output file path.

# import matplotlib.pyplot as plt
# from sklearn.cluster import STDBSCAN
# from sklearn.preprocessing import StandardScaler

# # Load your dataset 'db3' here, assuming it's a NumPy array with spatial and temporal coordinates

# # Standardize the data if necessary
# scaler = StandardScaler()
# db3_scaled = scaler.fit_transform(db3)

# # Define ST-DBSCAN parameters
# epsilon = 0.1  # Spatial neighborhood radius
# min_samples = 5  # Minimum number of samples in the neighborhood

# # Create and fit the ST-DBSCAN model
# st_dbscan = STDBSCAN(eps=epsilon, min_samples=min_samples)
# st_dbscan.fit(db3_scaled)

# # Get the cluster labels
# cluster_labels = st_dbscan.labels_

# # Plot the clusters
# plt.figure(figsize=(10, 6))
# unique_labels = set(cluster_labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# for label, color in zip(unique_labels, colors):
#     if label == -1:
#         # Noise points are plotted as black stars
#         noise_mask = (cluster_labels == label)
#         plt.scatter(db3_scaled[noise_mask, 0], db3_scaled[noise_mask, 1], c='k', marker='*', s=20, label='Noise')
#     else:
#         # Clustered points are plotted with different colors
#         cluster_mask = (cluster_labels == label)
#         plt.scatter(db3_scaled[cluster_mask, 0], db3_scaled[cluster_mask, 1], c=color, label=f'Cluster {label}')

# plt.title('ST-DBSCAN Clustering of Dataset db3')
# plt.xlabel('Spatial Dimension 1')
# plt.ylabel('Spatial Dimension 2')
# plt.legend()
# plt.show()


# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# # Load your dataset 'db3' into a pandas DataFrame
# # Replace 'your_data.csv' with the actual filename or path to your dataset
# df = pd.read_csv('db3.csv')

# # Assuming your dataset has features in columns 'feature1' and 'feature2', you can modify this accordingly
# X = df[['0', '1']]

# # Standardize the data if needed (VDBSCAN is distance-based)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Apply VDBSCAN clustering
# eps = 0.5  # Adjust epsilon (neighborhood radius) as needed
# min_samples = 5  # Adjust the minimum number of samples in a neighborhood as needed
# dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# clusters = dbscan.fit_predict(X_scaled)

# # Create a scatter plot to visualize the clustering
# plt.figure(figsize=(10, 8))
# plt.scatter(X['0'], X['1'], c=clusters, cmap='viridis')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('VDBSCAN Clustering for db3 Dataset')
# plt.show()

from sklearn.cluster import VariationalDBSCAN
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'db3' is your dataset, load it here
# You should have your dataset loaded into a variable, e.g., 'data'

# Create a VariationalDBSCAN instance
vdbscan = VariationalDBSCAN()

# Fit the model to your data and obtain cluster labels
cluster_labels = vdbscan.fit_predict(db3)

# Create a scatter plot to visualize the clusters
plt.scatter(db3[:, 0], db3[:, 1], c=cluster_labels, cmap='viridis')
plt.title('VariationalDBSCAN Clustering of Dataset "db3"')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')

plt.show()
