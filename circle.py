# import scipy.io
# import pandas as pd

# # Load the .mat file
# mat = scipy.io.loadmat('circle.mat')

# # Replace 'your_data.mat' with the actual file path.

# # Convert the loaded data to a DataFrame (assuming it's a dictionary with keys)
# data_dict = mat['circle']  # Replace 'your_variable_name' with the actual variable name in the .mat file
# df = pd.DataFrame(data_dict)

# # Save the DataFrame to a CSV file
# df.to_csv('output_data.csv', index=False)
# # Replace 'output_data.csv' with the desired output file path.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load your dataset 'db3' into a pandas DataFrame
# Replace 'your_data.csv' with the actual filename or path to your dataset
df = pd.read_csv('circle.csv')

# Assuming your dataset has features in columns 'feature1' and 'feature2', you can modify this accordingly
X = df[['0', '1']]

# Standardize the data if needed (VDBSCAN is distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply VDBSCAN clustering
eps = 0.5  # Adjust epsilon (neighborhood radius) as needed
min_samples = 5  # Adjust the minimum number of samples in a neighborhood as needed
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

# Create a scatter plot to visualize the clustering
plt.figure(figsize=(10, 8))
plt.scatter(X['0'], X['1'], c=clusters, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('VDBSCAN Clustering for db3 Dataset')
plt.show()