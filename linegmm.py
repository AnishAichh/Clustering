import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score

# Load data from a CSV file
data = pd.read_csv('line.csv') 

# specifying two featured columns 0 and 1 for line csv 

selected_columns = ['0', '1']
X = data[selected_columns]

# Creating a Gaussian Mixture Model
n_components = 3 
gmm = GaussianMixture(n_components=n_components, random_state=0)

# Fiting the model to the data
gmm.fit(X)

# Predicting cluster labels for each data point
labels = gmm.predict(X)

# Adding cluster labels to your original DataFrame
data['Cluster'] = labels

# Visualizing the clusters 2D data r karone
if len(selected_columns) == 2:
    plt.scatter(data[selected_columns[0]], data[selected_columns[1]], c=labels, cmap='viridis')
    plt.xlabel(selected_columns[0])
    plt.ylabel(selected_columns[1])
    plt.title('GMM Clustering on line dataset')
    plt.show()

bic_scores = []
aic_scores = []
n_clusters_range = range(2, 11)
for n_clusters in n_clusters_range:
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))
# plotting BIC AND AIC Scores
plt.plot(n_clusters_range, bic_scores, marker='o', label='BIC')
plt.plot(n_clusters_range, aic_scores, marker='o', label='AIC')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC/AIC Scores')
plt.legend()
plt.show()


