import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score

def STING(df, eps, min_pts):
  # Create a grid of cells.
  int(cells) = []
  for i in range(len(df)):
    cell = (df.loc[i, '0'], df.loc[i, '1'])
    cells.append(cell)

  # Create a cluster for each cell.
  int(clusters) = []
  for cell in cells:
    cluster = [cell]
    clusters.append(cluster)

  # Iterate over all cells.
  for i in range(len(cells)):
    cell = cells[i]

    # Check if the cell is already in a cluster.
    if cell in clusters:
      continue

    # Check if the cell is close to any other clusters.
    for j in range(len(clusters)):
      cluster = clusters[j]
      if np.linalg.norm(cell - cluster[0]) <= eps:
        cluster.append(cell)
        break

    # If the cell is not close to any other clusters, create a new cluster for it.
    else:
      cluster = [cell]
      clusters.append(cluster)

  # Return the clusters.
  return clusters


# Load the data.
df = pd.read_csv('line.csv')

# Run STING clustering.
clusters = STING(df, eps=0.5, min_pts=10)

# Print the clusters.
for cluster in clusters:
  print(cluster)