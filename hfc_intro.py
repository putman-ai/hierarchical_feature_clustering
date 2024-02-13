# hierarchical feature clustering
'''
A starter-block for HFC development.
'''

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# generate random features
data = pd.DataFrame(
    np.random.rand(100, 10),
    columns=[f'feature_{i}' for i in range(10)]
)

# calculate the distance matrix between features
corr_matrix = data.corr()
dist_matrix = np.sqrt((1 - corr_matrix.copy()) / 2)

# cluster the features based on their distance
linkage_matrix = linkage(pdist(dist_matrix), method='single')
clusters = fcluster(linkage_matrix, 3, criterion='maxclust')

print("Feature Clusters:")
for i in np.unique(clusters):
    print(f"Cluster {i}: {data.columns[clusters == i].tolist()}")
