import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

print(df.head())
print(df.info())

# on basis of z score
numeric_df = df.select_dtypes(include='number')

# Calculate Z-scores
z_scores = zscore(numeric_df)

# Define threshold (commonly 3)
outliers = (abs(z_scores) > 3).any(axis=1)

# Display results
print("Total outliers found using Z-score:", outliers.sum())
print("Outlier rows:\n", df[outliers])

df = df[["SepalLengthCm", "SepalWidthCm"]]
db = DBSCAN(eps=0.3, min_samples=10)
model = db.fit(df)

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=df, hue=model.labels_)
plt.show()

print('Outlier Are: ')
outliers = df[model.labels_ == -1]
print(len(outliers))

import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

X = df.select_dtypes(include='number')  # Only numeric columns

lof = LocalOutlierFactor(n_neighbors=20)
y_p = lof.fit_predict(X)

df['outlier'] = (y_p == -1)

print("LOF-Based Outliers Found:", df['outlier'].sum())

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='outlier', data=df, palette='Set2')
plt.title("LOF-Based Outliers")
plt.show()

# # Fit Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(X)
distances, ind = nbrs.kneighbors(X)

k_distances = distances[:, -1]
threshold = np.percentile(k_distances, 95)
outliers_knn = k_distances > threshold
print("KNN-Based Outliers Found:", outliers_knn.sum())
df['outlier'] = (k_distances > threshold)
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='outlier', data=df, palette='Set1')
plt.title("KNN-Based Outliers")
plt.show()
