import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import normalized_mutual_info_score

# Load dataset
df = sns.load_dataset("iris")
print(df.info())
X = df.drop("species", axis=1)
true_labels = df["species"]

# Step 1: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Step 2: Evaluation metrics

# 1. Inertia (sum of squared distances to cluster center)
inertia = kmeans.inertia_
print("Inertia (Within-cluster sum of squares):", inertia)

# 2. Silhouette Score (how well-separated and cohesive clusters are)
sil_score = silhouette_score(X, df['cluster'])
print("Silhouette Score:", sil_score)

# 3. Adjusted Rand Index (since we have true labels)
# Convert categorical labels to numeric
le = LabelEncoder()
true_labels_encoded = le.fit_transform(true_labels)
ari = adjusted_rand_score(true_labels_encoded, df['cluster'])
print("Adjusted Rand Index:", ari)

mis = normalized_mutual_info_score(true_labels, df['cluster'])
print("Mutual Info score is :", mis)

# Step 3: Visualize clustering
sns.scatterplot(x='sepal_length', y='sepal_width', hue='cluster', data=df, palette='Set1')
plt.title("KMeans Clustering (Sepal Length vs Sepal Width)")
plt.show()
