# 1. Dataset Loading and Initial Exploration

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 2. Data Exploration and Feature Selection

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Select features for clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

print("\nSelected Features:")
print(X.head())

print("\nSelected Features Shape:")
print(X.shape)

# 3. Feature Scaling

from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the selected features
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nScaled Features:")
print(X_scaled.head())

print("\nScaled Data Summary:")
print(X_scaled.describe())

# 4. Finding Optimal Number of Clusters - Elbow Method

from sklearn.cluster import KMeans

inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(range(2, 11), inertia, marker="o")

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))

plt.show()

# 5. Silhouette Score Analysis

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))

plt.plot(range(2, 11), silhouette_scores, marker="o")

plt.title("Silhouette Score for Different Numbers of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))

plt.show()

print("\nSilhouette Scores:")
for k, score in zip(range(2, 11), silhouette_scores):
    print(f"K = {k}: {score:.4f}")

# 6. Apply K-Means Clustering

optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Assignment:")
print(df[["CustomerID", "Age", "Annual Income (k$)",
          "Spending Score (1-100)", "Cluster"]].head(10))

print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# 7. Cluster Visualization

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_scaled["Age"],
    X_scaled["Annual Income (k$)"],
    X_scaled["Spending Score (1-100)"],
    c=df["Cluster"],
    s=60
)

ax.set_title("Customer Segments using K-Means Clustering")
ax.set_xlabel("Age (Scaled)")
ax.set_ylabel("Annual Income (Scaled)")
ax.set_zlabel("Spending Score (Scaled)")

plt.legend(*scatter.legend_elements(), title="Cluster")

plt.show()

# 8. Cluster Characteristics

cluster_summary = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print("\nCluster Characteristics:")
print(cluster_summary.round(2))

# 9. Cluster-wise Statistical Analysis

cluster_stats = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].agg(["mean", "min", "max", "count"])

print("\nDetailed Cluster Statistics:")
print(cluster_stats.round(2))

# 10. Cluster Interpretation

print("\nCluster-wise Mean Values:")

for cluster in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster]

    print(f"\nCluster {cluster}:")
    print(f"Average Age: {cluster_data['Age'].mean():.2f}")
    print(f"Average Annual Income: {cluster_data['Annual Income (k$)'].mean():.2f}")
    print(f"Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.2f}")
    print(f"Number of Customers: {len(cluster_data)}")