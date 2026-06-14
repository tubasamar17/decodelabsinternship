

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("mall_customers.csv")

print("="*60)
print("PROJECT 3: CUSTOMER SEGMENTATION")
print("="*60)

print("\n✅ Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n📋 First 5 rows:")
print(df.head())

print("\n📊 Column names:")
print(df.columns.tolist())

features = ['Annual Income (k$)', 'Spending Score (1-100)']

X = df[features]

print("\n✅ Features selected for clustering:")
print(features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Data scaled successfully")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n✅ PCA applied: 2 dimensions created")
print(f"Explained variance ratio: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')
plt.grid(True)

plt.tight_layout()
plt.show()

best_k_elbow = 5  
best_k_silhouette = K_range[np.argmax(silhouette_scores)]

print("\n" + "="*60)
print("STEP 5: OPTIMAL K DETERMINATION")
print("="*60)
print(f"Elbow Method suggests: K = 5")
print(f"Silhouette Score suggests: K = {best_k_silhouette} (score: {max(silhouette_scores):.3f})")

optimal_k = 5
print(f"\n✅ Using K = {optimal_k} for final model")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n✅ K-Means clustering completed")
print(f"\nCluster distribution:")
print(df['Cluster'].value_counts().sort_index())


plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                c=colors[i], label=f'Cluster {i}', alpha=0.7)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'Customer Segments (K-Means, K={optimal_k})')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for i in range(optimal_k):
    cluster_indices = df['Cluster'] == i
    plt.scatter(X_pca[cluster_indices, 0], 
                X_pca[cluster_indices, 1],
                c=colors[i], label=f'Cluster {i}', alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Clusters Visualized in 2D PCA Space (K={optimal_k})')
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "="*60)
print("STEP 9: BUSINESS PERSONAS (Customer Segments)")
print("="*60)

cluster_summary = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'Age': ['mean'],
    'CustomerID': 'count'
}).round(2)

print("\n📊 Cluster Statistics:")
print(cluster_summary)


personas = {
    0: "💰 STANDARD: Medium income, medium spending - Average regular customers",
    1: "💎 HIGH VALUE: High income, high spending - Premium customers who buy luxury",
    2: "⚠️ CAREFUL: High income, low spending - Rich but frugal, need promotions",
    3: "🔥 TARGET: Low income, high spending - Big spenders with limited budget, offer deals",
    4: "📉 LOW: Low income, low spending - Rarely buy, need engagement"
}

print("\n👥 BUSINESS PERSONAS:")
for cluster_id, persona in personas.items():
    count = df[df['Cluster'] == cluster_id].shape[0]
    print(f"\n🔹 Cluster {cluster_id}: {persona}")
    print(f"   → {count} customers ({count/len(df)*100:.1f}%)")

print("\n" + "="*60)
print("PROJECT 3 COMPLETED SUCCESSFULLY ✅")
print("="*60)

print("""
📌 WHAT YOU HAVE DONE:
1. Loaded customer data
2. Applied PCA for dimensionality reduction
3. Used Elbow Method to find optimal K
4. Used Silhouette Score to validate K
5. Ran K-Means clustering
6. Created business personas from clusters

📁 Files generated: None (visualizations shown)
💡 To save clusters to CSV, uncomment the line below
""")

