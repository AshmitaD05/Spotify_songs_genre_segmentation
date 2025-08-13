
# STEP 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# STEP 2: Load Dataset
df = pd.read_csv("/content/spotify dataset.csv")
print("Initial Data Sample:")
print(df.head())

# STEP 3: Data Cleaning & Preprocessing
print("\nShape before drop duplicates:", df.shape)
df = df.drop_duplicates()
df = df.dropna()
print("Shape after cleaning:", df.shape)

# Drop non-numeric/unnecessary columns for clustering
drop_cols = ['track_id', 'track_name', 'artist_name', 'album_name', 'playlist_id']
df_clustering = df.drop(columns=drop_cols, errors='ignore')

# Display datatypes
print("\nData types of clustering features:")
print(df_clustering.dtypes)

# STEP 4: Correlation Matrix
numeric_cols = df_clustering.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(15, 10))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# STEP 5: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clustering.select_dtypes(include=np.number))

# STEP 6: PCA for Dimensionality Reduction
pca = PCA(n_components=10)
pca_result = pca.fit_transform(scaled_data)

# STEP 7: Elbow Method to Find Optimal Clusters
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.plot(range(2, 11), sse, marker='o')
plt.title("Elbow Method For Optimal K")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.grid(True)
plt.show()

# STEP 8: KMeans Clustering (Final Model)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add PCA components and cluster labels
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]
df['cluster'] = clusters

print("\nFinal DataFrame Shape:", df.shape)
print("Columns:", df.columns.tolist())

# STEP 9: Assign Human-Readable Cluster Names
cluster_names = {}
for c in sorted(df['cluster'].unique()):
    top_genre = df[df['cluster'] == c]['playlist_genre'].value_counts().idxmax()
    cluster_names[c] = f"{top_genre.capitalize()} Lovers"

df['cluster_name'] = df['cluster'].map(cluster_names)

#  STEP 10: Visualization with Plotly
fig = px.scatter(
    df,
    x='pca1',
    y='pca2',
    color='cluster_name',
    hover_data=['track_name', 'artist_name']
)
fig.update_layout(title="Spotify Song Clusters by Genre", title_x=0.5)
fig.show()

# STEP 11: Cluster Summary
cluster_summary = df.groupby('cluster_name').mean(numeric_only=True)
print("\nCluster Summary (Mean Feature Values):")
print(cluster_summary)

# Dominant genres in each cluster
for c in sorted(df['cluster'].unique()):
    print(f"\n {cluster_names[c]} (Cluster {c}) Top Genres:")
    print(df[df['cluster'] == c]['playlist_genre'].value_counts().head(5))


# STEP 12: Song Recommendation Function
def recommend_similar(song_name, df, top_n=5):
    """
    Recommend similar songs based on cluster membership.
    """
    if song_name not in df['track_name'].values:
        return f"Song '{song_name}' not found in dataset."
    cluster_label = df[df['track_name'] == song_name]['cluster'].values[0]
    similar_songs = df[df['cluster'] == cluster_label]
    recommendations = similar_songs[similar_songs['track_name'] != song_name].sample(top_n)
    return recommendations[['track_name', 'artist_name', 'playlist_genre']]
# Example usage
print("\n Recommendations for 'Shape of You':")
print(recommend_similar("Shape of You", df, top_n=5))

#  STEP 13: Silhouette Score Check
for k in range(2, 11):
    pca_best = PCA(n_components=2)
    pca_data = pca_best.fit_transform(scaled_data)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pca_data)
    score = silhouette_score(pca_data, labels)
    print(f"PCA=2, k={k} → Silhouette: {score:.4f}")

# STEP 14: Save Final Data
df.to_csv("spotify_clustered_final.csv", index=False)
print("\nFinal clustered data saved as 'spotify_clustered_final.csv'")