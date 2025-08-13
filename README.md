
# 🎵 Spotify Genre Clustering & Recommendation System

An **unsupervised machine learning project** that clusters Spotify tracks based on audio features, names the clusters using dominant genres, and provides a recommendation function for similar songs.

---

## 📌 Project Overview
This project:
- Loads and cleans the Spotify dataset.
- Performs feature scaling and dimensionality reduction (PCA).
- Finds the optimal number of clusters using the Elbow Method.
- Applies **KMeans Clustering** to group songs.
- Assigns **human-readable cluster names** based on dominant genres.
- Visualizes clusters using Plotly.
- Implements a recommendation system to suggest similar tracks.
- Calculates silhouette scores for evaluating clustering performance.

---

## 🛠 Technologies Used
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn

---

## 📂 Dataset
The dataset used is:
```
spotify dataset.csv
```
It must contain:
- **playlist_genre** (string) — used to label clusters
- Audio feature columns (numeric) — used for clustering
- Optional: track details like `track_name`, `artist_name`, `album_name`

---


## 🎯 Features
- 📊 **Cluster Visualization** — Interactive scatter plot using Plotly.
- 🎼 **Cluster Naming** — Human-friendly labels based on genre.
- 🎧 **Recommendation System** — Suggests similar songs.
- 📈 **Silhouette Score Calculation** — Measures clustering quality.
- 💾 **Exportable Results** — Saves final dataset with cluster info.

---

## 📝 Example Output


📀 Rock mix Lovers Top Genres:
- pop         320
- dance pop   210
- indie pop    50

📀 Rap mix Lovers Top Genres:
- rock        280
- alt rock    140
- punk rock    60

---
