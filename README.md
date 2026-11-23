# Societal Challenge Theme Discovery from Crowdsourced Problem Statements

An end-to-end unsupervised machine learning and natural language processing (NLP) web platform designed to discover latent themes and clusters from crowdsourced real-world problem statements (such as Smart India Hackathon and civic innovation challenges).

---

## 🌟 Key Features

1. **Intelligent Text Normalization & Preprocessing**:
   - Automated handling of missing metadata and evaluation of variable relevance.
   - Filtering of standard stopwords and domain boilerplate noise (e.g., *'system'*, *'proposed'*, *'developed'*, *'background'*).
   - Numerical feature engineering (word count, character density, readability) with standard scaling.

2. **Dual-Distance Partition Clustering**:
   - **Euclidean Distance ($L_2$):** Standard K-Means with k-means++ initialization on normalized TF-IDF representations.
   - **Manhattan Distance ($L_1$):** Partitioning Around Medoids (K-Medoids) providing robust clustering against vocabulary outliers.
   - Multi-$k$ parameter sweep ($k = 2$ to $12$).

3. **Rigorous Quantitative Validation**:
   - **WCSS (Within-Cluster Sum of Squares) / Inertia:** Elbow curve analysis for diminishing returns.
   - **Silhouette Score:** Evaluates cluster cohesion vs. separation.
   - **Davies-Bouldin Index:** Identifies optimal cluster compactness and distinctness.
   - **Stability Testing:** Pairwise Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) across multiple random initializations.

4. **Hierarchical Comparison**:
   - Agglomerative Hierarchical Clustering (Ward, Complete, Average linkage).
   - Interactive Dendrogram showing bottom-up cluster taxonomy.
   - Cross-tabulation contingency table comparing partition clusters against hierarchical clusters.

5. **2D & 3D PCA Visualizer**:
   - Dimensionality reduction to 2 and 3 principal components.
   - Projected cluster centroids and interactive point inspection (Title, Organization, Pre-existing Theme, Cluster ID).
   - PCA Component Loadings explaining which keywords drive the variance axes.

6. **Thematic Profiling & Exemplar Discovery**:
   - Automated theme name generation from top centroid TF-IDF terms.
   - Top-contributing keyword bar charts per cluster.
   - Extraction of representative exemplars (statements closest to the cluster centroid).

7. **Responsible AI, Bias & Ethics Audit**:
   - Institutional concentration analysis (e.g., government ministry distribution).
   - Hardware vs. Software category skew analysis.
   - Ethical disclaimers ensuring exploratory clusters are treated as discovery lenses rather than permanent dogmatic categories.

8. **Export Studio**:
   - Download the clustered dataset with attached cluster labels and PCA coordinates as CSV.
   - Download a comprehensive executive Markdown report.

---

## 🚀 How to Run the Web Application

### Option A: One-Click Launcher (Windows)
Simply double-click the `run_app.bat` file inside this folder.

### Option B: Terminal Command
Run the following command from this directory:
```powershell
.\.venv\Scripts\streamlit run app.py
```
Or with Python:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Project Architecture

```
societal_challenge_theme_discovery/
│
├── data/
│   └── problem_statements.csv      # Crowdsourced Problem Statements dataset (231 entries)
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py             # Data auditing, cleaning, and numerical scaling
│   ├── vectorizer.py               # TF-IDF vectorizer (unigrams + bigrams)
│   ├── clustering.py               # K-Means, Manhattan K-Medoids, validation metrics, stability
│   ├── hierarchical.py             # Agglomerative clustering, dendrograms, contingency
│   ├── dim_reduction.py            # 2D/3D PCA & loading vectors
│   └── profiler.py                 # Centroid top features, exemplar discovery, bias audit
│
├── app.py                          # Full-featured Streamlit interactive web application
├── run_app.bat                     # Windows one-click launcher
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

