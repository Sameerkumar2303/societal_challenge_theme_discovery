# Societal Challenge Theme Discovery from Crowdsourced Problem Statements

An end-to-end unsupervised machine learning, natural language processing (NLP), and dynamic civic intelligence platform designed to discover latent themes and clusters from crowdsourced real-world problem statements (such as Smart India Hackathon and civic innovation challenges).

---

## 🌟 Key Features

1. **Intelligent Text Normalization & Preprocessing**:
   - Automated handling of missing metadata and evaluation of variable relevance.
   - Dynamic column auto-detection supporting varied schemas (`Domain Bucket`, `Category`, `Title`, `Description`).
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
   - Real-time PCA projection for live user submissions with distinct golden marker indicators.
   - PCA Component Loadings explaining which keywords drive the variance axes.

6. **Thematic Profiling & Exemplar Discovery**:
   - Automated theme name generation from top centroid TF-IDF terms.
   - Top-contributing keyword bar charts per cluster.
   - Extraction of representative exemplars (statements closest to the cluster centroid).

7. **Dynamic SQLite Database & Live Crowdsourcing**:
   - Persistent SQLite storage (`data/societal_challenges.db`) tracking all national challenges and citizen submissions.
   - Dynamic CRUD operations with on-the-fly re-clustering and synchronization.

8. **Real-Time Live AI Classifier**:
   - Submit any new problem statement and receive instant thematic prediction, match confidence, key linguistic terms, and top 3 nearest national challenges.
   - Real-time PCA projection plotting live submissions as glowing markers into the 2D/3D galaxy.

9. **RESTful API Backend (FastAPI)**:
   - Comprehensive REST endpoints (`/api/classify`, `/api/problems`, `/api/crowdsource`, `/api/themes`, `/api/stats`).
   - Interactive Swagger API documentation available at `http://localhost:8000/docs`.

10. **Responsible AI, Bias & Ethics Audit**:
    - Institutional concentration analysis (e.g., government ministry distribution).
    - Hardware vs. Software category skew analysis.
    - Ethical disclaimers ensuring exploratory clusters are treated as discovery lenses rather than permanent dogmatic categories.

11. **Export Studio**:
    - Download the clustered dataset with attached cluster labels and PCA coordinates as CSV.
    - Download a comprehensive executive Markdown report.

---

## 🚀 How to Run the Applications

### 1. Interactive Web Application (Streamlit)
* **One-Click Launcher (Windows):** Double-click `run_app.bat`
* **Terminal Command:**
  ```bash
  streamlit run app.py
  ```
  Opens automatically at `http://localhost:8501`.

### 2. RESTful Backend Service (FastAPI)
* **One-Click Launcher (Windows):** Double-click `run_api.bat`
* **Terminal Command:**
  ```bash
  uvicorn api:app --reload --port 8000
  ```
  Interactive Swagger UI documentation: `http://127.0.0.1:8000/docs`.

---

## 📁 Project Architecture

```
societal_challenge_theme_discovery/
│
├── data/
│   ├── problem_statements.csv      # Seed crowdsourced dataset (231 national challenges)
│   └── societal_challenges.db      # Persistent SQLite dynamic database
│
├── src/
│   ├── __init__.py
│   ├── database.py                 # SQLite storage, CRUD, seeding, and live persistence
│   ├── live_classifier.py          # Real-time inference, similarity scoring, PCA projection
│   ├── preprocessor.py             # Data auditing, cleaning, and numerical scaling
│   ├── vectorizer.py               # TF-IDF vectorizer (unigrams + bigrams)
│   ├── clustering.py               # K-Means, Manhattan K-Medoids, validation metrics, stability
│   ├── hierarchical.py             # Agglomerative clustering, dendrograms, contingency
│   ├── dim_reduction.py            # 2D/3D PCA & loading vectors
│   └── profiler.py                 # Centroid top features, exemplar discovery, bias audit
│
├── api.py                          # Dynamic FastAPI REST backend service
├── app.py                          # Dynamic Streamlit web application with 8 tabs
├── run_app.bat                     # Windows one-click web launcher
├── run_api.bat                     # Windows one-click API launcher
├── requirements.txt                # Python package dependencies
└── README.md                       # Comprehensive documentation
```
