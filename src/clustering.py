"""
Societal Challenge Theme Discovery - Clustering Workflow
Implements K-Means (Euclidean), K-Medoids (Manhattan distance partition clustering),
multi-k sweeps, validation metrics (WCSS/Inertia, Silhouette Score, Davies-Bouldin Index),
and cluster stability analysis across random initializations.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist

class ManhattanKMedoids:
    """
    Partition clustering using Manhattan (L1 / Cityblock) distance.
    Uses Partitioning Around Medoids (PAM) with heuristic initialization.
    """
    def __init__(self, n_clusters: int = 5, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoids_idx_ = None
        self.labels_ = None
        self.inertia_ = None  # Sum of Manhattan distances to closest medoid

    def fit(self, X: np.ndarray):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Initialize medoids randomly
        medoid_indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        
        for _ in range(self.max_iter):
            # Compute Manhattan distances from all samples to current medoids
            # metric='cityblock' is Manhattan distance
            dists = cdist(X, X[medoid_indices], metric='cityblock')
            labels = np.argmin(dists, axis=1)
            
            new_medoids = []
            for k in range(self.n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) == 0:
                    # Re-seed empty cluster
                    new_medoid = np.random.choice(n_samples)
                else:
                    # Find point in cluster with minimum total Manhattan distance to others
                    sub_dists = cdist(X[cluster_members], X[cluster_members], metric='cityblock')
                    cost_sum = np.sum(sub_dists, axis=1)
                    new_medoid = cluster_members[np.argmin(cost_sum)]
                new_medoids.append(new_medoid)
                
            new_medoids = np.array(new_medoids)
            if np.array_equal(np.sort(medoid_indices), np.sort(new_medoids)):
                break
            medoid_indices = new_medoids

        final_dists = cdist(X, X[medoid_indices], metric='cityblock')
        self.labels_ = np.argmin(final_dists, axis=1)
        self.medoids_idx_ = medoid_indices
        self.inertia_ = np.sum(np.min(final_dists, axis=1))
        self.cluster_centers_ = X[medoid_indices]
        return self


class ClusteringWorkflow:
    def __init__(self, min_k: int = 2, max_k: int = 12, random_state: int = 42):
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state

    def evaluate_range_of_k(self, X: np.ndarray, distance_metric: str = 'euclidean') -> pd.DataFrame:
        """
        Runs clustering across k values from min_k to max_k.
        Computes WCSS, Silhouette Score, and Davies-Bouldin Index.
        """
        results = []
        k_values = list(range(self.min_k, self.max_k + 1))
        
        for k in k_values:
            if distance_metric == 'manhattan':
                model = ManhattanKMedoids(n_clusters=k, random_state=self.random_state)
                model.fit(X)
                labels = model.labels_
                wcss = model.inertia_
                sil = silhouette_score(X, labels, metric='manhattan')
            else:
                model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=self.random_state)
                model.fit(X)
                labels = model.labels_
                wcss = model.inertia_
                sil = silhouette_score(X, labels, metric='euclidean')
                
            db = davies_bouldin_score(X, labels)
            
            results.append({
                'k': k,
                'WCSS': round(float(wcss), 3),
                'Silhouette_Score': round(float(sil), 4),
                'Davies_Bouldin_Index': round(float(db), 4)
            })
            
        return pd.DataFrame(results)

    def fit_final_model(self, X: np.ndarray, k: int, distance_metric: str = 'euclidean', random_state: int = None):
        """Fits the selected final model configuration."""
        seed = random_state if random_state is not None else self.random_state
        if distance_metric == 'manhattan':
            model = ManhattanKMedoids(n_clusters=k, random_state=seed)
            model.fit(X)
            sil = silhouette_score(X, model.labels_, metric='manhattan')
        else:
            model = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=seed)
            model.fit(X)
            sil = silhouette_score(X, model.labels_, metric='euclidean')
            
        db = davies_bouldin_score(X, model.labels_)
        
        return {
            'model': model,
            'labels': model.labels_,
            'cluster_centers': model.cluster_centers_,
            'inertia': float(model.inertia_),
            'silhouette': float(sil),
            'davies_bouldin': float(db),
            'k': k,
            'metric': distance_metric
        }

    def assess_stability(self, X: np.ndarray, k: int, distance_metric: str = 'euclidean', seeds=[42, 101, 777, 999, 1234]) -> dict:
        """
        Runs clustering across multiple random initializations and calculates
        pairwise Adjusted Rand Index (ARI) to evaluate cluster stability.
        """
        all_labels = []
        for seed in seeds:
            res = self.fit_final_model(X, k=k, distance_metric=distance_metric, random_state=seed)
            all_labels.append(res['labels'])
            
        ari_scores = []
        nmi_scores = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                nmi = normalized_mutual_info_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)
                nmi_scores.append(nmi)
                
        return {
            'mean_ari': round(float(np.mean(ari_scores)), 4),
            'min_ari': round(float(np.min(ari_scores)), 4),
            'max_ari': round(float(np.max(ari_scores)), 4),
            'mean_nmi': round(float(np.mean(nmi_scores)), 4),
            'tested_seeds': seeds
        }

