"""
Societal Challenge Theme Discovery - Hierarchical Clustering Module
Performs Agglomerative Hierarchical Clustering, linkage computation for dendrograms,
and quantitative comparison against K-Means partition clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

class HierarchicalWorkflow:
    def __init__(self, n_clusters: int = 5, linkage_method: str = 'ward', metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        # Ward linkage requires euclidean metric
        if linkage_method == 'ward':
            self.metric = 'euclidean'
        else:
            self.metric = metric

    def fit(self, X: np.ndarray):
        """Fits Agglomerative Clustering model."""
        if self.linkage_method == 'ward':
            model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        else:
            model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage_method, metric=self.metric)
            
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels, metric=self.metric)
        return {
            'model': model,
            'labels': labels,
            'silhouette': round(float(sil), 4),
            'linkage': self.linkage_method,
            'metric': self.metric,
            'n_clusters': self.n_clusters
        }

    def compute_linkage_matrix(self, X: np.ndarray):
        """Computes linkage matrix for dendrogram generation."""
        # For ward, metric must be euclidean
        m = 'euclidean' if self.linkage_method == 'ward' else self.metric
        return linkage(X, method=self.linkage_method, metric=m)

    def compare_with_kmeans(self, hier_labels: np.ndarray, kmeans_labels: np.ndarray) -> dict:
        """Computes concordance metrics between Hierarchical and K-Means partition clusters."""
        ari = adjusted_rand_score(kmeans_labels, hier_labels)
        nmi = normalized_mutual_info_score(kmeans_labels, hier_labels)
        contingency = pd.crosstab(
            pd.Series(kmeans_labels, name='KMeans_Cluster'),
            pd.Series(hier_labels, name='Hierarchical_Cluster')
        )
        return {
            'adjusted_rand_index': round(float(ari), 4),
            'normalized_mutual_info': round(float(nmi), 4),
            'contingency_table': contingency
        }

