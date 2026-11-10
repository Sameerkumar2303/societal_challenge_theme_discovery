"""
Societal Challenge Theme Discovery - Dimensionality Reduction & PCA Module
Computes 2D and 3D Principal Component Analysis (PCA), explained variance ratios,
centroid projections, and axis feature loadings for interpretability.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class PCAVisualizer:
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_coords = None
        self.explained_variance_ratio = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Projects high-dimensional feature matrix into PCA space."""
        self.pca_coords = self.pca.fit_transform(X)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        return self.pca_coords

    def project_centroids(self, centroids: np.ndarray) -> np.ndarray:
        """Projects cluster centroids into the same PCA space."""
        return self.pca.transform(centroids)

    def get_explained_variance(self) -> dict:
        """Returns individual and cumulative explained variance ratios."""
        return {
            'individual': [round(float(v) * 100, 2) for v in self.explained_variance_ratio],
            'cumulative': [round(float(v) * 100, 2) for v in np.cumsum(self.explained_variance_ratio)],
            'total_retained_variance': round(float(np.sum(self.explained_variance_ratio)) * 100, 2)
        }

    def get_component_loadings(self, feature_names: list, top_n: int = 8) -> dict:
        """
        Identifies the top positive and negative contributing words/features
        for PC1, PC2, and PC3 to interpret the geometric axes.
        """
        loadings = {}
        for comp_idx in range(self.n_components):
            comp = self.pca.components_[comp_idx]
            # Top positive
            pos_idx = np.argsort(comp)[-top_n:][::-1]
            # Top negative
            neg_idx = np.argsort(comp)[:top_n]
            
            loadings[f'PC{comp_idx + 1}'] = {
                'positive': [(feature_names[i], round(float(comp[i]), 4)) for i in pos_idx],
                'negative': [(feature_names[i], round(float(comp[i]), 4)) for i in neg_idx]
            }
        return loadings

