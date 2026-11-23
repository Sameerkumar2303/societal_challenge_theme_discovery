"""
Societal Challenge Theme Discovery - Vectorization Module
Transforms cleaned problem statements into dense/sparse numerical vectors using TF-IDF.
Supports customizable n-gram ranges, min/max document frequency, and normalization.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pandas as pd

class TextVectorizer:
    def __init__(self, max_features: int = 1500, ngram_range=(1, 2), min_df: int = 2, max_df: float = 0.85, norm: str = 'l2'):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            norm=self.norm,
            sublinear_tf=True
        )
        self.feature_names = None
        self.tfidf_matrix = None

    def fit_transform(self, texts: list) -> np.ndarray:
        """Fits vectorizer on text corpus and returns dense/sparse representation."""
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix.toarray()

    def get_feature_names(self):
        return self.feature_names

    def combine_with_numerical(self, text_vectors: np.ndarray, numerical_features: np.ndarray, text_weight: float = 0.9) -> np.ndarray:
        """Combines normalized text vectors with standardized numerical metadata."""
        # Scale text features and numerical features proportionally
        norm_text = text_vectors * text_weight
        norm_num = numerical_features * (1.0 - text_weight)
        return np.hstack([norm_text, norm_num])

