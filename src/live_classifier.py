"""
Societal Challenge Theme Discovery - Real-Time Live AI Classifier
Provides on-the-fly thematic classification, confidence scoring,
nearest exemplar matching, and live 2D/3D PCA projection for new challenge submissions.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import clean_text

class LiveChallengeClassifier:
    """Real-time inference engine that maps individual challenge statements to discovered themes."""
    def __init__(
        self,
        vectorizer,
        cluster_centers: np.ndarray,
        profiles: dict,
        pca_model,
        X_existing: np.ndarray,
        df_existing: pd.DataFrame,
        distance_metric: str = 'euclidean',
        remove_domain_stopwords: bool = True
    ):
        self.vectorizer = vectorizer
        self.cluster_centers = cluster_centers
        self.profiles = profiles
        self.pca_model = pca_model
        self.X_existing = X_existing
        self.df_existing = df_existing.reset_index(drop=True)
        self.distance_metric = distance_metric
        self.remove_domain_stopwords = remove_domain_stopwords
        self.feature_names = np.array(vectorizer.get_feature_names())

    def classify(self, title: str, description: str, top_k_similar: int = 3) -> dict:
        """
        Performs real-time classification on a single problem statement.
        Returns:
            - assigned_cluster_id
            - theme_title
            - theme_icon
            - suggested_theme
            - confidence_score (%)
            - cluster_probabilities
            - matching_keywords
            - nearest_national_challenges
            - pca_coords (2D and 3D)
        """
        raw_combined = f"{title.strip()}. {description.strip()}"
        cleaned = clean_text(raw_combined, remove_domain_stopwords=self.remove_domain_stopwords)
        
        # If input has no usable words
        if not cleaned.strip():
            cleaned = "technology development innovation system"

        # Transform to TF-IDF feature space
        query_vec = self.vectorizer.transform([cleaned])
        query_dense = query_vec.toarray() if hasattr(query_vec, 'toarray') else np.asarray(query_vec)

        # 1. Distances & Similarities to Centroids
        if self.distance_metric == 'manhattan':
            dists = cdist(query_dense, self.cluster_centers, metric='cityblock').flatten()
        else:
            dists = cdist(query_dense, self.cluster_centers, metric='euclidean').flatten()

        sims = cosine_similarity(query_dense, self.cluster_centers).flatten()
        
        # Best matching cluster
        best_cluster_id = int(np.argmin(dists))
        best_profile = self.profiles[best_cluster_id]

        # Softmax-like normalized confidence score
        # Use negative distance normalized
        inv_dists = np.exp(-dists / (np.std(dists) + 1e-6))
        probs = (inv_dists / np.sum(inv_dists)) * 100.0
        confidence_score = round(float(probs[best_cluster_id]), 1)
        # Fallback if distances are very close
        if confidence_score < 25.0:
            cosine_pct = max(0.0, float(sims[best_cluster_id])) * 100.0
            confidence_score = round(max(confidence_score, min(95.0, cosine_pct + 15.0)), 1)

        # 2. Extract Matching Characteristic Keywords
        query_nonzero_idx = np.nonzero(query_dense)[1]
        query_terms_weights = {self.feature_names[i]: float(query_dense[0, i]) for i in query_nonzero_idx}
        
        centroid_terms = {t['term'].lower(): t['weight'] for t in best_profile.get('top_features', [])}
        
        matched_keywords = []
        for term, q_w in sorted(query_terms_weights.items(), key=lambda x: x[1], reverse=True):
            if term.lower() in centroid_terms:
                matched_keywords.append({
                    'term': term,
                    'query_weight': round(q_w, 4),
                    'theme_relevance': round(centroid_terms[term.lower()], 4)
                })

        # If sparse overlap, take the highest weighted query terms
        if len(matched_keywords) < 3:
            for term, q_w in sorted(query_terms_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                if not any(m['term'] == term for m in matched_keywords):
                    matched_keywords.append({
                        'term': term,
                        'query_weight': round(q_w, 4),
                        'theme_relevance': 0.05
                    })

        # 3. Find Top-K Nearest Existing Problem Statements in the Database
        doc_sims = cosine_similarity(query_dense, self.X_existing).flatten()
        top_doc_indices = np.argsort(doc_sims)[-top_k_similar:][::-1]
        
        nearest_challenges = []
        for idx in top_doc_indices:
            row = self.df_existing.iloc[idx]
            sim_pct = round(float(doc_sims[idx]) * 100, 1)
            nearest_challenges.append({
                'ps_id': row.get('Problem Statement ID', f"PS-{idx+1}"),
                'title': row.get('Problem Statement Title', 'Untitled'),
                'organization': row.get('Organization', 'Unknown'),
                'cluster_assigned': row.get('Theme_Title', f"Theme {row.get('Cluster_ID', 0) + 1}"),
                'similarity_pct': max(0.0, sim_pct)
            })

        # 4. Project into 2D and 3D PCA coordinates
        try:
            pca_proj = self.pca_model.pca.transform(query_dense)
            pca_x = float(pca_proj[0, 0])
            pca_y = float(pca_proj[0, 1])
            pca_z = float(pca_proj[0, 2]) if pca_proj.shape[1] >= 3 else 0.0
        except Exception:
            pca_x, pca_y, pca_z = 0.0, 0.0, 0.0

        all_cluster_breakdown = []
        for k in range(len(self.cluster_centers)):
            all_cluster_breakdown.append({
                'cluster_id': k,
                'theme_title': self.profiles[k]['theme_title'],
                'icon': self.profiles[k].get('icon', '🌐'),
                'probability_pct': round(float(probs[k]), 1),
                'distance': round(float(dists[k]), 4),
                'cosine_similarity': round(float(sims[k]), 4)
            })
        all_cluster_breakdown.sort(key=lambda x: x['probability_pct'], reverse=True)

        return {
            'cleaned_text': cleaned,
            'assigned_cluster_id': best_cluster_id,
            'theme_title': best_profile['theme_title'],
            'theme_icon': best_profile.get('icon', '🌐'),
            'suggested_theme': best_profile['suggested_theme'],
            'confidence_score': confidence_score,
            'cluster_breakdown': all_cluster_breakdown,
            'matched_keywords': matched_keywords[:6],
            'nearest_challenges': nearest_challenges,
            'pca_point': {
                'x': pca_x,
                'y': pca_y,
                'z': pca_z
            }
        }
