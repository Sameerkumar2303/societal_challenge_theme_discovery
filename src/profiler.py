"""
Societal Challenge Theme Discovery - Profiling, Interpretation & Bias Audit Module
Extracts key feature contributions per cluster centroid, identifies representative exemplars,
compares clusters against metadata, and generates responsible AI & bias audit findings.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class ClusterProfiler:
    def __init__(self, feature_names: list):
        self.feature_names = np.array(feature_names)

    def extract_top_features_per_cluster(self, centroids: np.ndarray, top_n: int = 10) -> dict:
        """
        Extracts top N TF-IDF terms with the highest centroid weight for each cluster.
        Assigns intuitive icons, deduplicated descriptive labels, and thematic summaries.
        """
        ICON_KEYWORD_MAP = [
            ({'weather', 'satellite', 'atmospheric', 'rainfall', 'forecast', 'forecasts', 'climate', 'radar', 'space'}, "🛰️"),
            ({'farmer', 'farmers', 'rural', 'agriculture', 'agricultural', 'crop', 'crops', 'soil', 'irrigation'}, "🌾"),
            ({'defence', 'defense', 'drone', 'drones', 'surveillance', 'border', 'security', 'threat', 'military'}, "🛡️"),
            ({'health', 'healthcare', 'medical', 'patient', 'patients', 'disease', 'hospital', 'diagnosis', 'clinical'}, "🏥"),
            ({'water', 'river', 'flood', 'floods', 'disaster', 'drainage', 'waste', 'pollution', 'groundwater'}, "💧"),
            ({'land', 'records', 'governance', 'compliance', 'citizen', 'registry', 'department', 'legal', 'tax'}, "🏛️"),
            ({'railway', 'train', 'traffic', 'vehicle', 'transport', 'road', 'roads', 'fleet', 'transit'}, "🚦"),
            ({'education', 'skill', 'skills', 'training', 'learning', 'school', 'multilingual', 'teacher'}, "🎓"),
            ({'energy', 'solar', 'power', 'battery', 'electricity', 'grid', 'renewable', 'clean'}, "⚡"),
            ({'hardware', 'device', 'devices', 'sensor', 'sensors', 'iot', 'edge', 'embedded'}, "🔌"),
            ({'cyber', 'cybersecurity', 'fraud', 'blockchain', 'malware', 'network', 'authentication'}, "🔐")
        ]

        cluster_profiles = {}
        for k in range(centroids.shape[0]):
            center = centroids[k]
            top_indices = np.argsort(center)[-top_n:][::-1]
            top_terms = [
                {'term': self.feature_names[i], 'weight': round(float(center[i]), 4)}
                for i in top_indices
            ]
            
            # Deduplicate keywords for title generation (avoid "Weather & Weather Forecast")
            selected_words = []
            selected_tokens_set = set()
            for t in top_terms:
                term_tokens = set(t['term'].lower().split())
                # Only include if it adds a genuinely new concept
                if not term_tokens.issubset(selected_tokens_set):
                    selected_words.append(t['term'].title())
                    selected_tokens_set.update(term_tokens)
                if len(selected_words) >= 3:
                    break
            
            theme_title = " & ".join(selected_words) if selected_words else "Thematic Domain"
            
            # Determine appropriate thematic icon
            cluster_vocab = set()
            for t in top_terms[:6]:
                cluster_vocab.update(t['term'].lower().split())
            
            chosen_icon = "🌐"
            for kw_set, icon in ICON_KEYWORD_MAP:
                if cluster_vocab.intersection(kw_set):
                    chosen_icon = icon
                    break
            
            formatted_name = f"{chosen_icon} Theme {k + 1}: {theme_title}"
            
            cluster_profiles[k] = {
                'cluster_id': k,
                'icon': chosen_icon,
                'theme_title': theme_title,
                'suggested_theme': formatted_name,
                'top_features': top_terms
            }
        return cluster_profiles

    def find_representative_exemplars(self, X: np.ndarray, centroids: np.ndarray, labels: np.ndarray, df: pd.DataFrame, top_k_exemplars: int = 3) -> dict:
        """
        Finds the problem statements located closest to each cluster center.
        """
        exemplars = {}
        for k in range(centroids.shape[0]):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                exemplars[k] = []
                continue
            
            cluster_points = X[cluster_indices]
            center = centroids[k].reshape(1, -1)
            # Compute distances to center
            dists = cdist(cluster_points, center, metric='euclidean').flatten()
            closest_local_indices = np.argsort(dists)[:top_k_exemplars]
            closest_global_indices = cluster_indices[closest_local_indices]
            
            cluster_exemplars = []
            for g_idx in closest_global_indices:
                row = df.iloc[g_idx]
                cluster_exemplars.append({
                    'id': row.get('Problem Statement ID', 'N/A'),
                    'title': row.get('Problem Statement Title', 'N/A'),
                    'org': row.get('Organization', 'N/A'),
                    'dept': row.get('Department', 'N/A'),
                    'category': row.get('Category', 'N/A'),
                    'existing_theme': row.get('Theme', 'N/A'),
                    'distance_to_center': round(float(cdist(X[g_idx].reshape(1, -1), center)[0][0]), 4)
                })
            exemplars[k] = cluster_exemplars
        return exemplars

    def cross_tabulate_metadata(self, labels: np.ndarray, df: pd.DataFrame) -> dict:
        """
        Cross-tabulates discovered clusters against Category (HW/SW), Organization, and initial Themes.
        """
        temp_df = df.copy()
        temp_df['Discovered_Cluster'] = [f'Cluster {l + 1}' for l in labels]
        
        crosstab_category = pd.crosstab(temp_df['Discovered_Cluster'], temp_df['Category'], margins=True) if 'Category' in temp_df else None
        crosstab_theme = pd.crosstab(temp_df['Discovered_Cluster'], temp_df['Theme'], margins=True) if 'Theme' in temp_df else None
        top_orgs = temp_df['Organization'].value_counts().head(8).index
        temp_df['Top_Orgs'] = temp_df['Organization'].apply(lambda x: x if x in top_orgs else 'Other')
        crosstab_org = pd.crosstab(temp_df['Discovered_Cluster'], temp_df['Top_Orgs'], margins=True)
        
        return {
            'by_category': crosstab_category,
            'by_theme': crosstab_theme,
            'by_organization': crosstab_org
        }

    def generate_bias_and_limitations_report(self, df: pd.DataFrame, labels: np.ndarray) -> dict:
        """
        Assesses institutional representation bias, domain skew, and outlines responsible-use considerations.
        """
        total = len(df)
        org_counts = df['Organization'].value_counts()
        top_org_pct = round((org_counts.iloc[0] / total) * 100, 1)
        
        cat_counts = df['Category'].value_counts()
        sw_pct = round((cat_counts.get('Software', 0) / total) * 100, 1)
        hw_pct = round((cat_counts.get('Hardware', 0) / total) * 100, 1)
        
        # Check cluster balance
        cluster_dist = pd.Series(labels).value_counts().sort_index()
        cluster_percentages = {f'Cluster {k + 1}': round((count / total) * 100, 1) for k, count in cluster_dist.items()}
        
        findings = [
            f"Institutional Representation: '{org_counts.index[0]}' represents {top_org_pct}% of all submissions, indicating notable institutional concentration.",
            f"Category Skew: The corpus is {sw_pct}% Software and {hw_pct}% Hardware. Linguistic features inherently emphasize software/algorithmic terminology.",
            f"Cluster Sizing: Largest cluster holds {max(cluster_percentages.values())}% of submissions while smallest holds {min(cluster_percentages.values())}%.",
            "Lexical Polysemy: Words like 'monitoring', 'detection', and 'analytics' span multiple societal sectors and require contextual disambiguation.",
            "Responsible AI Guidance: These discovered themes are exploratory pattern-discovery groupings, not immutable taxonomic truths. They reflect what organizations submitted to hackathons rather than the entirety of India's societal needs."
        ]
        
        return {
            'findings': findings,
            'top_organization': org_counts.index[0],
            'top_org_percentage': top_org_pct,
            'software_percentage': sw_pct,
            'hardware_percentage': hw_pct,
            'cluster_proportions': cluster_percentages
        }

