"""
Societal Challenge Theme Discovery - Preprocessing Module
Handles variable relevance, text normalization, domain-specific stop word filtering,
numerical feature engineering, standardization, and categorical encoding.
Includes flexible column auto-detection for custom datasets.
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Domain boilerplate words commonly found in hackathon & civic problem statements
DEFAULT_DOMAIN_STOPWORDS = {
    'background', 'description', 'expected', 'solution', 'system', 'problem',
    'statement', 'objective', 'proposed', 'developed', 'using', 'based',
    'platform', 'provide', 'support', 'need', 'needs', 'require', 'required',
    'including', 'various', 'across', 'also', 'order', 'features', 'capable',
    'outcomes', 'detailed', 'currently', 'existing', 'challenges', 'presently',
    'aims', 'seeks', 'must', 'participants', 'should', 'would', 'could', 'team',
    'teams', 'develop', 'development', 'design', 'india', 'indian', 'real', 'time'
}

STANDARD_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren'
}

def clean_text(text: str, remove_domain_stopwords: bool = True) -> str:
    """Cleans raw text: lowercases, removes urls, special chars, numbers, and stopwords."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 3. Remove punctuation and non-alpha characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 4. Tokenize by whitespace
    tokens = text.split()
    
    # 5. Filter short tokens and stopwords
    all_stops = STANDARD_STOPWORDS.union(DEFAULT_DOMAIN_STOPWORDS if remove_domain_stopwords else set())
    cleaned_tokens = [t for t in tokens if len(t) > 2 and t not in all_stops]
    
    return " ".join(cleaned_tokens)

class ProblemDataPipeline:
    """End-to-end preprocessing pipeline for problem statements and text records."""
    def __init__(self, remove_domain_stopwords: bool = True):
        self.remove_domain_stopwords = remove_domain_stopwords
        self.scaler = StandardScaler()
        self.numerical_cols = []
        self.raw_df = None
        self.processed_df = None

    def _detect_and_standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligently maps varying column names to canonical schema."""
        col_map = {}
        cols_lower = {c.strip().lower(): c for c in df.columns}
        
        # ID column aliases
        id_aliases = ['problem statement id', 'id', 'statement id', 'issue id', 'ticket id', 'index', 'ps_id', 'task id', 'ps number', 's.no.']
        for alias in id_aliases:
            if alias in cols_lower:
                col_map[cols_lower[alias]] = 'Problem Statement ID'
                break
                
        # Title column aliases
        title_aliases = ['problem statement title', 'title', 'subject', 'headline', 'summary', 'name', 'problem title']
        for alias in title_aliases:
            if alias in cols_lower:
                col_map[cols_lower[alias]] = 'Problem Statement Title'
                break
                
        # Description column aliases
        desc_aliases = ['description', 'desc', 'text', 'content', 'body', 'problem description', 'details', 'issue description']
        for alias in desc_aliases:
            if alias in cols_lower:
                col_map[cols_lower[alias]] = 'Description'
                break

        # Organization column aliases
        org_aliases = ['organization', 'org', 'ministry', 'agency', 'company', 'source', 'department']
        for alias in org_aliases:
            if alias in cols_lower and cols_lower[alias] not in col_map:
                col_map[cols_lower[alias]] = 'Organization'
                break

        # Category column aliases (Software / Hardware / Project Type)
        cat_aliases = ['category', 'project type', 'tech category', 'mode', 'type']
        for alias in cat_aliases:
            if alias in cols_lower and cols_lower[alias] not in col_map:
                col_map[cols_lower[alias]] = 'Category'
                break

        # Theme column aliases (Societal Theme / Domain Bucket / Sector)
        theme_aliases = ['domain bucket', 'theme', 'domain', 'sector', 'topic', 'area', 'bucket', 'label']
        for alias in theme_aliases:
            if alias in cols_lower and cols_lower[alias] not in col_map:
                col_map[cols_lower[alias]] = 'Theme'
                break

        df = df.rename(columns=col_map)
        
        # Ensure mandatory canonical columns exist
        if 'Problem Statement ID' not in df.columns:
            df['Problem Statement ID'] = np.arange(1, len(df) + 1)
        if 'Problem Statement Title' not in df.columns:
            if 'Description' in df.columns:
                df['Problem Statement Title'] = df['Description'].apply(lambda x: " ".join(str(x).split()[:7]) + "..." if pd.notna(x) else "Untitled Statement")
            else:
                df['Problem Statement Title'] = "Problem Statement " + df['Problem Statement ID'].astype(str)
        if 'Description' not in df.columns:
            df['Description'] = df['Problem Statement Title']
        if 'Organization' not in df.columns:
            df['Organization'] = 'General / Crowdsourced'
        if 'Department' not in df.columns:
            df['Department'] = df['Organization']
        if 'Category' not in df.columns:
            df['Category'] = 'Software'
        if 'Theme' not in df.columns:
            df['Theme'] = 'General'
            
        return df

    def load_and_audit(self, filepath_or_buffer) -> pd.DataFrame:
        """Loads data, evaluates variable relevance, and handles missing values."""
        if isinstance(filepath_or_buffer, pd.DataFrame):
            df = filepath_or_buffer.copy()
        else:
            if hasattr(filepath_or_buffer, 'seek'):
                try:
                    filepath_or_buffer.seek(0)
                except Exception:
                    pass
            df = pd.read_csv(filepath_or_buffer)
        self.raw_df = df.copy()
        
        # Clean column whitespace and detect standard columns
        df.columns = [c.strip() for c in df.columns]
        df = self._detect_and_standardize_columns(df)
        
        # Fill missing values gracefully
        df['Problem Statement Title'] = df['Problem Statement Title'].fillna('')
        df['Description'] = df['Description'].fillna('')
        df['Theme'] = df['Theme'].fillna('Uncategorized')
        df['Category'] = df['Category'].fillna('Unknown')
        df['Organization'] = df['Organization'].fillna('Unknown')
        df['Department'] = df['Department'].fillna('Unknown')
            
        # Drop columns with extreme missingness (>80%) that offer no clustering signal
        sparse_cols = ['Youtube Link', 'Dataset Link', 'Contact info']
        existing_sparse = [c for c in sparse_cols if c in df.columns]
        if existing_sparse:
            df = df.drop(columns=existing_sparse)
        
        # Combine Title and Description for rich semantic representation
        if df['Description'].equals(df['Problem Statement Title']):
            df['Combined_Raw_Text'] = df['Problem Statement Title'].astype(str)
        else:
            df['Combined_Raw_Text'] = df['Problem Statement Title'].astype(str) + ". " + df['Description'].astype(str)
        
        # Feature Engineering: Numerical properties
        df['title_word_count'] = df['Problem Statement Title'].astype(str).apply(lambda x: len(x.split()))
        df['desc_word_count'] = df['Description'].astype(str).apply(lambda x: len(x.split()))
        df['char_count'] = df['Combined_Raw_Text'].apply(len)
        df['avg_word_length'] = df['Combined_Raw_Text'].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # Clean text
        df['Cleaned_Text'] = df['Combined_Raw_Text'].apply(
            lambda t: clean_text(t, remove_domain_stopwords=self.remove_domain_stopwords)
        )
        
        # Category binary encoding (Software=1, Hardware=0)
        df['is_software'] = df['Category'].apply(lambda x: 1 if str(x).lower().strip() == 'software' else 0)
        
        # Numerical feature scaling
        self.numerical_cols = ['title_word_count', 'desc_word_count', 'char_count', 'avg_word_length']
        scaled_features = self.scaler.fit_transform(df[self.numerical_cols])
        for i, col in enumerate(self.numerical_cols):
            df[f'{col}_scaled'] = scaled_features[:, i]
            
        self.processed_df = df
        return df

    def get_preprocessing_summary(self) -> dict:
        """Returns structured audit information for documentation."""
        if self.processed_df is None:
            return {}
        return {
            'total_records': len(self.processed_df),
            'raw_columns': list(self.raw_df.columns),
            'retained_columns': list(self.processed_df.columns),
            'dropped_columns': [c for c in self.raw_df.columns if c not in self.processed_df.columns],
            'categories': self.processed_df['Category'].value_counts().to_dict() if 'Category' in self.processed_df else {},
            'organizations_count': self.processed_df['Organization'].nunique() if 'Organization' in self.processed_df else 0,
            'avg_raw_words': round(float(self.processed_df['desc_word_count'].mean()), 1),
            'avg_cleaned_words': round(float(self.processed_df['Cleaned_Text'].apply(lambda x: len(x.split())).mean()), 1)
        }
