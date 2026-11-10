"""
Societal Challenge Theme Discovery - Preprocessing Module
Handles variable relevance, text normalization, domain-specific stop word filtering,
numerical feature engineering, standardization, and categorical encoding.
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Domain boilerplate words commonly found in hackathon problem statements
DEFAULT_DOMAIN_STOPWORDS = {
    'background', 'description', 'expected', 'solution', 'solutions', 'system', 'systems',
    'problem', 'problems', 'statement', 'statements', 'objective', 'objectives', 'proposed',
    'developed', 'using', 'based', 'platform', 'provide', 'provides', 'support', 'supports',
    'need', 'needs', 'require', 'requires', 'required', 'requirement', 'requirements',
    'including', 'various', 'across', 'also', 'order', 'features', 'capable', 'capability',
    'outcomes', 'detailed', 'currently', 'existing', 'challenges', 'presently', 'aims',
    'seeks', 'must', 'participants', 'should', 'would', 'could', 'team', 'teams',
    'develop', 'develops', 'development', 'design', 'designing', 'india', 'indian',
    'real', 'time', 'student', 'students', 'innovation', 'innovative', 'data', 'information',
    'model', 'models', 'high', 'low', 'conditions', 'source', 'sources', 'management',
    'process', 'processes', 'new', 'used', 'help', 'helps', 'helping', 'like', 'user',
    'users', 'application', 'applications', 'technology', 'technologies', 'tools', 'tool',
    'different', 'efficient', 'effective', 'better', 'enable', 'enables', 'ensure', 'ensures',
    'level', 'levels', 'area', 'areas', 'project', 'projects', 'create', 'make', 'give',
    'given', 'work', 'working', 'available', 'way', 'ways', 'method', 'methods', 'approach',
    'approaches', 'specific', 'overall', 'type', 'types', 'many', 'good', 'well',
    'etc', 'sector', 'sectors', 'ideas', 'idea', 'rich', 'poor', 'aspects', 'aspect',
    'point', 'points', 'field', 'fields', 'part', 'parts', 'key', 'main', 'primary'
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
    
    # 3. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 4. Tokenize by whitespace
    tokens = text.split()
    
    # 5. Filter short tokens and stopwords
    all_stops = STANDARD_STOPWORDS.union(DEFAULT_DOMAIN_STOPWORDS if remove_domain_stopwords else set())
    cleaned_tokens = [t for t in tokens if len(t) > 2 and t not in all_stops]
    
    return " ".join(cleaned_tokens)

class ProblemDataPipeline:
    """End-to-end preprocessing pipeline for problem statements."""
    def __init__(self, remove_domain_stopwords: bool = True):
        self.remove_domain_stopwords = remove_domain_stopwords
        self.scaler = StandardScaler()
        self.numerical_cols = []
        self.categorical_cols = []
        self.raw_df = None
        self.processed_df = None

    def load_and_audit(self, filepath_or_buffer) -> pd.DataFrame:
        """Loads data, evaluates variable relevance, and handles missing values."""
        df = pd.read_csv(filepath_or_buffer)
        self.raw_df = df.copy()
        
        # Standardize column names
        df.columns = [c.strip() for c in df.columns]
        
        # Fill missing values gracefully
        text_cols = ['Problem Statement Title', 'Description']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        if 'Theme' in df.columns:
            df['Theme'] = df['Theme'].fillna('Uncategorized')
        if 'Category' in df.columns:
            df['Category'] = df['Category'].fillna('Unknown')
        if 'Organization' in df.columns:
            df['Organization'] = df['Organization'].fillna('Unknown')
        if 'Department' in df.columns:
            df['Department'] = df['Department'].fillna('Unknown')
            
        # Drop columns with extreme missingness (>80%) that offer no clustering signal
        sparse_cols = ['Youtube Link', 'Dataset Link', 'Contact info']
        existing_sparse = [c for c in sparse_cols if c in df.columns]
        df = df.drop(columns=existing_sparse)
        
        # Combine Title and Description for rich semantic representation
        df['Combined_Raw_Text'] = df['Problem Statement Title'] + ". " + df['Description']
        
        # Feature Engineering: Numerical properties
        df['title_word_count'] = df['Problem Statement Title'].apply(lambda x: len(x.split()))
        df['desc_word_count'] = df['Description'].apply(lambda x: len(x.split()))
        df['char_count'] = df['Combined_Raw_Text'].apply(len)
        df['avg_word_length'] = df['Combined_Raw_Text'].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # Clean text
        df['Cleaned_Text'] = df['Combined_Raw_Text'].apply(
            lambda t: clean_text(t, remove_domain_stopwords=self.remove_domain_stopwords)
        )
        
        # Category binary encoding (Software=1, Hardware=0)
        if 'Category' in df.columns:
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
            'avg_raw_words': round(self.processed_df['desc_word_count'].mean(), 1),
            'avg_cleaned_words': round(self.processed_df['Cleaned_Text'].apply(lambda x: len(x.split())).mean(), 1)
        }

