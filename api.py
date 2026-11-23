"""
Societal Challenge Theme Discovery - Dynamic FastAPI Backend Service
Exposes RESTful API endpoints for crowdsourced challenge submissions,
database queries, live AI thematic classification, and model metrics.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is on python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database import ChallengeDatabase
from src.preprocessor import ProblemDataPipeline
from src.vectorizer import TextVectorizer
from src.clustering import ClusteringWorkflow
from src.dim_reduction import PCAVisualizer
from src.profiler import ClusterProfiler
from src.live_classifier import LiveChallengeClassifier

app = FastAPI(
    title="Societal Challenge Theme Discovery API",
    description="Dynamic RESTful backend service for crowdsourcing and unsupervised AI clustering of national problem statements.",
    version="2.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database
db = ChallengeDatabase()
db.seed_from_csv()

# Pipeline Global State Cache
_pipeline_cache: Dict[str, Any] = {}

def get_or_train_pipeline():
    """Initializes or retrieves the cached clustering pipeline and live classifier."""
    global _pipeline_cache
    if _pipeline_cache.get('classifier') is not None:
        return _pipeline_cache

    df = db.get_all_problems()
    if df.empty:
        db.seed_from_csv(force=True)
        df = db.get_all_problems()

    pipeline = ProblemDataPipeline(remove_domain_stopwords=True)
    processed_df = pipeline.load_and_audit(df)

    vectorizer = TextVectorizer(max_features=1200, ngram_range=(1, 2), min_df=2, max_df=0.85)
    X = vectorizer.fit_transform(processed_df['Cleaned_Text'].tolist())
    feature_names = vectorizer.get_feature_names()

    workflow = ClusteringWorkflow(min_k=2, max_k=8, random_state=42)
    final_res = workflow.fit_final_model(X, k=5, distance_metric='euclidean', random_state=42)
    labels = final_res['labels']
    centroids = final_res['cluster_centers']

    pca_viz = PCAVisualizer(n_components=3, random_state=42)
    pca_coords = pca_viz.fit_transform(X)

    profiler = ClusterProfiler(feature_names)
    profiles = profiler.extract_top_features_per_cluster(centroids, top_n=8)

    processed_df['Cluster_ID'] = labels
    processed_df['Theme_Title'] = [profiles[l]['theme_title'] for l in labels]

    classifier = LiveChallengeClassifier(
        vectorizer=vectorizer,
        cluster_centers=centroids,
        profiles=profiles,
        pca_model=pca_viz,
        X_existing=X,
        df_existing=processed_df,
        distance_metric='euclidean'
    )

    _pipeline_cache = {
        'pipeline': pipeline,
        'vectorizer': vectorizer,
        'workflow': workflow,
        'final_res': final_res,
        'profiles': profiles,
        'pca_viz': pca_viz,
        'classifier': classifier,
        'df': processed_df
    }
    return _pipeline_cache


# ------------------------------------------
# PYDANTIC SCHEMAS
# ------------------------------------------
class ProblemSubmission(BaseModel):
    title: str = Field(..., min_length=5, description="Headline or title of the challenge statement")
    description: str = Field(..., min_length=15, description="Full context, root cause, and requirements")
    organization: Optional[str] = Field("General / Crowdsourced", description="Ministry, agency, or department")
    department: Optional[str] = Field("Public Citizen", description="Specific department or unit")
    category: Optional[str] = Field("Software", description="'Software' or 'Hardware'")
    submitted_by: Optional[str] = Field("Anonymous Citizen", description="Contributor name or team")

class ClassifyRequest(BaseModel):
    title: str = Field(..., description="Problem statement title")
    description: str = Field(..., description="Problem statement details")
    top_k_similar: Optional[int] = Field(3, ge=1, le=10, description="Number of nearest existing challenges to retrieve")


# ------------------------------------------
# ENDPOINTS
# ------------------------------------------
@app.get("/", tags=["Overview"])
def root():
    return {
        "project": "Societal Challenge Theme Discovery API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": [
            "/api/health",
            "/api/stats",
            "/api/problems",
            "/api/classify",
            "/api/crowdsource",
            "/api/themes"
        ]
    }

@app.get("/api/health", tags=["System"])
def health_check():
    stats = db.get_stats()
    return {
        "status": "healthy",
        "database_connected": True,
        "total_records": stats['total_statements'],
        "crowdsourced_records": stats['crowdsourced_count']
    }

@app.get("/api/stats", tags=["System"])
def get_stats():
    return db.get_stats()

@app.get("/api/problems", tags=["Problem Statements"])
def list_problems(
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None,
    category: Optional[str] = None,
    crowdsourced_only: bool = False
):
    df = db.get_all_problems(include_crowdsourced=True)
    if df.empty:
        return {"total": 0, "problems": []}

    if crowdsourced_only:
        df = df[df['is_crowdsourced'] == 1]
    if category and category.lower() != 'all':
        df = df[df['Category'].str.lower() == category.lower()]
    if search:
        q = search.lower().strip()
        df = df[
            df['Problem Statement Title'].str.lower().str.contains(q, na=False) |
            df['Description'].str.lower().str.contains(q, na=False)
        ]

    total = len(df)
    page_items = df.iloc[offset : offset + limit].to_dict(orient="records")
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "problems": page_items
    }

@app.post("/api/problems", status_code=status.HTTP_201_CREATED, tags=["Problem Statements"])
def submit_problem(submission: ProblemSubmission):
    result = db.add_problem(
        title=submission.title,
        description=submission.description,
        organization=submission.organization,
        department=submission.department,
        category=submission.category,
        submitted_by=submission.submitted_by
    )
    return {"message": "Problem statement successfully submitted!", "data": result}

@app.post("/api/classify", tags=["AI Inference"])
def classify_statement(payload: ClassifyRequest):
    pipeline_state = get_or_train_pipeline()
    classifier = pipeline_state['classifier']
    results = classifier.classify(
        title=payload.title,
        description=payload.description,
        top_k_similar=payload.top_k_similar
    )
    return results

@app.post("/api/crowdsource", status_code=status.HTTP_201_CREATED, tags=["Crowdsourcing & AI"])
def crowdsource_and_classify(submission: ProblemSubmission):
    """
    Submits a new crowdsourced challenge to the database AND
    immediately executes real-time AI inference to discover its societal theme.
    """
    pipeline_state = get_or_train_pipeline()
    classifier = pipeline_state['classifier']
    
    # 1. Real-time AI classification
    inference_res = classifier.classify(
        title=submission.title,
        description=submission.description,
        top_k_similar=3
    )

    # 2. Persist to SQLite Database with the discovered AI theme
    db_res = db.add_problem(
        title=submission.title,
        description=submission.description,
        organization=submission.organization,
        department=submission.department,
        category=submission.category,
        theme=inference_res['theme_title'],
        submitted_by=submission.submitted_by
    )

    return {
        "message": "Challenge statement crowdsourced and categorized!",
        "database_record": db_res,
        "ai_discovery": inference_res
    }

@app.get("/api/themes", tags=["AI Inference"])
def get_discovered_themes():
    pipeline_state = get_or_train_pipeline()
    profiles = pipeline_state['profiles']
    return {"discovered_themes": profiles}
