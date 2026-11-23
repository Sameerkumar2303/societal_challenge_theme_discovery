"""
Societal Challenge Theme Discovery from Crowdsourced Problem Statements
Multi-Page Interactive AI Discovery Platform powered by Streamlit, Scikit-Learn, and Plotly.
Features high-contrast glassmorphism, animated ambient mesh gradients, neon glows, and hover-enlarge graphs.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Societal Challenge Discovery • Smart India Hackathon",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Living Aurora Ambient Mesh, Glassmorphism, Neon Glows & Hover Physics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 1. Shifting Ambient Aurora Mesh Background */
    @keyframes auroraMesh {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    .stApp {
        background: linear-gradient(135deg, #070b14 0%, #0c162d 25%, #131d3d 50%, #0d172e 75%, #060912 100%) !important;
        background-size: 280% 280% !important;
        animation: auroraMesh 20s ease-in-out infinite !important;
        color: #f8fafc !important;
    }

    /* 2. Frosted Glass Panels with Backdrop Blur */
    .glass-card {
        background: rgba(15, 23, 42, 0.72) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(59, 130, 246, 0.22) !important;
        border-radius: 16px !important;
        padding: 1.4rem 1.6rem !important;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.28s ease, box-shadow 0.28s ease !important;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 16px 40px -8px rgba(56, 189, 248, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-3px) !important;
    }

    /* 3. Hover Enlarge Effects on Graphs */
    .graph-enlarge-container {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(59, 130, 246, 0.22) !important;
        border-radius: 18px !important;
        padding: 1.2rem 1.4rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.6) !important;
        transition: transform 0.32s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.32s ease, border-color 0.32s ease !important;
    }
    .graph-enlarge-container:hover {
        transform: translateY(-5px) scale(1.012) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 20px 50px -10px rgba(56, 189, 248, 0.3) !important;
    }
    .graph-caption-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
    }

    /* 4. Page Entrance Transitions */
    @keyframes pageFadeIn {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .page-wrapper {
        animation: pageFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* 5. Rotating 3D Holographic Orbit */
    @keyframes orbit3D {
        0% { transform: rotateY(0deg) rotateX(15deg); }
        50% { transform: rotateY(180deg) rotateX(-15deg); }
        100% { transform: rotateY(360deg) rotateX(15deg); }
    }
    .hologram-orb {
        display: inline-block;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #3b82f6 40%, #1e1b4b 80%, #0f172a 100%);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.7), inset 0 0 15px rgba(255, 255, 255, 0.4);
        animation: orbit3D 10s linear infinite;
        vertical-align: middle;
        margin-right: 12px;
    }

    /* 6. Neon Pulse Dots & Badges */
    @keyframes neonPulse {
        0%, 100% {
            box-shadow: 0 0 6px #34d399, 0 0 14px rgba(16, 185, 129, 0.6);
            opacity: 1;
        }
        50% {
            box-shadow: 0 0 14px #34d399, 0 0 26px rgba(16, 185, 129, 0.9);
            opacity: 0.75;
        }
    }
    .live-status-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #10b981;
        animation: neonPulse 2s infinite ease-in-out;
        margin-right: 8px;
    }
    .neon-glow-val {
        color: #ffffff !important;
        text-shadow: 0 0 18px rgba(56, 189, 248, 0.5) !important;
    }

    /* 7. Dedicated Page Hero Banners */
    .page-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 27, 75, 0.85) 45%, rgba(30, 58, 138, 0.85) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 35px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }
    .page-hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.25);
        border: 1px solid rgba(147, 197, 253, 0.4);
        color: #93c5fd !important;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    .page-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0.1rem 0 0.5rem 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .page-hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1 !important;
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* 8. KPI Stat Cards */
    .stat-card {
        background: rgba(20, 27, 45, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(51, 65, 85, 0.7) !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.3rem !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
        text-align: left !important;
    }
    .stat-card:hover {
        transform: translateY(-3px) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 12px 30px rgba(56, 189, 248, 0.2) !important;
    }
    .stat-icon {
        font-size: 1.6rem !important;
        margin-bottom: 0.35rem !important;
    }
    .stat-label {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94a3b8 !important;
        margin-bottom: 0.2rem !important;
    }
    .stat-value {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        line-height: 1.1 !important;
    }
    .stat-caption {
        font-size: 0.76rem !important;
        color: #64748b !important;
        margin-top: 0.3rem !important;
    }

    /* 9. Sidebar Navigation Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(9, 14, 26, 0.95) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.15) !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 6px !important;
    }
    .stRadio [role="radiogroup"] label {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 10px !important;
        padding: 0.6rem 0.9rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    .stRadio [role="radiogroup"] label:hover {
        background: rgba(59, 130, 246, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.15) !important;
        transform: translateX(3px) !important;
    }

    /* Theme Showcase Card */
    .theme-card {
        background: rgba(20, 27, 45, 0.78) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(51, 65, 85, 0.8) !important;
        border-top: 4px solid #3b82f6 !important;
        border-radius: 14px !important;
        padding: 1.4rem 1.6rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 1.3rem !important;
        transition: transform 0.25s ease, border-color 0.25s ease !important;
    }
    .theme-card:hover {
        transform: translateY(-3px) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 14px 35px rgba(56, 189, 248, 0.18) !important;
    }

    /* Keyword Pills */
    .keyword-pill {
        display: inline-block !important;
        background: #0f172a !important;
        color: #e2e8f0 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 0.25rem 0.7rem !important;
        border-radius: 6px !important;
        margin-right: 0.45rem !important;
        margin-bottom: 0.45rem !important;
        border: 1px solid #334155 !important;
        transition: all 0.18s ease !important;
    }
    .keyword-pill:hover {
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        background: #1e293b !important;
        transform: scale(1.04) !important;
    }

    /* Expanders in Dark Theme */
    [data-testid="stExpander"] {
        background: rgba(20, 27, 45, 0.7) !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        margin-bottom: 0.9rem !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Add source directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessor import ProblemDataPipeline
from src.vectorizer import TextVectorizer
from src.clustering import ClusteringWorkflow
from src.hierarchical import HierarchicalWorkflow
from src.dim_reduction import PCAVisualizer
from src.profiler import ClusterProfiler

# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div class="hologram-orb"></div>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; letter-spacing: -0.01em;">Societal AI</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Theme Discovery Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧭 Platform Navigation")
    nav_pages = [
        "🏠 Overview & Hub",
        "🌟 Discovered Themes",
        "🔍 Challenge Inspector",
        "🌌 3D Theme Galaxy",
        "⚖️ Bias & Ethics Audit",
        "🔬 Science & Validation",
        "📥 Export Studio"
    ]
    
    selected_page = st.radio(
        "Select Page View",
        nav_pages,
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()

    # 1. Dataset Selection
    st.subheader("1. Data Source")
    default_data_path = os.path.join(BASE_DIR, "data", "problem_statements.csv")
    uploaded_file = st.file_uploader("Upload Custom CSV (Optional)", type=['csv'])
    
    if uploaded_file is not None:
        data_source = uploaded_file
        st.success("Custom CSV loaded!")
    else:
        data_source = default_data_path
        st.caption("Using default: **Smart India Hackathon (231 Challenges)**")

    st.divider()

    # 2. Hyperparameter controls
    st.subheader("2. AI Clustering Settings")
    num_clusters = st.slider(
        "Number of Themes to Discover (k)",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="How many distinct societal categories should the AI segment the problems into?"
    )
    
    with st.expander("⚙️ Advanced Hyperparameters", expanded=False):
        distance_metric = st.selectbox(
            "Distance Metric",
            ["Euclidean (Standard K-Means)", "Manhattan (K-Medoids Partition)"],
            index=0
        )
        metric_key = 'euclidean' if "Euclidean" in distance_metric else 'manhattan'
        remove_domain_stops = st.checkbox("Filter Domain Boilerplate Words", value=True)
        max_features = st.slider("Max Vocabulary Features", 500, 2500, 1200, 100)
        sweep_max_k = st.slider("Multi-k Validation Range", 5, 12, 8, 1)
        random_seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("""
    <div style="margin-top: 1.5rem; padding: 0.6rem 0.8rem; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px;">
        <span class="live-status-dot"></span>
        <span style="color: #34d399; font-weight: 700; font-size: 0.8rem;">Model Engine Active</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# CACHED DATA PIPELINE EXECUTION
# ==========================================
@st.cache_data(show_spinner=False)
def run_full_pipeline(source, remove_stops, max_feats, metric, k_val, max_k_eval, seed):
    # Step 1: Preprocessing
    pipeline = ProblemDataPipeline(remove_domain_stopwords=remove_stops)
    df = pipeline.load_and_audit(source)
    summary = pipeline.get_preprocessing_summary()

    # Step 2: Vectorization
    vectorizer = TextVectorizer(max_features=max_feats, ngram_range=(1, 2), min_df=2, max_df=0.85)
    X = vectorizer.fit_transform(df['Cleaned_Text'].tolist())
    feature_names = vectorizer.get_feature_names()

    # Step 3: Multi-k Evaluation Sweep
    workflow = ClusteringWorkflow(min_k=2, max_k=max_k_eval, random_state=seed)
    eval_df = workflow.evaluate_range_of_k(X, distance_metric=metric)

    # Step 4: Final Clustering Model
    final_res = workflow.fit_final_model(X, k=k_val, distance_metric=metric, random_state=seed)
    labels = final_res['labels']
    centroids = final_res['cluster_centers']

    # Step 5: Stability Assessment
    stability = workflow.assess_stability(X, k=k_val, distance_metric=metric, seeds=[seed, seed+59, seed+735])

    # Step 6: Hierarchical Clustering
    hier_wf = HierarchicalWorkflow(n_clusters=k_val, linkage_method='ward' if metric == 'euclidean' else 'average')
    hier_res = hier_wf.fit(X)
    linkage_mat = hier_wf.compute_linkage_matrix(X)
    hier_comp = hier_wf.compare_with_kmeans(hier_res['labels'], labels)

    # Step 7: PCA Dimensionality Reduction
    pca_viz = PCAVisualizer(n_components=3, random_state=seed)
    pca_coords = pca_viz.fit_transform(X)
    pca_centroids = pca_viz.project_centroids(centroids)
    pca_exp_var = pca_viz.get_explained_variance()
    pca_loadings = pca_viz.get_component_loadings(feature_names, top_n=6)

    # Step 8: Profiling & Feature Contributions
    profiler = ClusterProfiler(feature_names)
    profiles = profiler.extract_top_features_per_cluster(centroids, top_n=8)
    exemplars = profiler.find_representative_exemplars(X, centroids, labels, df, top_k_exemplars=3)
    crosstabs = profiler.cross_tabulate_metadata(labels, df)
    bias_audit = profiler.generate_bias_and_limitations_report(df, labels)

    # Enrich df with results
    enriched_df = df.copy()
    enriched_df['Cluster_ID'] = labels
    enriched_df['Cluster_Label'] = [f"Theme {l + 1}" for l in labels]
    enriched_df['Theme_Name'] = [profiles[l]['suggested_theme'] for l in labels]
    enriched_df['Theme_Title'] = [profiles[l]['theme_title'] for l in labels]
    enriched_df['Theme_Icon'] = [profiles[l].get('icon', '🌐') for l in labels]
    enriched_df['PCA_Dim1'] = pca_coords[:, 0]
    enriched_df['PCA_Dim2'] = pca_coords[:, 1]
    enriched_df['PCA_Dim3'] = pca_coords[:, 2]

    return {
        'df': enriched_df,
        'summary': summary,
        'eval_df': eval_df,
        'final_res': final_res,
        'stability': stability,
        'hier_res': hier_res,
        'linkage_mat': linkage_mat,
        'hier_comp': hier_comp,
        'pca_coords': pca_coords,
        'pca_centroids': pca_centroids,
        'pca_exp_var': pca_exp_var,
        'pca_loadings': pca_loadings,
        'profiles': profiles,
        'exemplars': exemplars,
        'crosstabs': crosstabs,
        'bias_audit': bias_audit,
        'feature_names': feature_names
    }

with st.spinner("Analyzing problem statements with AI clustering pipeline..."):
    results = run_full_pipeline(
        data_source, remove_domain_stops, max_features,
        metric_key, num_clusters, sweep_max_k, random_seed
    )

df = results['df']
profiles = results['profiles']
exemplars = results['exemplars']
summary = results['summary']


# ==============================================================================
# PAGE 1: OVERVIEW & EXECUTIVE HUB
# ==============================================================================
if selected_page == "🏠 Overview & Hub":
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-badge">Smart India Hackathon • Unsupervised Thematic AI</div>
        <div class="page-hero-title">Societal Challenge Theme Discovery</div>
        <div class="page-hero-subtitle">
            Scanning, analyzing, and grouping hundreds of crowdsourced national challenge problem statements into cohesive societal mission themes — eliminating manual categorization guesswork through natural language processing and spatial partition clustering.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Metrics Cards
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    with col_m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">📄</div>
            <div class="stat-label">Total Challenges</div>
            <div class="stat-value neon-glow-val">{len(df):,}</div>
            <div class="stat-caption">Crowdsourced submissions</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🏷️</div>
            <div class="stat-label">Discovered Themes</div>
            <div class="stat-value neon-glow-val">{num_clusters}</div>
            <div class="stat-caption">Autonomous AI clusters</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        sw_count = (df['Category'].str.lower() == 'software').sum()
        sw_pct = int(sw_count / len(df) * 100) if len(df) > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">💻</div>
            <div class="stat-label">Tech Modality</div>
            <div class="stat-value neon-glow-val">{sw_pct}% SW</div>
            <div class="stat-caption">{100 - sw_pct}% Hardware focused</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        stability_val = results['stability']['mean_ari']
        stability_status = "Very Stable" if stability_val > 0.6 else ("Moderately Stable" if stability_val > 0.3 else "Exploratory")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-label">Model Stability</div>
            <div class="stat-value neon-glow-val">{stability_val:.2f}</div>
            <div class="stat-caption">{stability_status} across seeds</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m5:
        org_count = df['Organization'].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🏛️</div>
            <div class="stat-label">Public Bodies</div>
            <div class="stat-value neon-glow-val">{org_count}</div>
            <div class="stat-caption">Ministries & departments</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Interactive Flow & Visuals
    col_ov1, col_ov2 = st.columns([1.1, 1], gap="large")

    with col_ov1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #60a5fa; margin-top: 0; font-size: 1.25rem;">🔬 Unsupervised Methodology Pipeline</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
                How the AI extracts structure from unstructured civic challenge texts:
            </p>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 0.8rem;">
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.86rem; margin-bottom: 0.2rem;">1. Text Normalization & Cleaning</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Cleans special characters, expands acronyms, and removes domain boilerplate noise (e.g., <i>'system'</i>, <i>'proposed'</i>, <i>'participants'</i>).</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 0.8rem;">
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.86rem; margin-bottom: 0.2rem;">2. TF-IDF High-Dimensional Embeddings</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Extracts salient unigrams and bigrams with L2 sublinear term frequency weighting across vocabulary features.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 0.8rem;">
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.86rem; margin-bottom: 0.2rem;">3. Spatial Partition Clustering & PCA Projection</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Partitions vectors into optimal centroids using Euclidean K-Means or Manhattan K-Medoids, projected onto 2D/3D PCA manifolds.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem; border-radius: 12px; border: 1px solid #334155;">
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.86rem; margin-bottom: 0.2rem;">4. Thematic Keyword Extraction & Exemplars</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Extracts highest-weighted centroid terms to generate human-readable societal mission titles and closest real-world exemplars.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ov2:
        st.markdown('<div class="graph-enlarge-container">', unsafe_allow_html=True)
        st.markdown('<div class="graph-caption-badge">Interactive Distribution • Hover to Enlarge</div>', unsafe_allow_html=True)
        st.markdown("#### 🏛️ Top Participating Government Bodies")
        top_orgs_overview = df['Organization'].value_counts().head(6).reset_index()
        top_orgs_overview.columns = ['Organization', 'Count']
        fig_ov_bar = px.bar(
            top_orgs_overview, x='Count', y='Organization', orientation='h',
            color='Count', color_continuous_scale='Blues'
        )
        fig_ov_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            yaxis={'categoryorder': 'total ascending'},
            height=320,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_ov_bar, use_container_width=True)
        st.caption("Hover over bars to inspect challenge counts per submitting ministry.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: DISCOVERED SOCIETAL THEMES
# ==============================================================================
elif selected_page == "🌟 Discovered Themes":
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-badge">Autonomous Thematic Synthesis</div>
        <div class="page-hero-title">🌟 Discovered Societal Challenge Themes</div>
        <div class="page-hero-subtitle">
            Unsupervised clusters automatically discovered from crowdsourced problem statements.
            Titles and characteristic keywords are derived directly from the geometric centroid weights of the vocabulary space.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter selector for themes
    theme_options = ["All Themes"] + [f"{prof['icon']} {prof['theme_title']}" for prof in profiles.values()]
    selected_theme_filter = st.selectbox("Filter by Discovered Theme", theme_options)

    # Grid of discovered theme cards
    for k_idx, prof in profiles.items():
        theme_full_name = f"{prof['icon']} {prof['theme_title']}"
        if selected_theme_filter != "All Themes" and selected_theme_filter != theme_full_name:
            continue

        cluster_subset = df[df['Cluster_ID'] == k_idx]
        cluster_count = len(cluster_subset)
        cluster_pct = (cluster_count / len(df)) * 100
        icon = prof.get('icon', '🌐')
        title = prof.get('theme_title', f"Theme {k_idx + 1}")

        st.markdown(f"""
        <div class="theme-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;">
                <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff;">
                    {icon} Theme {k_idx + 1}: {title}
                </div>
                <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); padding: 0.35rem 0.85rem; border-radius: 9999px; font-weight: 700; color: #38bdf8; font-size: 0.84rem;">
                    {cluster_count} Challenges ({cluster_pct:.1f}% of national pool)
                </div>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-right: 0.6rem;">Top Characteristic Vocabulary:</span>
        """, unsafe_allow_html=True)

        # Keyword pills
        kw_html = "".join([f'<span class="keyword-pill">{t["term"]} ({t["weight"]})</span>' for t in prof['top_features']])
        st.markdown(f"{kw_html}</div>", unsafe_allow_html=True)

        # Representative Exemplars Expander
        with st.expander(f"📌 View Top Exemplars for Theme {k_idx + 1} (Closest to Centroid)"):
            theme_exemplars = exemplars.get(k_idx, [])
            if theme_exemplars:
                for ex in theme_exemplars:
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.5); border-left: 3px solid #38bdf8; padding: 0.6rem 0.9rem; margin-bottom: 0.6rem; border-radius: 0 8px 8px 0;">
                        <div style="font-size: 0.76rem; color: #94a3b8; font-weight: 600;">[{ex['id']}] • {ex['org']} • Modality: {ex['category']} • Distance: {ex['distance_to_center']}</div>
                        <div style="font-size: 0.9rem; color: #f1f5f9; font-weight: 600; margin-top: 0.2rem;">{ex['title']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No exemplars found for this cluster.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: CHALLENGE SEARCH & INSPECTOR
# ==============================================================================
elif selected_page == "🔍 Challenge Inspector":
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-badge">Granular Challenge Analysis</div>
        <div class="page-hero-title">🔍 Challenge Search & Problem Inspector</div>
        <div class="page-hero-subtitle">
            Query and filter across all crowdsourced problem statements. Examine raw challenge descriptions alongside their autonomous AI-assigned societal mission themes and organizational metadata.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search_query = st.text_input("Search by keyword or challenge title", placeholder="e.g. satellite, soil, medicine, traffic, fraud...")
    with col_s2:
        org_filter = st.selectbox("Filter by Ministry / Body", ["All Organizations"] + sorted(df['Organization'].dropna().unique().tolist()))
    with col_s3:
        cat_filter = st.selectbox("Filter by Modality", ["All Categories"] + sorted(df['Category'].dropna().unique().tolist()))

    # Apply search filter
    filtered_view = df.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_view = filtered_view[
            filtered_view['Problem Statement Title'].str.lower().str.contains(q, na=False) |
            filtered_view['Description'].str.lower().str.contains(q, na=False) |
            filtered_view['Theme_Title'].str.lower().str.contains(q, na=False)
        ]
    if org_filter != "All Organizations":
        filtered_view = filtered_view[filtered_view['Organization'] == org_filter]
    if cat_filter != "All Categories":
        filtered_view = filtered_view[filtered_view['Category'] == cat_filter]

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 0.5rem 0.9rem; margin-bottom: 1.2rem; display: inline-block;">
        <span style="color: #38bdf8; font-weight: 700;">Found {len(filtered_view)}</span> <span style="color: #94a3b8;">matching problem statements</span>
    </div>
    """, unsafe_allow_html=True)

    for _, row in filtered_view.head(20).iterrows():
        icon = row['Theme_Icon']
        title = row['Problem Statement Title']
        ps_id = row['Problem Statement ID']
        theme_assigned = row['Theme_Title']
        org = row['Organization']
        desc = row['Description'] if pd.notna(row['Description']) else "No description provided."
        
        with st.expander(f"[{ps_id}] {title} — Assigned: {icon} {theme_assigned}"):
            st.markdown(f"**🏛️ Submitting Body:** `{org}` &nbsp;|&nbsp; **Modality:** `{row['Category']}` &nbsp;|&nbsp; **Original Tag:** `{row['Theme']}`")
            st.markdown(f"**AI Discovered Theme:** `{icon} {theme_assigned}`")
            st.markdown("**Full Problem Description:**")
            st.text_area("", value=desc, height=140, key=f"desc_{ps_id}", disabled=True)

    if len(filtered_view) > 20:
        st.info(f"Displaying first 20 of {len(filtered_view)} results. Refine your search query for more targeted inspection.")

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: 3D & 2D THEME GALAXY
# ==============================================================================
elif selected_page == "🌌 3D Theme Galaxy":
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-badge">Dimensionality Reduction • PCA Manifold</div>
        <div class="page-hero-title">🌌 Theme Galaxy: Spatial Semantic Manifold</div>
        <div class="page-hero-subtitle">
            Every point represents one national challenge statement. High-dimensional TF-IDF linguistic feature vectors are projected into 2D and 3D principal components to reveal geometric clusters and semantic orbits.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns([3, 1])
    with col_g2:
        viz_dimension = st.radio("Galaxy Perspective", ["3D Interactive Space", "2D Planetary Map"], index=0)
        show_centroids = st.checkbox("Show Theme Centers (Centroids)", value=True)
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; margin-top: 1rem;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">PCA Explained Variance</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #38bdf8; margin: 0.3rem 0;">{results['pca_exp_var']['total_retained_variance']}%</div>
            <div style="font-size: 0.76rem; color: #64748b;">Retained in 3 components</div>
        </div>
        """, unsafe_allow_html=True)

    with col_g1:
        st.markdown('<div class="graph-enlarge-container">', unsafe_allow_html=True)
        st.markdown('<div class="graph-caption-badge">Geometric Embedding • Hover or Drag to Orbit</div>', unsafe_allow_html=True)
        
        if viz_dimension == "2D Planetary Map":
            fig_galaxy = px.scatter(
                df, x='PCA_Dim1', y='PCA_Dim2',
                color='Theme_Name',
                hover_name='Problem Statement Title',
                hover_data={'Problem Statement ID': True, 'Organization': True, 'Theme': True, 'PCA_Dim1': False, 'PCA_Dim2': False},
                title="2D Map of Discovered Societal Challenge Themes",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            if show_centroids:
                cent = results['pca_centroids']
                fig_galaxy.add_trace(go.Scatter(
                    x=cent[:, 0], y=cent[:, 1], mode='markers+text',
                    marker=dict(symbol='diamond', size=16, color='#38bdf8', line=dict(width=2, color='white')),
                    text=[f"Theme {i+1} Center" for i in range(len(cent))],
                    textposition="top center",
                    name="Theme Centers"
                ))
            fig_galaxy.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font_color="#e2e8f0",
                height=560,
                legend=dict(orientation="h", yanchor="bottom", y=-0.35)
            )
            st.plotly_chart(fig_galaxy, use_container_width=True)
        else:
            fig_galaxy3d = px.scatter_3d(
                df, x='PCA_Dim1', y='PCA_Dim2', z='PCA_Dim3',
                color='Theme_Name',
                hover_name='Problem Statement Title',
                hover_data={'Problem Statement ID': True, 'Organization': True, 'Theme': True},
                title="3D Interactive Manifold of Societal Challenge Themes",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            if show_centroids:
                cent = results['pca_centroids']
                fig_galaxy3d.add_trace(go.Scatter3d(
                    x=cent[:, 0], y=cent[:, 1], z=cent[:, 2],
                    mode='markers+text',
                    marker=dict(symbol='diamond', size=10, color='#38bdf8', line=dict(width=2, color='white')),
                    text=[f"Center {i+1}" for i in range(len(cent))],
                    textposition="top center",
                    name="Theme Centers"
                ))
            fig_galaxy3d.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=620
            )
            st.plotly_chart(fig_galaxy3d, use_container_width=True)
        
        st.caption("Each cluster forms a distinct semantic galaxy. Hover over any node to inspect the challenge title, department, and PCA coordinates.")
        st.markdown('</div>', unsafe_allow_html=True)

    # PCA Component Loadings Interpretation Table
    st.markdown("#### 📐 PCA Geometric Axis Interpretation (Top Word Loadings)")
    loadings = results['pca_loadings']
    col_l1, col_l2, col_l3 = st.columns(3)
    
    with col_l1:
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">Principal Component 1 (PC1)</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.5rem;">Primary semantic variance axis</div>
        """, unsafe_allow_html=True)
        st.write("**Top Positive Features:**")
        st.write(", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC1']['positive'][:4]]))
        st.write("**Top Negative Features:**")
        st.write(", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC1']['negative'][:4]]))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_l2:
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">Principal Component 2 (PC2)</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.5rem;">Orthogonal variance axis</div>
        """, unsafe_allow_html=True)
        st.write("**Top Positive Features:**")
        st.write(", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC2']['positive'][:4]]))
        st.write("**Top Negative Features:**")
        st.write(", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC2']['negative'][:4]]))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_l3:
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">Principal Component 3 (PC3)</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.5rem;">Tertiary depth dimension</div>
        """, unsafe_allow_html=True)
        st.write("**Top Positive Features:**")
        st.write(", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC3']['positive'][:4]]))
        st.write("**Top Negative Features:**")
        st.write(", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC3']['negative'][:4]]))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: AI VS HUMAN TAGS & BIAS AUDIT
# ==============================================================================
elif selected_page == "⚖️ Bias & Ethics Audit":
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-badge">Responsible AI • Taxonomy Audit</div>
        <div class="page-hero-title">⚖️ AI vs. Human Tags & Institutional Bias Audit</div>
        <div class="page-hero-subtitle">
            Auditing how unsupervised clustering resolves ambiguities in manual human labels, investigating institutional submission concentration, and documenting ethical boundaries of automated categorization.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cp1, col_cp2 = st.columns(2, gap="large")
    
    with col_cp1:
        st.markdown('<div class="graph-enlarge-container">', unsafe_allow_html=True)
        st.markdown('<div class="graph-caption-badge">Institutional Concentration</div>', unsafe_allow_html=True)
        st.markdown("#### 1. Top Submitting Ministries & Agencies")
        top_orgs = df['Organization'].value_counts().head(8).reset_index()
        top_orgs.columns = ['Organization', 'Challenge Count']
        fig_org_bar = px.bar(
            top_orgs, x='Challenge Count', y='Organization', orientation='h',
            color='Challenge Count', color_continuous_scale='Blues'
        )
        fig_org_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            yaxis={'categoryorder': 'total ascending'},
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_org_bar, use_container_width=True)
        st.caption("Highlights institutional skew: a handful of public bodies account for a major share of submissions.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cp2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 2. Resolving Vague Catch-All Tags")
        st.markdown("""
        In raw crowdsourced datasets, generic catch-all labels are pervasive.
        Unsupervised NLP disambiguates these generic tags into concrete technical clusters:
        """)
        vague_df = df[df['Theme'].isin(['Smart Automation', 'Miscellaneous'])]
        if len(vague_df) > 0:
            regroup_counts = vague_df.groupby(['Theme', 'Theme_Title']).size().unstack(fill_value=0)
            st.dataframe(regroup_counts, use_container_width=True)
            st.caption("Demonstrates how NLP reveals specific domain clusters within vague labels.")
        else:
            top_themes = df['Theme'].value_counts().head(5).index.tolist()
            subset_df = df[df['Theme'].isin(top_themes)]
            if len(subset_df) > 0:
                regroup_counts = subset_df.groupby(['Theme', 'Theme_Title']).size().unstack(fill_value=0)
                st.dataframe(regroup_counts, use_container_width=True)
                st.caption("Cross-tabulation of discovered AI themes against top source themes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🚨 Responsible AI & Ethical Boundaries")
    col_eth1, col_eth2 = st.columns(2, gap="large")
    
    with col_eth1:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #f59e0b;">
            <div style="font-weight: 700; color: #fbbf24; font-size: 0.95rem; margin-bottom: 0.4rem;">⚠️ Exploratory Discovery vs. Rigid Dogma</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                Unsupervised clusters are mathematically discovered groupings based on lexical co-occurrence. 
                They serve as discovery lenses for policy makers and evaluators, not as permanent or exclusive real-world boundaries.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_eth2:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #38bdf8;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; margin-bottom: 0.4rem;">ℹ️ Linguistic & Lexical Polysemy</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                Words such as <i>'monitoring'</i>, <i>'detection'</i>, and <i>'prediction'</i> naturally cut across multiple domains 
                (agriculture, healthcare, defense). Contextual human oversight remains essential for borderline challenges.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: MATHEMATICAL SCIENCE & VALIDATION
# ==============================================================================
elif selected_page == "🔬 Science & Validation":
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-badge">Rigorous Quantitative Verification</div>
        <div class="page-hero-title">🔬 Data Science, Metrics & Cluster Validation</div>
        <div class="page-hero-subtitle">
            Validating clustering quality across multi-k parameter sweeps, inspecting bottom-up Agglomerative Hierarchical linkage dendrograms, and assessing random seed stability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Multi-k sweep curves in hover-enlarge container
    st.markdown('<div class="graph-enlarge-container">', unsafe_allow_html=True)
    st.markdown('<div class="graph-caption-badge">Multi-k Sweep • Hover to Enlarge</div>', unsafe_allow_html=True)
    st.markdown("#### 📈 Multi-k Hyperparameter Validation Sweep")
    
    eval_df = results['eval_df']
    col_ev1, col_ev2, col_ev3 = st.columns(3)
    
    with col_ev1:
        fig_wcss = px.line(eval_df, x='k', y='WCSS', markers=True, title="Elbow Curve (Inertia)")
        fig_wcss.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_wcss, use_container_width=True)
        st.caption("Diminishing returns elbow inspection.")

    with col_ev2:
        fig_sil = px.line(eval_df, x='k', y='Silhouette_Score', markers=True, title="Silhouette Score Curve")
        fig_sil.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_sil, use_container_width=True)
        st.caption("Cohesion vs. separation (higher is better).")

    with col_ev3:
        fig_db = px.line(eval_df, x='k', y='Davies_Bouldin_Index', markers=True, title="Davies-Bouldin Index")
        fig_db.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_db, use_container_width=True)
        st.caption("Cluster similarity index (lower is better).")

    st.markdown('</div>', unsafe_allow_html=True)

    # Hierarchical Clustering Dendrogram
    col_h1, col_h2 = st.columns([1.3, 1], gap="large")
    
    with col_h1:
        st.markdown('<div class="graph-enlarge-container">', unsafe_allow_html=True)
        st.markdown('<div class="graph-caption-badge">Taxonomic Hierarchy</div>', unsafe_allow_html=True)
        st.markdown("#### 🌳 Agglomerative Hierarchical Linkage Dendrogram")
        fig_dend, ax_dend = plt.subplots(figsize=(10, 4.2))
        fig_dend.patch.set_facecolor('#0f172a')
        ax_dend.set_facecolor('#0f172a')
        
        dendrogram(
            results['linkage_mat'],
            truncate_mode='lastp',
            p=20,
            leaf_rotation=90.,
            leaf_font_size=9.,
            show_contracted=True,
            ax=ax_dend,
            color_threshold=results['linkage_mat'][-num_clusters+1, 2] if num_clusters > 1 else None
        )
        ax_dend.tick_params(colors='#94a3b8')
        for spine in ax_dend.spines.values():
            spine.set_color('#334155')
        st.pyplot(fig_dend)
        st.caption("Bottom-up hierarchical dendrogram showing how challenge themes merge at increasing distance thresholds.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_h2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Stability & Concordance")
        stability_res = results['stability']
        st.markdown(f"""
        - **Mean Adjusted Rand Index (ARI):** `{stability_res['mean_ari']:.4f}`
        - **Stability Status:** `{"Very High (>0.6)" if stability_res['mean_ari'] > 0.6 else "Moderate"}`
        - **Hierarchical vs K-Means ARI:** `{results['hier_comp']['adjusted_rand_index']:.4f}`
        - **Hierarchical vs K-Means NMI:** `{results['hier_comp']['normalized_mutual_info']:.4f}`
        """)
        st.write("**Contingency Cross-Tabulation:**")
        contingency_df = results['hier_comp'].get('contingency_matrix', results['hier_comp'].get('contingency_table'))
        if contingency_df is not None:
            st.dataframe(contingency_df, use_container_width=True)
        st.caption("High concordance between partition and hierarchical clustering validates natural cluster emergence.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: EXPORT STUDIO & REPORTS
# ==============================================================================
elif selected_page == "📥 Export Studio":
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-badge">Artifacts & Data Delivery</div>
        <div class="page-hero-title">📥 Executive Export Studio & Reports</div>
        <div class="page-hero-subtitle">
            Download the complete clustered dataset with autonomous theme tags, centroid distances, and 2D/3D PCA coordinates, or export a comprehensive Markdown executive summary.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns(2, gap="large")

    with col_exp1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #38bdf8; margin-top: 0;">📊 Clustered Dataset (.csv)</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Includes all original problem statement fields plus:
            </p>
            <ul style="color: #94a3b8; font-size: 0.86rem;">
                <li><code>Cluster_ID</code> (0 to k-1)</li>
                <li><code>Theme_Title</code> (Autonomous AI Title)</li>
                <li><code>PCA_Dim1</code>, <code>PCA_Dim2</code>, <code>PCA_Dim3</code></li>
                <li>Scaled numerical features & word density</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Dataset (CSV)",
            data=csv_data,
            file_name="societal_challenge_themes_enriched.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #38bdf8; margin-top: 0;">📑 Executive Summary Report (.md)</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Markdown report containing quantitative validation scores, discovered theme breakdowns, top feature keywords, and ethical bias audit findings.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        report_md = f"""# Executive Analysis Report: Societal Challenge Theme Discovery
## Smart India Hackathon Crowdsourced Problem Statement Clustering

### 1. Executive Metrics
- **Total Problem Statements Analyzed:** {len(df)}
- **Discovered Thematic Clusters:** {num_clusters}
- **Clustering Algorithm:** {distance_metric}
- **Silhouette Validation Score:** {results['final_res']['silhouette']:.4f}
- **Davies-Bouldin Index:** {results['final_res']['davies_bouldin']:.4f}
- **Cluster Stability (ARI across seeds):** {results['stability']['mean_ari']:.4f}

### 2. Discovered Societal Mission Themes
"""
        for k_idx, prof in profiles.items():
            count_k = len(df[df['Cluster_ID'] == k_idx])
            pct_k = count_k / len(df) * 100
            top_t = ", ".join([f"{t['term']} ({t['weight']})" for t in prof['top_features'][:5]])
            report_md += f"\n#### {prof.get('icon', '🌐')} {prof['theme_title']}\n- **Count:** {count_k} challenges ({pct_k:.1f}% of total)\n- **Top Key Terms:** {top_t}\n"

        report_md += """
### 3. Policy & Operational Guidance
1. **Targeted Hackathon Tracks:** Align evaluation panels with discovered thematic clusters rather than broad buckets.
2. **Inter-Departmental Collaboration:** Challenges with shared lexical roots should be addressed collaboratively between relevant ministries.
"""
        st.download_button(
            label="📑 Download Executive Report (.md)",
            data=report_md.encode('utf-8'),
            file_name="societal_challenge_executive_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 🔍 Dataset Preview")
    preview_cols = ['Problem Statement ID', 'Problem Statement Title', 'Organization', 'Category', 'Theme_Title', 'PCA_Dim1', 'PCA_Dim2', 'PCA_Dim3']
    st.dataframe(df[preview_cols].head(15), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
