"""
Societal Challenge Theme Discovery from Crowdsourced Problem Statements
Interactive AI Discovery Platform powered by Streamlit, Scikit-Learn, and Plotly.
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

# Custom Styling (High-Contrast Modern Dark UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #1e3a8a 100%);
        border-radius: 16px;
        padding: 2.2rem 2.4rem;
        color: #ffffff !important;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.25);
        border: 1px solid rgba(147, 197, 253, 0.4);
        color: #93c5fd !important;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0.2rem 0 0.6rem 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1 !important;
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Stat Cards */
    .stat-card {
        background: #141b2d !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.3rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.15s ease, border-color 0.15s ease !important;
        text-align: left !important;
    }
    .stat-card:hover {
        transform: translateY(-2px) !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2) !important;
    }
    .stat-icon {
        font-size: 1.6rem !important;
        margin-bottom: 0.4rem !important;
    }
    .stat-label {
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94a3b8 !important;
        margin-bottom: 0.25rem !important;
    }
    .stat-value {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        line-height: 1.1 !important;
    }
    .stat-caption {
        font-size: 0.78rem !important;
        color: #64748b !important;
        margin-top: 0.35rem !important;
    }

    /* Explanation Boxes */
    .explainer-card {
        background: #141b2d !important;
        border: 1px solid #334155 !important;
        border-left: 5px solid #3b82f6 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
    }
    .explainer-title {
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        color: #60a5fa !important;
        margin-bottom: 0.4rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    .explainer-text {
        font-size: 0.94rem !important;
        color: #cbd5e1 !important;
        line-height: 1.65 !important;
        margin: 0 !important;
    }

    /* Theme Showcase Card */
    .theme-card {
        background: #141b2d !important;
        border: 1px solid #334155 !important;
        border-top: 4px solid #3b82f6 !important;
        border-radius: 14px !important;
        padding: 1.4rem 1.6rem !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
        margin-bottom: 1.2rem !important;
    }
    .theme-header {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 0.9rem !important;
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    .theme-title {
        font-size: 1.3rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
        letter-spacing: -0.01em !important;
    }
    .theme-badge {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #60a5fa !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.8rem !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(96, 165, 250, 0.4) !important;
    }
    .keyword-label {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        margin-right: 0.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
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
        transition: all 0.15s ease !important;
    }
    .keyword-pill:hover {
        border-color: #60a5fa !important;
        color: #ffffff !important;
        background: #1e293b !important;
    }

    /* Expanders in Dark Theme */
    [data-testid="stExpander"] {
        background: #141b2d !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        margin-bottom: 1rem !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #60a5fa !important;
    }

    /* Tab Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        border-bottom: 2px solid #334155 !important;
        padding-bottom: 2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc !important;
        background: rgba(59, 130, 246, 0.1) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
        border-bottom: 3px solid #3b82f6 !important;
        background: rgba(59, 130, 246, 0.15) !important;
    }

    /* Text Area Disabled State */
    .stTextArea textarea:disabled {
        background: #0f172a !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
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
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/128/artificial-intelligence.png", width=70)
    st.title("Settings & Controls")
    
    # Mode selector: Executive vs Advanced
    user_mode = st.radio(
        "Display Mode",
        ["🌟 Intuitive / Executive Mode", "🔬 Data Scientist / Research Mode"],
        index=0,
        help="Switch between a user-friendly high-level overview and in-depth mathematical cluster diagnostics."
    )
    is_advanced = "Data Scientist" in user_mode
    
    st.divider()
    
    # 1. Dataset Selection
    st.subheader("1. Data Source")
    default_data_path = os.path.join(BASE_DIR, "data", "problem_statements.csv")
    uploaded_file = st.file_uploader("Upload Custom CSV (Optional)", type=['csv'])
    
    if uploaded_file is not None:
        data_source = uploaded_file
        st.success("Custom CSV uploaded!")
    else:
        data_source = default_data_path
        st.caption("Using default: **Smart India Hackathon Dataset (231 Statements)**")

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
    
    if is_advanced:
        st.subheader("Advanced Hyperparameters")
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
    else:
        # Intuitive defaults for executive mode
        distance_metric = "Euclidean (Standard K-Means)"
        metric_key = 'euclidean'
        remove_domain_stops = True
        max_features = 1200
        sweep_max_k = 8
        random_seed = 42

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

# ==========================================
# HERO BANNER
# ==========================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">Smart India Hackathon • Unsupervised Thematic AI</div>
    <div class="hero-title">Societal Challenge Theme Discovery</div>
    <div class="hero-subtitle">
        Automatically scanning, analyzing, and grouping hundreds of crowdsourced national challenge problem statements into cohesive societal mission themes — eliminating manual categorization guesswork.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# QUICK START GUIDE (COLLAPSIBLE FOR BEGINNERS)
# ==========================================
with st.expander("💡 **New here? Click to read the 60-Second Quick Start Guide**", expanded=False):
    st.markdown("""
    #### 🚀 What is this application solving?
    When government bodies and universities host large-scale open innovation competitions (like the **Smart India Hackathon**), 
    hundreds of ministries, departments, and citizens submit unstructured challenge descriptions.
    
    - **The Problem**: Manual tagging is slow and subjective. Over 24% of submissions were vaguely tagged as *"Smart Automation"* or dumped into *"Miscellaneous"*.
    - **How AI Helps**: This system reads the full English text of every problem statement, removes irrelevant noise, learns word relationships (using **TF-IDF NLP**), and groups similar problems together into **Latent Societal Themes** using machine learning (**K-Means / K-Medoids**).
    - **What You Can Do**:
        1. Explore the **Discovered Themes** below to see real problem clusters (e.g. Agriculture, Weather & Disaster, Land Records, Surveillance).
        2. Use the **Problem Search & Explorer** to look up any challenge by keyword (like *"flood"* or *"drone"*) or by Ministry.
        3. View the **Theme Galaxy (2D/3D Map)** to see how challenges visually orbit near similar topics.
    """)

# ==========================================
# KEY METRIC SUMMARY CARDS
# ==========================================
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📄</div>
        <div class="stat-label">Total Challenges</div>
        <div class="stat-value">{len(df):,}</div>
        <div class="stat-caption">Crowdsourced submissions</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">🏷️</div>
        <div class="stat-label">Discovered Themes</div>
        <div class="stat-value">{num_clusters}</div>
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
        <div class="stat-value">{sw_pct}% SW</div>
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
        <div class="stat-value">{stability_val:.2f}</div>
        <div class="stat-caption">{stability_status} across seeds</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    org_count = df['Organization'].nunique()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">🏛️</div>
        <div class="stat-label">Public Bodies</div>
        <div class="stat-value">{org_count}</div>
        <div class="stat-caption">Ministries & departments</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# MAIN APPLICATION TABS
# ==========================================
tab_showcase, tab_search, tab_galaxy, tab_comparison, tab_science, tab_export = st.tabs([
    "🌟 Discovered Themes",
    "🔍 Problem Search & Inspector",
    "🌌 Theme Galaxy (2D/3D Map)",
    "⚖️ AI vs Human Tags & Bias",
    "🔬 Data Science & Validation",
    "📥 Executive Summary & Export"
])

# ------------------------------------------
# TAB 1: DISCOVERED THEMES SHOWCASE
# ------------------------------------------
with tab_showcase:
    st.markdown("""
    <div class="explainer-card">
        <div class="explainer-title">📌 What are you looking at?</div>
        <div class="explainer-text">
            Below are the distinct societal mission themes discovered automatically by the AI.
            The title and keywords are extracted from the most characteristic vocabulary of problem statements in each group.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grid of discovered theme cards
    for k_idx, prof in profiles.items():
        cluster_subset = df[df['Cluster_ID'] == k_idx]
        cluster_count = len(cluster_subset)
        cluster_pct = (cluster_count / len(df)) * 100
        icon = prof.get('icon', '🌐')
        title = prof.get('theme_title', f"Theme {k_idx + 1}")
        
        with st.container():
            st.markdown(f"""
            <div class="theme-card">
                <div class="theme-header">
                    <div class="theme-title">{icon} {title}</div>
                    <span class="theme-badge">{cluster_count} Problems ({cluster_pct:.1f}%)</span>
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="keyword-label">KEY TOPICS:</span>
                    {"".join([f'<span class="keyword-pill">{t["term"].title()}</span>' for t in prof["top_features"][:6]])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Details & Exemplars Accordion
            with st.expander(f"👀 View Representative Examples & Problems for {icon} {title} ({cluster_count} statements)"):
                col_ex_left, col_ex_right = st.columns([1, 1])
                
                with col_ex_left:
                    st.markdown("**🏆 Most Representative Problems (Closest to Theme Center):**")
                    for ex in exemplars[k_idx]:
                        st.markdown(f"- **[{ex['id']}] {ex['title']}**")
                        st.caption(f"🏛️ Organization: {ex['org']} | 📂 Dept: {ex['dept']}")
                
                with col_ex_right:
                    terms_df = pd.DataFrame(prof['top_features'][:6])
                    fig_bar = px.bar(
                        terms_df, x='weight', y='term', orientation='h',
                        title="Top Keyword Influence in this Theme",
                        color='weight', color_continuous_scale='Blues',
                        labels={'weight': 'Relevance Weight', 'term': 'Term'}
                    )
                    fig_bar.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        yaxis={'categoryorder': 'total ascending'},
                        height=240,
                        margin=dict(l=0, r=0, t=35, b=0)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("**All Problem Statements in this Theme:**")
                st.dataframe(
                    cluster_subset[['Problem Statement ID', 'Problem Statement Title', 'Organization', 'Category', 'Theme']],
                    height=200,
                    use_container_width=True
                )
            st.write("")

# ------------------------------------------
# TAB 2: INTERACTIVE PROBLEM SEARCH & INSPECTOR
# ------------------------------------------
with tab_search:
    st.markdown("""
    <div class="explainer-card">
        <div class="explainer-title">🔎 Search & Inspect Any Challenge</div>
        <div class="explainer-text">
            Type any keyword (e.g., <i>'flood'</i>, <i>'drone'</i>, <i>'crop'</i>, <i>'railway'</i>, <i>'hospital'</i>) or filter by public ministry to see which theme the AI placed it in, along with the full challenge details.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search_query = st.text_input("Search by keyword or challenge title", placeholder="e.g. satellite, soil, medicine, traffic...")
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

    st.markdown(f"**Found {len(filtered_view)} matching problem statements:**")

    for _, row in filtered_view.head(15).iterrows():
        icon = row['Theme_Icon']
        title = row['Problem Statement Title']
        ps_id = row['Problem Statement ID']
        theme_assigned = row['Theme_Title']
        org = row['Organization']
        desc = row['Description'] if pd.notna(row['Description']) else "No description provided."
        
        with st.expander(f"[{ps_id}] {title} — Assigned to: {icon} {theme_assigned}"):
            st.markdown(f"**🏛️ Submitting Body:** `{org}` &nbsp;|&nbsp; **Category:** `{row['Category']}` &nbsp;|&nbsp; **Original Human Tag:** `{row['Theme']}`")
            st.markdown(f"**AI Discovered Theme:** `{icon} {theme_assigned}`")
            st.markdown("**Full Description:**")
            st.text_area("", value=desc, height=140, key=f"desc_{ps_id}", disabled=True)

    if len(filtered_view) > 15:
        st.info(f"Showing first 15 of {len(filtered_view)} results. Refine search criteria for narrower lookups.")

# ------------------------------------------
# TAB 3: THEME GALAXY (2D & 3D PCA MAP)
# ------------------------------------------
with tab_galaxy:
    st.markdown("""
    <div class="explainer-card">
        <div class="explainer-title">🌌 How to interpret the Theme Galaxy</div>
        <div class="explainer-text">
            Each dot represents <b>one real problem statement</b>. The AI collapsed thousands of vocabulary words into 2 or 3 geometric dimensions. 
            Problems discussing similar technologies and societal goals are mapped closer to each other.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns([3, 1])
    with col_g2:
        viz_dimension = st.radio("Galaxy Perspective", ["2D Planetary Map", "3D Interactive Space"], index=0)
        show_centroids = st.checkbox("Show Theme Centers (Centroids)", value=True)

    with col_g1:
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
                    marker=dict(symbol='diamond', size=16, color='#60a5fa', line=dict(width=2, color='white')),
                    text=[f"Theme {i+1} Center" for i in range(len(cent))],
                    textposition="top center",
                    name="Theme Centers"
                ))
            fig_galaxy.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font_color="#e2e8f0",
                height=520,
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
            fig_galaxy3d.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=600
            )
            st.plotly_chart(fig_galaxy3d, use_container_width=True)

# ------------------------------------------
# TAB 4: AI VS HUMAN TAGS & BIAS AUDIT
# ------------------------------------------
with tab_comparison:
    st.markdown("""
    <div class="explainer-card">
        <div class="explainer-title">⚖️ Auditing Human Categorization & Ministry Representation</div>
        <div class="explainer-text">
            Compare how the AI clustered the problems compared to the pre-existing manual tags, and see which public bodies dominate the challenge pool.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cp1, col_cp2 = st.columns(2)
    
    with col_cp1:
        st.markdown("#### 1. Top Submitting Ministries & Departments")
        top_orgs = df['Organization'].value_counts().head(8).reset_index()
        top_orgs.columns = ['Organization', 'Challenge Count']
        fig_org_bar = px.bar(
            top_orgs, x='Challenge Count', y='Organization', orientation='h',
            color='Challenge Count', color_continuous_scale='Blues',
            title="Institutions with Most Problem Submissions"
        )
        fig_org_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            yaxis={'categoryorder': 'total ascending'},
            height=320
        )
        st.plotly_chart(fig_org_bar, use_container_width=True)

    with col_cp2:
        st.markdown("#### 2. Disambiguating 'Smart Automation' & 'Miscellaneous'")
        st.markdown("""
        In the original manual dataset:
        - **55 challenges** were tagged generically as *"Smart Automation"*.
        - **15 challenges** were dumped into *"Miscellaneous"*.
        
        **How the AI regrouped them:**
        """)
        vague_df = df[df['Theme'].isin(['Smart Automation', 'Miscellaneous'])]
        if len(vague_df) > 0:
            regroup_counts = vague_df.groupby(['Theme', 'Theme_Title']).size().unstack(fill_value=0)
            st.dataframe(regroup_counts, use_container_width=True)
            st.caption("The table demonstrates how unsupervised NLP reveals specific domain clusters within vague labels.")

    st.markdown("---")
    st.markdown("#### 🚨 Responsible AI & Ethical Use Disclaimers")
    col_eth1, col_eth2 = st.columns(2)
    with col_eth1:
        st.warning("""
        **Exploratory Discovery vs. Rigid Dogma:**
        Unsupervised clusters are mathematically discovered groupings based on lexical co-occurrence. They should serve as discovery lenses for policy makers, not as permanent or exclusive real-world boundaries.
        """)
    with col_eth2:
        st.info("""
        **Participation Bias:**
        The dataset reflects priorities of ministries that actively participate in national hackathons (e.g., Land Resources, Defence, Space) and should not be assumed to mirror the complete societal challenge landscape of India.
        """)

# ------------------------------------------
# TAB 5: DATA SCIENCE VALIDATION & RIGOR
# ------------------------------------------
with tab_science:
    st.markdown("""
    <div class="explainer-card">
        <div class="explainer-title">🔬 Quantitative Validation & Mathematical Soundness</div>
        <div class="explainer-text">
            For academic and technical reviewers: this tab demonstrates the quantitative validation metrics used to evaluate cluster compactness, separation, and stability across different hyperparameter sweeps.
        </div>
    </div>
    """, unsafe_allow_html=True)

    eval_df = results['eval_df']
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    
    with col_sc1:
        fig_elbow = px.line(eval_df, x='k', y='WCSS', markers=True, title="Elbow Method (Inertia vs k)",
                            color_discrete_sequence=['#3b82f6'])
        fig_elbow.add_vline(x=num_clusters, line_dash="dash", line_color="#ef4444", annotation_text=f"Selected k={num_clusters}")
        fig_elbow.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=280
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.caption("Marks diminishing compactness returns as k increases.")

    with col_sc2:
        fig_sil = px.line(eval_df, x='k', y='Silhouette_Score', markers=True, title="Silhouette Score vs k (Higher = Better)",
                          color_discrete_sequence=['#10B981'])
        fig_sil.add_vline(x=num_clusters, line_dash="dash", line_color="#ef4444", annotation_text=f"Selected k={num_clusters}")
        fig_sil.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=280
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        st.caption("Measures cluster cohesion vs separation.")

    with col_sc3:
        fig_db = px.line(eval_df, x='k', y='Davies_Bouldin_Index', markers=True, title="Davies-Bouldin vs k (Lower = Better)",
                         color_discrete_sequence=['#F59E0B'])
        fig_db.add_vline(x=num_clusters, line_dash="dash", line_color="#ef4444", annotation_text=f"Selected k={num_clusters}")
        fig_db.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=280
        )
        st.plotly_chart(fig_db, use_container_width=True)
        st.caption("Evaluates cluster similarity; lower values indicate better distinctness.")

    st.markdown("#### 🌳 Agglomerative Hierarchical Dendrogram")
    fig_dend, ax = plt.subplots(figsize=(10, 3.8))
    fig_dend.patch.set_facecolor('#141b2d')
    ax.set_facecolor('#0f172a')
    ax.tick_params(colors='#cbd5e1', labelsize=8)
    ax.xaxis.label.set_color('#f8fafc')
    ax.yaxis.label.set_color('#f8fafc')
    ax.title.set_color('#60a5fa')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    dendrogram(results['linkage_mat'], truncate_mode='lastp', p=25, leaf_rotation=45, leaf_font_size=9, show_contracted=True, ax=ax)
    ax.set_title("Agglomerative Linkage Dendrogram (Truncated to 25 Branch Nodes)")
    ax.set_xlabel("Aggregated Cluster Leaf / Index")
    ax.set_ylabel("Linkage Merge Distance")
    st.pyplot(fig_dend)

    st.markdown("#### 📊 Multi-k Sweep Summary Table")
    st.dataframe(eval_df, use_container_width=True)

# ------------------------------------------
# TAB 6: EXECUTIVE SUMMARY & EXPORT
# ------------------------------------------
with tab_export:
    st.markdown("""
    <div class="explainer-card">
        <div class="explainer-title">📥 Download Results & Executive Report</div>
        <div class="explainer-text">
            Export the complete enriched challenge dataset with attached AI theme tags and PCA coordinates, or download a publication-ready Executive Analysis report in Markdown format.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("#### 📄 Enriched Dataset (CSV)")
        st.write("Contains all 231 problem statements, assigned theme labels, keywords, and PCA dimensions.")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Dataset (CSV)",
            data=csv_data,
            file_name="societal_challenge_themes_enriched.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        st.markdown("#### 📑 Executive Analysis Report (Markdown)")
        st.write("Summary report containing validation scores, theme breakdowns, and ethical audit findings.")
        
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
