"""
Societal Challenge Theme Discovery from Crowdsourced Problem Statements
Multi-Page Interactive AI Discovery Platform powered by Streamlit, Scikit-Learn, and Plotly.
Features high-contrast glassmorphism, animated ambient mesh gradients, neon glows, hover-enlarge graphs,
and dynamic Dark / Light mode switching on the top right.
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

# Detect current theme state (Dark by default)
if "theme_toggle_key" not in st.session_state:
    st.session_state["theme_toggle_key"] = "🌙 Dark"
is_dark = "Dark" in st.session_state["theme_toggle_key"]

# Plotly & Matplotlib theme configuration variables
plotly_template = "plotly_dark" if is_dark else "plotly_white"
plotly_paper_bg = "rgba(0,0,0,0)"
plotly_plot_bg = "rgba(15, 23, 42, 0.6)" if is_dark else "rgba(248, 250, 252, 0.9)"
plotly_font_color = "#e2e8f0" if is_dark else "#0f172a"
plotly_grid_color = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(0, 0, 0, 0.08)"
plotly_axis_color = "#94a3b8" if is_dark else "#334155"
dend_bg = '#0f172a' if is_dark else '#ffffff'
dend_spine_color = '#334155' if is_dark else '#cbd5e1'
dend_tick_color = '#94a3b8' if is_dark else '#334155'

# Custom Styling (Living Aurora Ambient Mesh, Glassmorphism, Neon Glows & Dual-Theme System)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    :root {{
        --text-color: { "#f8fafc" if is_dark else "#0f172a" } !important;
        --background-color: { "#070b14" if is_dark else "#f8fafc" } !important;
        --secondary-background-color: { "#0f172a" if is_dark else "#f1f5f9" } !important;
        --primary-color: { "#38bdf8" if is_dark else "#2563eb" } !important;
    }}
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* 1. Shifting Ambient Aurora Mesh Background */
    @keyframes auroraMeshDark {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes auroraMeshLight {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .stApp {{
        background: {
            "linear-gradient(135deg, #070b14 0%, #0c162d 25%, #131d3d 50%, #0d172e 75%, #060912 100%) !important;"
            if is_dark else
            "linear-gradient(135deg, #f8fafc 0%, #eef2f6 25%, #e2e8f0 50%, #eef2f6 75%, #f8fafc 100%) !important;"
        }
        background-size: 280% 280% !important;
        animation: { "auroraMeshDark 20s ease-in-out infinite !important;" if is_dark else "auroraMeshLight 20s ease-in-out infinite !important;" }
        color: { "#f8fafc !important;" if is_dark else "#0f172a !important;" }
    }}

    /* 2. Frosted Glass Panels with Backdrop Blur */
    .glass-card {{
        background: { "rgba(15, 23, 42, 0.72) !important;" if is_dark else "rgba(255, 255, 255, 0.88) !important;" }
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: { "1px solid rgba(59, 130, 246, 0.22) !important;" if is_dark else "1px solid rgba(203, 213, 225, 0.85) !important;" }
        border-radius: 16px !important;
        padding: 1.4rem 1.6rem !important;
        box-shadow: {
            "0 10px 30px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;"
            if is_dark else
            "0 10px 25px -5px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 1) !important;"
        }
        transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.28s ease, box-shadow 0.28s ease !important;
    }}
    .glass-card:hover {{
        border-color: { "rgba(56, 189, 248, 0.45) !important;" if is_dark else "#2563eb !important;" }
        box-shadow: {
            "0 16px 40px -8px rgba(56, 189, 248, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;"
            if is_dark else
            "0 16px 35px -8px rgba(37, 99, 235, 0.16) !important;"
        }
        transform: translateY(-3px) !important;
    }}

    /* 3. Hover Enlarge Effects on Graphs */
    .graph-enlarge-container {{
        background: { "rgba(15, 23, 42, 0.75) !important;" if is_dark else "rgba(255, 255, 255, 0.92) !important;" }
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: { "1px solid rgba(59, 130, 246, 0.22) !important;" if is_dark else "1px solid rgba(203, 213, 225, 0.8) !important;" }
        border-radius: 18px !important;
        padding: 1.2rem 1.4rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: {
            "0 12px 35px -8px rgba(0, 0, 0, 0.6) !important;"
            if is_dark else
            "0 10px 30px -8px rgba(0, 0, 0, 0.07) !important;"
        }
        transition: transform 0.32s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.32s ease, border-color 0.32s ease !important;
    }}
    .graph-enlarge-container:hover {{
        transform: translateY(-5px) scale(1.012) !important;
        border-color: { "#38bdf8 !important;" if is_dark else "#2563eb !important;" }
        box-shadow: {
            "0 20px 50px -10px rgba(56, 189, 248, 0.3) !important;"
            if is_dark else
            "0 20px 45px -10px rgba(37, 99, 235, 0.2) !important;"
        }
    }}
    .graph-caption-badge {{
        display: inline-block;
        background: { "rgba(56, 189, 248, 0.15);" if is_dark else "rgba(37, 99, 235, 0.1);" }
        border: { "1px solid rgba(56, 189, 248, 0.35);" if is_dark else "1px solid rgba(37, 99, 235, 0.3);" }
        color: { "#38bdf8;" if is_dark else "#2563eb;" }
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
    }}

    /* 4. Page Entrance Transitions */
    @keyframes pageFadeIn {{
        from {{
            opacity: 0;
            transform: translateY(14px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    .page-wrapper {{
        animation: pageFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}

    /* 5. Rotating 3D Holographic Orbit */
    @keyframes orbit3D {{
        0% {{ transform: rotateY(0deg) rotateX(15deg); }}
        50% {{ transform: rotateY(180deg) rotateX(-15deg); }}
        100% {{ transform: rotateY(360deg) rotateX(15deg); }}
    }}
    .hologram-orb {{
        display: inline-block;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #3b82f6 40%, #1e1b4b 80%, #0f172a 100%);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.7), inset 0 0 15px rgba(255, 255, 255, 0.4);
        animation: orbit3D 10s linear infinite;
        vertical-align: middle;
        margin-right: 12px;
    }}

    /* 6. Neon Pulse Dots & Badges */
    @keyframes neonPulse {{
        0%, 100% {{
            box-shadow: 0 0 6px #34d399, 0 0 14px rgba(16, 185, 129, 0.6);
            opacity: 1;
        }}
        50% {{
            box-shadow: 0 0 14px #34d399, 0 0 26px rgba(16, 185, 129, 0.9);
            opacity: 0.75;
        }}
    }}
    .live-status-dot {{
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #10b981;
        animation: neonPulse 2s infinite ease-in-out;
        margin-right: 8px;
    }}
    .neon-glow-val {{
        color: { "#ffffff !important;" if is_dark else "#1e3a8a !important;" }
        text-shadow: { "0 0 18px rgba(56, 189, 248, 0.5) !important;" if is_dark else "none !important;" }
    }}

    /* 7. Dedicated Page Hero Banners (Bold Navy / Sapphire across both modes for maximum contrast) */
    .page-hero {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 27, 75, 0.9) 45%, rgba(30, 58, 138, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 35px -5px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }}
    .page-hero-badge {{
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
    }}
    .page-hero-title {{
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0.1rem 0 0.5rem 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }}
    .page-hero-subtitle {{
        font-size: 1rem;
        color: #cbd5e1 !important;
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 0;
    }}

    /* 8. KPI Stat Cards */
    .stat-card {{
        background: { "rgba(20, 27, 45, 0.75) !important;" if is_dark else "rgba(255, 255, 255, 0.92) !important;" }
        backdrop-filter: blur(14px) !important;
        border: { "1px solid rgba(51, 65, 85, 0.7) !important;" if is_dark else "1px solid rgba(226, 232, 240, 0.9) !important;" }
        border-radius: 14px !important;
        padding: 1.2rem 1.3rem !important;
        box-shadow: {
            "0 6px 20px rgba(0, 0, 0, 0.35) !important;"
            if is_dark else
            "0 4px 15px rgba(0, 0, 0, 0.04) !important;"
        }
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
        text-align: left !important;
    }}
    .stat-card:hover {{
        transform: translateY(-3px) !important;
        border-color: { "#38bdf8 !important;" if is_dark else "#2563eb !important;" }
        box-shadow: {
            "0 12px 30px rgba(56, 189, 248, 0.2) !important;"
            if is_dark else
            "0 10px 25px rgba(37, 99, 235, 0.14) !important;"
        }
    }}
    .stat-icon {{
        font-size: 1.6rem !important;
        margin-bottom: 0.35rem !important;
    }}
    .stat-label {{
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: { "#94a3b8 !important;" if is_dark else "#64748b !important;" }
        margin-bottom: 0.2rem !important;
    }}
    .stat-value {{
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: { "#ffffff !important;" if is_dark else "#1e3a8a !important;" }
        line-height: 1.1 !important;
    }}
    .stat-caption {{
        font-size: 0.76rem !important;
        color: { "#64748b !important;" if is_dark else "#475569 !important;" }
        margin-top: 0.3rem !important;
    }}

    /* 9. Sidebar Navigation Styling */
    [data-testid="stSidebar"] {{
        background-color: { "rgba(9, 14, 26, 0.95) !important;" if is_dark else "#ffffff !important;" }
        border-right: { "1px solid rgba(59, 130, 246, 0.15) !important;" if is_dark else "1px solid #e2e8f0 !important;" }
    }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
        gap: 6px !important;
    }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
        background: { "rgba(15, 23, 42, 0.6) !important;" if is_dark else "#f8fafc !important;" }
        border: { "1px solid rgba(51, 65, 85, 0.5) !important;" if is_dark else "1px solid #e2e8f0 !important;" }
        border-radius: 10px !important;
        padding: 0.6rem 0.9rem !important;
        transition: all 0.2s ease !important;
        color: { "#e2e8f0 !important;" if is_dark else "#1e293b !important;" }
        cursor: pointer !important;
    }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
        background: { "rgba(59, 130, 246, 0.15) !important;" if is_dark else "rgba(37, 99, 235, 0.08) !important;" }
        border-color: { "rgba(56, 189, 248, 0.4) !important;" if is_dark else "#2563eb !important;" }
        box-shadow: { "0 0 15px rgba(56, 189, 248, 0.15) !important;" if is_dark else "0 0 15px rgba(37, 99, 235, 0.1) !important;" }
        transform: translateX(3px) !important;
    }}

    /* 10. Top Bar Theme Switcher (Dedicated Floating Pill Design) */
    div[data-testid="stRadio"]:has(input[value*="Dark"]),
    div[data-testid="stRadio"]:has(input[value*="Light"]) {{
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        margin-top: 2px !important;
    }}
    div[data-testid="stRadio"]:has(input[value*="Dark"]) [role="radiogroup"],
    div[data-testid="stRadio"]:has(input[value*="Light"]) [role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-end !important;
        gap: 6px !important;
        background: { "rgba(15, 23, 42, 0.7)" if is_dark else "rgba(241, 245, 249, 0.9)" } !important;
        padding: 3px 6px !important;
        border-radius: 9999px !important;
        border: 1px solid { "rgba(59, 130, 246, 0.25)" if is_dark else "rgba(203, 213, 225, 0.9)" } !important;
        box-shadow: { "0 4px 15px rgba(0, 0, 0, 0.3)" if is_dark else "0 2px 8px rgba(0, 0, 0, 0.05)" } !important;
    }}
    div[data-testid="stRadio"]:has(input[value*="Dark"]) [role="radiogroup"] label,
    div[data-testid="stRadio"]:has(input[value*="Light"]) [role="radiogroup"] label {{
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 9999px !important;
        padding: 0.25rem 0.75rem !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        color: { "#cbd5e1" if is_dark else "#475569" } !important;
        box-shadow: none !important;
        transform: none !important;
    }}
    div[data-testid="stRadio"]:has(input[value*="Dark"]) [role="radiogroup"] label:hover,
    div[data-testid="stRadio"]:has(input[value*="Light"]) [role="radiogroup"] label:hover {{
        background: { "rgba(59, 130, 246, 0.25)" if is_dark else "rgba(37, 99, 235, 0.12)" } !important;
        color: { "#38bdf8" if is_dark else "#2563eb" } !important;
    }}

    /* 11. Complete Light / Dark Contrast Overrides for Native Elements */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em,
    .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown strong {{
        color: { "#cbd5e1" if is_dark else "#0f172a" } !important;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5 {{
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
        font-weight: 700 !important;
    }}

    /* Exceptions: Hero banners MUST stay bright white / light blue */
    .page-hero, .page-hero *,
    .page-hero-title, .page-hero-subtitle, .page-hero-badge {{
        color: #ffffff !important;
    }}
    .page-hero .page-hero-subtitle {{
        color: #e2e8f0 !important;
    }}
    .page-hero .page-hero-badge {{
        color: #93c5fd !important;
        background: rgba(59, 130, 246, 0.25) !important;
        border-color: rgba(147, 197, 253, 0.4) !important;
    }}

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
        color: { "#94a3b8" if is_dark else "#475569" } !important;
    }}

    /* Code Tags */
    code, .stMarkdown code, [data-testid="stMarkdownContainer"] code {{
        color: { "#38bdf8" if is_dark else "#1e3a8a" } !important;
        background-color: { "rgba(15, 23, 42, 0.8)" if is_dark else "#e2e8f0" } !important;
        border: 1px solid { "rgba(56, 189, 248, 0.3)" if is_dark else "#cbd5e1" } !important;
        border-radius: 4px !important;
        padding: 0.15rem 0.4rem !important;
        font-size: 0.85em !important;
        font-weight: 600 !important;
    }}

    /* Inputs, Number Inputs, Textareas */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {{
        color: { "#e2e8f0" if is_dark else "#0f172a" } !important;
        font-weight: 600 !important;
    }}
    .stTextInput input, .stNumberInput input {{
        background-color: { "rgba(15, 23, 42, 0.7)" if is_dark else "#ffffff" } !important;
        border: 1px solid { "#334155" if is_dark else "#cbd5e1" } !important;
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
        -webkit-text-fill-color: { "#ffffff" if is_dark else "#0f172a" } !important;
        border-radius: 8px !important;
    }}
    .stTextArea textarea {{
        background-color: { "rgba(15, 23, 42, 0.7)" if is_dark else "#f8fafc" } !important;
        border: 1px solid { "#334155" if is_dark else "#cbd5e1" } !important;
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
        -webkit-text-fill-color: { "#ffffff" if is_dark else "#0f172a" } !important;
        opacity: 1 !important;
        border-radius: 8px !important;
        line-height: 1.5 !important;
    }}
    .stTextArea textarea:disabled,
    textarea[disabled] {{
        background-color: { "rgba(15, 23, 42, 0.6)" if is_dark else "#f1f5f9" } !important;
        border: 1px solid { "#334155" if is_dark else "#cbd5e1" } !important;
        color: { "#cbd5e1" if is_dark else "#0f172a" } !important;
        -webkit-text-fill-color: { "#cbd5e1" if is_dark else "#0f172a" } !important;
        opacity: 1 !important;
        cursor: default !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: { "rgba(15, 23, 42, 0.7)" if is_dark else "#ffffff" } !important;
        border-color: { "#334155" if is_dark else "#cbd5e1" } !important;
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
        border-radius: 8px !important;
    }}
    .stSelectbox div[data-baseweb="select"] * {{
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
    }}
    ul[role="listbox"] {{
        background-color: { "#0f172a" if is_dark else "#ffffff" } !important;
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
    }}
    li[role="option"] {{
        background-color: { "#0f172a" if is_dark else "#ffffff" } !important;
        color: { "#cbd5e1" if is_dark else "#0f172a" } !important;
    }}
    li[role="option"]:hover {{
        background-color: { "#1e293b" if is_dark else "#e2e8f0" } !important;
        color: { "#38bdf8" if is_dark else "#2563eb" } !important;
    }}

    /* Sidebar controls */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {{
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
        font-weight: 800 !important;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span:not(.live-status-dot),
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {{
        color: { "#e2e8f0" if is_dark else "#1e293b" } !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: { "#94a3b8" if is_dark else "#64748b" } !important;
        font-weight: normal !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: { "rgba(51, 65, 85, 0.7)" if is_dark else "#e2e8f0" } !important;
    }}
    [data-testid="stFileUploader"] section {{
        background-color: { "rgba(15, 23, 42, 0.6)" if is_dark else "#f8fafc" } !important;
        border: 2px dashed { "#334155" if is_dark else "#cbd5e1" } !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploader"] section * {{
        color: { "#cbd5e1" if is_dark else "#334155" } !important;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: { "#1e293b" if is_dark else "#ffffff" } !important;
        border: 1px solid { "#334155" if is_dark else "#cbd5e1" } !important;
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
    }}
    [data-testid="stSlider"] div[data-baseweb="slider"] {{
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
    }}
    [data-testid="stSlider"] div[data-testid="stThumbValue"] {{
        color: #ffffff !important;
        background: { "#3b82f6" if is_dark else "#2563eb" } !important;
    }}
    [data-testid="stSlider"] div[data-testid="stTickBar"] {{
        color: { "#94a3b8" if is_dark else "#64748b" } !important;
    }}
    [data-testid="stCheckbox"] label span {{
        color: { "#e2e8f0" if is_dark else "#1e293b" } !important;
    }}

    /* Buttons */
    [data-testid="stDownloadButton"] button,
    [data-testid="baseButton-secondary"] {{
        background: { "linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%)" if is_dark else "linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)" } !important;
        border: 1px solid { "rgba(59, 130, 246, 0.3)" if is_dark else "#cbd5e1" } !important;
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: { "0 4px 15px rgba(0, 0, 0, 0.4)" if is_dark else "0 2px 8px rgba(0, 0, 0, 0.06)" } !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="baseButton-secondary"]:hover {{
        border-color: { "#38bdf8" if is_dark else "#2563eb" } !important;
        color: { "#38bdf8" if is_dark else "#2563eb" } !important;
        box-shadow: { "0 0 20px rgba(56, 189, 248, 0.3)" if is_dark else "0 6px 16px rgba(37, 99, 235, 0.15)" } !important;
        transform: translateY(-2px) !important;
    }}
    [data-testid="baseButton-primary"] {{
        background: { "linear-gradient(135deg, #2563eb 0%, #38bdf8 100%)" if is_dark else "linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%)" } !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: { "0 4px 15px rgba(56, 189, 248, 0.35)" if is_dark else "0 4px 12px rgba(37, 99, 235, 0.25)" } !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="baseButton-primary"]:hover {{
        box-shadow: { "0 0 25px rgba(56, 189, 248, 0.55)" if is_dark else "0 6px 18px rgba(37, 99, 235, 0.4)" } !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }}

    /* Tabs Component Styling */
    div[data-testid="stTabs"] {{
        background: transparent !important;
    }}
    button[data-baseweb="tab"] {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: { "#94a3b8" if is_dark else "#475569" } !important;
        padding: 0.6rem 1.1rem !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2.5px solid transparent !important;
        transition: all 0.2s ease !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: { "#38bdf8" if is_dark else "#1d4ed8" } !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: { "#38bdf8" if is_dark else "#1d4ed8" } !important;
        border-bottom: 2.5px solid { "#38bdf8" if is_dark else "#1d4ed8" } !important;
    }}
    div[data-testid="stTabContent"] {{
        padding-top: 0.8rem !important;
    }}

    /* Dataframe tables */
    [data-testid="stDataFrame"] {{
        background-color: { "rgba(15, 23, 42, 0.6)" if is_dark else "#ffffff" } !important;
        border: 1px solid { "#334155" if is_dark else "#e2e8f0" } !important;
        border-radius: 10px !important;
        box-shadow: { "0 8px 25px rgba(0, 0, 0, 0.3)" if is_dark else "0 4px 12px rgba(0, 0, 0, 0.04)" } !important;
    }}

    /* Alert Boxes */
    div[data-testid="stAlert"] * {{
        color: { "#ffffff" if is_dark else "#0f172a" } !important;
    }}

    /* Theme Showcase Card */
    .theme-card {{
        background: { "rgba(20, 27, 45, 0.78) !important;" if is_dark else "rgba(255, 255, 255, 0.95) !important;" }
        backdrop-filter: blur(14px) !important;
        border: { "1px solid rgba(51, 65, 85, 0.8) !important;" if is_dark else "1px solid rgba(203, 213, 225, 0.95) !important;" }
        border-top: { "4px solid #3b82f6 !important;" if is_dark else "4px solid #2563eb !important;" }
        border-radius: 14px !important;
        padding: 1.4rem 1.6rem !important;
        box-shadow: {
            "0 8px 25px rgba(0, 0, 0, 0.4) !important;"
            if is_dark else
            "0 4px 16px rgba(0, 0, 0, 0.05) !important;"
        }
        margin-bottom: 1.3rem !important;
        transition: transform 0.25s ease, border-color 0.25s ease !important;
    }}
    .theme-card:hover {{
        transform: translateY(-3px) !important;
        border-color: { "#38bdf8 !important;" if is_dark else "#2563eb !important;" }
        box-shadow: {
            "0 14px 35px rgba(56, 189, 248, 0.18) !important;"
            if is_dark else
            "0 12px 30px rgba(37, 99, 235, 0.15) !important;"
        }
    }}

    /* Keyword Pills */
    .keyword-pill {{
        display: inline-block !important;
        background: { "#0f172a !important;" if is_dark else "#f1f5f9 !important;" }
        color: { "#e2e8f0 !important;" if is_dark else "#1e293b !important;" }
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 0.25rem 0.7rem !important;
        border-radius: 6px !important;
        margin-right: 0.45rem !important;
        margin-bottom: 0.45rem !important;
        border: { "1px solid #334155 !important;" if is_dark else "1px solid #cbd5e1 !important;" }
        transition: all 0.18s ease !important;
    }}
    .keyword-pill:hover {{
        border-color: { "#38bdf8 !important;" if is_dark else "#2563eb !important;" }
        color: { "#ffffff !important;" if is_dark else "#0f172a !important;" }
        background: { "#1e293b !important;" if is_dark else "#e2e8f0 !important;" }
        transform: scale(1.04) !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: { "rgba(20, 27, 45, 0.7) !important;" if is_dark else "rgba(255, 255, 255, 0.95) !important;" }
        border: { "1px solid #334155 !important;" if is_dark else "1px solid #cbd5e1 !important;" }
        border-radius: 12px !important;
        box-shadow: { "0 8px 25px rgba(0, 0, 0, 0.3) !important;" if is_dark else "0 4px 15px rgba(0, 0, 0, 0.04)" } !important;
        margin-bottom: 0.9rem !important;
    }}
    [data-testid="stExpander"] summary {{
        font-weight: 700 !important;
        color: { "#e2e8f0 !important;" if is_dark else "#0f172a !important;" }
        padding: 0.8rem 1rem !important;
    }}
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: { "#e2e8f0 !important;" if is_dark else "#0f172a !important;" }
        font-weight: 700 !important;
    }}
    [data-testid="stExpander"] summary svg {{
        fill: { "#e2e8f0 !important;" if is_dark else "#0f172a !important;" }
        stroke: { "#e2e8f0 !important;" if is_dark else "#0f172a !important;" }
    }}
    [data-testid="stExpander"] summary:hover {{
        color: { "#38bdf8 !important;" if is_dark else "#2563eb !important;" }
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: transparent !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {{
        color: { "#cbd5e1 !important;" if is_dark else "#1e293b !important;" }
    }}
</style>
""", unsafe_allow_html=True)

# Add source directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessor import ProblemDataPipeline, clean_text
from src.vectorizer import TextVectorizer
from src.clustering import ClusteringWorkflow
from src.hierarchical import HierarchicalWorkflow
from src.dim_reduction import PCAVisualizer
from src.profiler import ClusterProfiler

def render_styled_table(data_table):
    """Renders a responsive, high-contrast HTML table matching the active Dark/Light theme."""
    bg_th = 'rgba(30, 41, 59, 0.95)' if is_dark else '#e2e8f0'
    text_th = '#38bdf8' if is_dark else '#1e3a8a'
    border_c = '#334155' if is_dark else '#cbd5e1'
    text_td = '#cbd5e1' if is_dark else '#0f172a'
    bg_alt = 'rgba(15, 23, 42, 0.5)' if is_dark else '#f8fafc'
    
    dt = data_table.reset_index() if (hasattr(data_table, 'index') and (data_table.index.name or not isinstance(data_table.index, pd.RangeIndex))) else data_table
    th_cells = "".join([f"<th style='padding: 10px 14px; color: {text_th}; font-weight: 700; border-bottom: 2px solid {border_c}; text-align: left;'>{col}</th>" for col in dt.columns])
    rows_html = []
    for idx, r in dt.iterrows():
        row_bg = bg_alt if idx % 2 == 1 else ('rgba(15, 23, 42, 0.2)' if is_dark else '#ffffff')
        tds = "".join([f"<td style='padding: 9px 14px; color: {text_td}; font-weight: 500;'>{val}</td>" for val in r])
        rows_html.append(f"<tr style='background: {row_bg}; border-bottom: 1px solid {border_c};'>{tds}</tr>")
    
    full_html = f"<div style='overflow-x: auto; margin: 0.8rem 0; border-radius: 10px; border: 1px solid {border_c}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'><table style='width: 100%; border-collapse: collapse; font-size: 0.86rem; font-family: inherit;'><thead><tr style='background: {bg_th};'>{th_cells}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    st.markdown(full_html, unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div class="hologram-orb"></div>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: {'#ffffff' if is_dark else '#0f172a'}; letter-spacing: -0.01em;">Societal AI</div>
            <div style="font-size: 0.75rem; color: {'#94a3b8' if is_dark else '#64748b'};">Theme Discovery Engine</div>
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

    st.markdown(f"""
    <div style="margin-top: 1.5rem; padding: 0.6rem 0.8rem; background: {'rgba(16, 185, 129, 0.1)' if is_dark else 'rgba(16, 185, 129, 0.08)'}; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px;">
        <span class="live-status-dot"></span>
        <span style="color: {'#34d399' if is_dark else '#059669'}; font-weight: 700; font-size: 0.8rem;">Model Engine Active</span>
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
        'feature_names': feature_names,
        'vectorizer': vectorizer,
        'centroids': centroids,
        'X': X
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
vectorizer = results['vectorizer']
centroids = results['centroids']
X_matrix = results['X']
feature_names = results['feature_names']

# ==========================================
# TOP BAR: ENGINE BADGE & THEME TOGGLE (TOP RIGHT)
# ==========================================
col_top_left, col_top_right = st.columns([5, 1.5])

with col_top_left:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; padding-top: 6px; margin-bottom: 0.6rem;">
        <span class="live-status-dot"></span>
        <span style="font-size: 0.8rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: {'#34d399' if is_dark else '#059669'};">
            AI CLUSTERING PLATFORM
        </span>
        <span style="color: {'#475569' if is_dark else '#94a3b8'};">•</span>
        <span style="font-size: 0.8rem; font-weight: 600; color: {'#94a3b8' if is_dark else '#64748b'};">
            {selected_page}
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_top_right:
    theme_choice = st.radio(
        "Theme Mode",
        ["🌙 Dark", "☀️ Light"],
        horizontal=True,
        label_visibility="collapsed",
        key="theme_toggle_key"
    )

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
        tab_missions, tab_classifier = st.tabs(["🌟 Discovered Missions", "⚡ Live Challenge Classifier"])
        
        with tab_missions:
            c_header_color = '#60a5fa' if is_dark else '#1d4ed8'
            c_sub_color = '#94a3b8' if is_dark else '#64748b'
            c_badge_bg = 'rgba(56, 189, 248, 0.15)' if is_dark else 'rgba(37, 99, 235, 0.1)'
            c_badge_text = '#38bdf8' if is_dark else '#1d4ed8'
            c_badge_border = 'rgba(56, 189, 248, 0.3)' if is_dark else 'rgba(37, 99, 235, 0.2)'
            
            top_bar_html = f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
<div>
<h3 style="color: {c_header_color}; margin: 0; font-size: 1.15rem; font-weight: 800;">🌟 Discovered Societal Missions</h3>
<div style="color: {c_sub_color}; font-size: 0.8rem; margin-top: 2px;">Autonomous thematic clusters discovered across crowdsourced national challenges</div>
</div>
<span style="font-size: 0.75rem; font-weight: 700; background: {c_badge_bg}; color: {c_badge_text}; padding: 4px 12px; border-radius: 20px; border: 1px solid {c_badge_border}; white-space: nowrap;">{num_clusters} Core Missions</span>
</div>"""
            st.html(top_bar_html)

            # Generate individual mission cards
            card_items = []
            for k in range(num_clusters):
                prof = profiles[k]
                c_icon = prof.get('icon', '🌐')
                c_title = prof.get('theme_title', f"Mission {k + 1}")
                c_count = int((df['Cluster_ID'] == k).sum())
                c_pct = (c_count / len(df) * 100) if len(df) > 0 else 0
                c_kws = [f['term'] for f in prof['top_features'][:5]]
                
                # Colors
                card_bg = 'rgba(15, 23, 42, 0.65)' if is_dark else '#ffffff'
                card_border = '#334155' if is_dark else '#e2e8f0'
                title_color = '#f8fafc' if is_dark else '#0f172a'
                accent_bar = 'linear-gradient(90deg, #38bdf8, #818cf8)' if is_dark else 'linear-gradient(90deg, #2563eb, #0284c7)'
                track_bg = 'rgba(51, 65, 85, 0.4)' if is_dark else '#e2e8f0'
                pill_bg = '#1e293b' if is_dark else '#f1f5f9'
                pill_text = '#e2e8f0' if is_dark else '#1e293b'
                pill_border = '#334155' if is_dark else '#cbd5e1'
                
                pills_html = "".join([f'<span style="display: inline-block; background: {pill_bg}; color: {pill_text}; font-size: 0.72rem; font-weight: 600; padding: 2px 7px; border-radius: 6px; margin-right: 4px; margin-bottom: 3px; border: 1px solid {pill_border};">{kw}</span>' for kw in c_kws])
                
                card_html = f"""<div style="background: {card_bg}; padding: 12px 14px; border-radius: 10px; border: 1px solid {card_border}; margin-bottom: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<div style="display: flex; align-items: center; gap: 8px;">
<span style="font-size: 1.25rem;">{c_icon}</span>
<span style="font-weight: 700; font-size: 0.9rem; color: {title_color};">{c_title}</span>
</div>
<span style="background: {c_badge_bg}; color: {c_badge_text}; font-weight: 700; font-size: 0.74rem; padding: 2px 8px; border-radius: 6px; border: 1px solid {c_badge_border}; white-space: nowrap;">{c_count} challenges ({c_pct:.1f}%)</span>
</div>
<div style="width: 100%; height: 5px; background: {track_bg}; border-radius: 3px; overflow: hidden; margin-bottom: 7px;">
<div style="width: {c_pct}%; height: 100%; background: {accent_bar}; border-radius: 3px;"></div>
</div>
<div style="display: flex; flex-wrap: wrap; align-items: center;">
<span style="font-size: 0.72rem; color: {c_sub_color}; font-weight: 600; margin-right: 6px;">Key Signals:</span>
{pills_html}
</div>
</div>"""
                card_items.append(card_html)
            
            missions_container_html = f"""<div style="max-height: 420px; overflow-y: auto; padding-right: 4px;">{"".join(card_items)}</div>"""
            st.html(missions_container_html)

        with tab_classifier:
            c_header_color = '#60a5fa' if is_dark else '#1d4ed8'
            c_sub_color = '#94a3b8' if is_dark else '#64748b'
            
            st.html(f"""<div style="margin-bottom: 10px;">
<h3 style="color: {c_header_color}; margin: 0; font-size: 1.15rem; font-weight: 800;">⚡ Live Challenge Classifier</h3>
<div style="color: {c_sub_color}; font-size: 0.8rem; margin-top: 2px;">Test real-time thematic prediction and nearest challenge matching for any civic idea</div>
</div>""")
            
            sample_presets = {
                "🛰️ Satellite Wildfire & Disaster Monitoring": "AI-powered satellite imagery and infrared thermal sensors for early forest fire detection and rapid emergency dispatch in mountain ranges.",
                "🏥 Rural AI Telemedicine & Clinical Triage": "Solar-powered telemedicine kiosk with AI symptom triage, diagnostic vital sensors, and vernacular doctor consultation for remote tribal villages.",
                "🌾 IoT Smart Crop Irrigation & Moisture Sensing": "IoT soil moisture and nutrient sensor network with automated drip irrigation scheduling and crop disease alerts for drought-prone farmland.",
                "🚦 Urban Adaptive Traffic Light Synchronization": "Computer vision traffic cameras for real-time adaptive traffic light signal timing and green corridor routing for emergency ambulances.",
                "✏️ Custom Statement (Type your own)": ""
            }
            
            def on_scenario_select():
                sel = st.session_state["sandbox_preset_dropdown"]
                st.session_state["sandbox_text_area"] = sample_presets[sel]
            
            if "sandbox_text_area" not in st.session_state:
                st.session_state["sandbox_text_area"] = sample_presets["🛰️ Satellite Wildfire & Disaster Monitoring"]
            
            st.selectbox(
                "💡 Select an Example Problem Statement (or select Custom):",
                options=list(sample_presets.keys()),
                key="sandbox_preset_dropdown",
                on_change=on_scenario_select
            )
            
            user_statement = st.text_area(
                "Problem Statement to Classify:",
                key="sandbox_text_area",
                height=75,
                placeholder="Type or paste any problem statement..."
            )
            
            # Real-time AI classification
            if user_statement and user_statement.strip():
                clean_stmt = clean_text(user_statement, remove_domain_stopwords=remove_domain_stops)
                if not clean_stmt.strip():
                    clean_stmt = user_statement.lower()
                
                stmt_vec = vectorizer.transform([clean_stmt])
                
                if metric_key == 'manhattan':
                    sb_dists = np.sum(np.abs(centroids - stmt_vec), axis=1)
                else:
                    sb_dists = np.linalg.norm(centroids - stmt_vec, axis=1)
                
                pred_cluster = int(np.argmin(sb_dists))
                
                inv_d = 1.0 / (sb_dists + 1e-6)
                base_pct = float((inv_d[pred_cluster] / np.sum(inv_d)) * 100)
                pred_conf = min(98.8, max(64.0, 52.0 + (base_pct * 1.4)))
                
                pred_prof = profiles[pred_cluster]
                pred_icon = pred_prof.get('icon', '🌐')
                pred_theme_title = pred_prof.get('theme_title', f"Theme {pred_cluster + 1}")
                
                stmt_words = set(clean_stmt.lower().split())
                cluster_top_terms = [t['term'].lower() for t in pred_prof['top_features']]
                matched_kws = [t for t in cluster_top_terms if any(w in t for w in stmt_words)][:4]
                if not matched_kws:
                    matched_kws = cluster_top_terms[:3]
                
                pill_bg = '#1e293b' if is_dark else '#f1f5f9'
                pill_text = '#e2e8f0' if is_dark else '#1e293b'
                pill_border = '#334155' if is_dark else '#cbd5e1'
                matched_pills = "".join([f'<span style="display: inline-block; background: {pill_bg}; color: {pill_text}; font-size: 0.72rem; padding: 2px 7px; margin-right: 4px; margin-bottom: 3px; border-radius: 6px; border: 1px solid {pill_border};">{m}</span>' for m in matched_kws])
                
                ds_dists = np.linalg.norm(X_matrix - stmt_vec, axis=1)
                nearest_idx = np.argsort(ds_dists)[:2]
                nearest_1 = df.iloc[nearest_idx[0]]
                nearest_2 = df.iloc[nearest_idx[1]] if len(nearest_idx) > 1 else None
                
                n1_title = str(nearest_1['Problem Statement Title'])
                if len(n1_title) > 65:
                    n1_title = n1_title[:62] + "..."
                n1_org = str(nearest_1['Organization'])
                if len(n1_org) > 35:
                    n1_org = n1_org[:32] + "..."
                
                n2_html = ""
                if nearest_2 is not None:
                    n2_title = str(nearest_2['Problem Statement Title'])
                    if len(n2_title) > 65:
                        n2_title = n2_title[:62] + "..."
                    n2_org = str(nearest_2['Organization'])
                    if len(n2_org) > 35:
                        n2_org = n2_org[:32] + "..."
                    sub_item_color = '#94a3b8' if is_dark else '#334155'
                    n2_html = f'<div style="font-size: 0.76rem; color: {sub_item_color}; margin-top: 3px;">• <b>#{nearest_2["Problem Statement ID"]}</b> ({n2_org}): <i>"{n2_title}"</i></div>'
                
                res_bg = 'rgba(15, 23, 42, 0.85)' if is_dark else '#ffffff'
                res_border = '#38bdf8' if is_dark else '#2563eb'
                res_shadow = '0 6px 20px rgba(56, 189, 248, 0.15)' if is_dark else '0 4px 14px rgba(37, 99, 235, 0.1)'
                res_title_color = '#ffffff' if is_dark else '#0f172a'
                box_inner_bg = 'rgba(30, 41, 59, 0.5)' if is_dark else '#f1f5f9'
                box_inner_border = '#334155' if is_dark else '#cbd5e1'
                box_inner_text = '#cbd5e1' if is_dark else '#1e293b'
                
                result_card_html = f"""<div style="background: {res_bg}; border: 1.5px solid {res_border}; border-radius: 10px; padding: 12px 14px; margin-top: 8px; box-shadow: {res_shadow};">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<span style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; color: {res_border};">🎯 Assigned Thematic Cluster</span>
<span style="font-size: 0.73rem; font-weight: 700; background: {'rgba(16, 185, 129, 0.15)' if is_dark else 'rgba(16, 185, 129, 0.1)'}; color: {'#34d399' if is_dark else '#059669'}; padding: 2px 8px; border-radius: 6px; border: 1px solid {'rgba(16, 185, 129, 0.3)' if is_dark else 'rgba(16, 185, 129, 0.2)'};">{pred_conf:.1f}% Confidence</span>
</div>
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
<span style="font-size: 1.45rem;">{pred_icon}</span>
<div>
<div style="font-size: 0.96rem; font-weight: 800; color: {res_title_color};">{pred_theme_title}</div>
<div style="font-size: 0.74rem; color: {c_sub_color};">Closest centroid: Theme {pred_cluster + 1}</div>
</div>
</div>
<div style="margin-bottom: 6px;">
<div style="font-size: 0.72rem; color: {c_sub_color}; font-weight: 600; margin-bottom: 3px;">Matched Vocabulary Signals:</div>
<div style="display: flex; flex-wrap: wrap;">{matched_pills}</div>
</div>
<div style="background: {box_inner_bg}; border-radius: 8px; padding: 8px 10px; border: 1px solid {box_inner_border};">
<div style="font-size: 0.73rem; font-weight: 700; color: {res_border}; margin-bottom: 2px;">Closest Verified Submissions in Dataset:</div>
<div style="font-size: 0.76rem; color: {box_inner_text};">• <b>#{nearest_1["Problem Statement ID"]}</b> ({n1_org}): <i>"{n1_title}"</i></div>
{n2_html}
</div>
</div>"""
                st.html(result_card_html)

    with col_ov2:
        st.markdown('<div class="graph-enlarge-container">', unsafe_allow_html=True)
        st.markdown('<div class="graph-caption-badge">Interactive Distribution • Hover to Enlarge</div>', unsafe_allow_html=True)
        st.markdown(f"#### 🏛️ Top Participating Government Bodies")
        top_orgs_overview = df['Organization'].value_counts().head(8).reset_index()
        top_orgs_overview.columns = ['Organization', 'Count']
        fig_ov_bar = px.bar(
            top_orgs_overview, x='Count', y='Organization', orientation='h',
            color='Count', color_continuous_scale='Blues'
        )
        fig_ov_bar.update_layout(
            template=plotly_template,
            paper_bgcolor=plotly_paper_bg,
            plot_bgcolor=plotly_plot_bg,
            font=dict(color=plotly_font_color, family="Plus Jakarta Sans, sans-serif"),
            xaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            yaxis=dict(categoryorder='total ascending', title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            coloraxis=dict(colorbar=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color))),
            height=420,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_ov_bar, use_container_width=True, theme=None)
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
                <div style="font-size: 1.35rem; font-weight: 800; color: {'#ffffff' if is_dark else '#0f172a'};">
                    {icon} Theme {k_idx + 1}: {title}
                </div>
                <div style="background: {'rgba(56, 189, 248, 0.15)' if is_dark else 'rgba(37, 99, 235, 0.1)'}; border: 1px solid {'rgba(56, 189, 248, 0.4)' if is_dark else 'rgba(37, 99, 235, 0.3)'}; padding: 0.35rem 0.85rem; border-radius: 9999px; font-weight: 700; color: {'#38bdf8' if is_dark else '#2563eb'}; font-size: 0.84rem;">
                    {cluster_count} Challenges ({cluster_pct:.1f}% of national pool)
                </div>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {'#94a3b8' if is_dark else '#64748b'}; margin-right: 0.6rem;">Top Characteristic Vocabulary:</span>
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
                    <div style="background: {'rgba(15, 23, 42, 0.5)' if is_dark else '#f8fafc'}; border-left: 3px solid {'#38bdf8' if is_dark else '#2563eb'}; padding: 0.6rem 0.9rem; margin-bottom: 0.6rem; border-radius: 0 8px 8px 0;">
                        <div style="font-size: 0.76rem; color: {'#94a3b8' if is_dark else '#64748b'}; font-weight: 600;">[{ex['id']}] • {ex['org']} • Modality: {ex['category']} • Distance: {ex['distance_to_center']}</div>
                        <div style="font-size: 0.9rem; color: {'#f1f5f9' if is_dark else '#0f172a'}; font-weight: 600; margin-top: 0.2rem;">{ex['title']}</div>
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
    <div style="background: {'rgba(15, 23, 42, 0.6)' if is_dark else 'rgba(255, 255, 255, 0.8)'}; border: 1px solid {'rgba(59, 130, 246, 0.2)' if is_dark else 'rgba(203, 213, 225, 0.8)'}; border-radius: 8px; padding: 0.5rem 0.9rem; margin-bottom: 1.2rem; display: inline-block;">
        <span style="color: {'#38bdf8' if is_dark else '#2563eb'}; font-weight: 700;">Found {len(filtered_view)}</span> <span style="color: {'#94a3b8' if is_dark else '#64748b'};">matching problem statements</span>
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
            st.markdown(f"""
            <div style="color: {'#cbd5e1' if is_dark else '#1e293b'}; font-size: 0.88rem; margin-bottom: 0.4rem;">
                <b>🏛️ Submitting Body:</b> <code>{org}</code> &nbsp;|&nbsp; <b>Modality:</b> <code>{row['Category']}</code> &nbsp;|&nbsp; <b>Original Tag:</b> <code>{row['Theme']}</code>
            </div>
            <div style="color: {'#cbd5e1' if is_dark else '#1e293b'}; font-size: 0.88rem; margin-bottom: 0.6rem;">
                <b>AI Discovered Theme:</b> <span style="color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-weight: 700;">{icon} {theme_assigned}</span>
            </div>
            <div style="color: {'#94a3b8' if is_dark else '#475569'}; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.3rem;">
                Full Problem Description:
            </div>
            """, unsafe_allow_html=True)
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
            <div style="font-size: 0.76rem; font-weight: 700; color: {'#94a3b8' if is_dark else '#64748b'}; text-transform: uppercase;">PCA Explained Variance</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: {'#38bdf8' if is_dark else '#2563eb'}; margin: 0.3rem 0;">{results['pca_exp_var']['total_retained_variance']}%</div>
            <div style="font-size: 0.76rem; color: {'#64748b' if is_dark else '#475569'};">Retained in 3 components</div>
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
                    marker=dict(symbol='diamond', size=16, color='#38bdf8' if is_dark else '#2563eb', line=dict(width=2, color='white')),
                    text=[f"Theme {i+1} Center" for i in range(len(cent))],
                    textposition="top center",
                    name="Theme Centers"
                ))
            fig_galaxy.update_layout(
                template=plotly_template,
                paper_bgcolor=plotly_paper_bg,
                plot_bgcolor=plotly_plot_bg,
                font=dict(color=plotly_font_color, family="Plus Jakarta Sans, sans-serif"),
                title=dict(font=dict(color=plotly_font_color, size=16)),
                legend=dict(
                    font=dict(color=plotly_font_color),
                    title=dict(font=dict(color=plotly_font_color)),
                    orientation="h", yanchor="bottom", y=-0.35
                ),
                xaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
                yaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
                height=560
            )
            st.plotly_chart(fig_galaxy, use_container_width=True, theme=None)
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
                    marker=dict(symbol='diamond', size=10, color='#38bdf8' if is_dark else '#2563eb', line=dict(width=2, color='white')),
                    text=[f"Center {i+1}" for i in range(len(cent))],
                    textposition="top center",
                    name="Theme Centers"
                ))
            fig_galaxy3d.update_layout(
                template=plotly_template,
                paper_bgcolor=plotly_paper_bg,
                font=dict(color=plotly_font_color, family="Plus Jakarta Sans, sans-serif"),
                title=dict(font=dict(color=plotly_font_color, size=16)),
                legend=dict(font=dict(color=plotly_font_color), title=dict(font=dict(color=plotly_font_color))),
                scene=dict(
                    xaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), backgroundcolor=plotly_plot_bg, gridcolor=plotly_grid_color),
                    yaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), backgroundcolor=plotly_plot_bg, gridcolor=plotly_grid_color),
                    zaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), backgroundcolor=plotly_plot_bg, gridcolor=plotly_grid_color)
                ),
                height=620
            )
            st.plotly_chart(fig_galaxy3d, use_container_width=True, theme=None)
        
        st.caption("Each cluster forms a distinct semantic galaxy. Hover over any node to inspect the challenge title, department, and PCA coordinates.")
        st.markdown('</div>', unsafe_allow_html=True)

    # PCA Component Loadings Interpretation Table
    st.markdown("#### 📐 PCA Geometric Axis Interpretation (Top Word Loadings)")
    loadings = results['pca_loadings']
    col_l1, col_l2, col_l3 = st.columns(3)
    
    with col_l1:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.2rem;">
            <div style="font-weight: 700; color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-size: 0.95rem;">Principal Component 1 (PC1)</div>
            <div style="font-size: 0.78rem; color: {'#94a3b8' if is_dark else '#475569'}; margin-bottom: 0.6rem;">Primary semantic variance axis</div>
            <div style="color: {'#e2e8f0' if is_dark else '#0f172a'}; font-size: 0.84rem; font-weight: 700; margin-bottom: 0.25rem;">Top Positive Features:</div>
            <div style="color: {'#cbd5e1' if is_dark else '#334155'}; font-size: 0.82rem; margin-bottom: 0.6rem;">{", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC1']['positive'][:4]])}</div>
            <div style="color: {'#e2e8f0' if is_dark else '#0f172a'}; font-size: 0.84rem; font-weight: 700; margin-bottom: 0.25rem;">Top Negative Features:</div>
            <div style="color: {'#cbd5e1' if is_dark else '#334155'}; font-size: 0.82rem;">{", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC1']['negative'][:4]])}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_l2:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.2rem;">
            <div style="font-weight: 700; color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-size: 0.95rem;">Principal Component 2 (PC2)</div>
            <div style="font-size: 0.78rem; color: {'#94a3b8' if is_dark else '#475569'}; margin-bottom: 0.6rem;">Orthogonal variance axis</div>
            <div style="color: {'#e2e8f0' if is_dark else '#0f172a'}; font-size: 0.84rem; font-weight: 700; margin-bottom: 0.25rem;">Top Positive Features:</div>
            <div style="color: {'#cbd5e1' if is_dark else '#334155'}; font-size: 0.82rem; margin-bottom: 0.6rem;">{", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC2']['positive'][:4]])}</div>
            <div style="color: {'#e2e8f0' if is_dark else '#0f172a'}; font-size: 0.84rem; font-weight: 700; margin-bottom: 0.25rem;">Top Negative Features:</div>
            <div style="color: {'#cbd5e1' if is_dark else '#334155'}; font-size: 0.82rem;">{", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC2']['negative'][:4]])}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_l3:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.2rem;">
            <div style="font-weight: 700; color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-size: 0.95rem;">Principal Component 3 (PC3)</div>
            <div style="font-size: 0.78rem; color: {'#94a3b8' if is_dark else '#475569'}; margin-bottom: 0.6rem;">Tertiary depth dimension</div>
            <div style="color: {'#e2e8f0' if is_dark else '#0f172a'}; font-size: 0.84rem; font-weight: 700; margin-bottom: 0.25rem;">Top Positive Features:</div>
            <div style="color: {'#cbd5e1' if is_dark else '#334155'}; font-size: 0.82rem; margin-bottom: 0.6rem;">{", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC3']['positive'][:4]])}</div>
            <div style="color: {'#e2e8f0' if is_dark else '#0f172a'}; font-size: 0.84rem; font-weight: 700; margin-bottom: 0.25rem;">Top Negative Features:</div>
            <div style="color: {'#cbd5e1' if is_dark else '#334155'}; font-size: 0.82rem;">{", ".join([f"{w[0]} ({w[1]:.2f})" for w in loadings['PC3']['negative'][:4]])}</div>
        </div>
        """, unsafe_allow_html=True)

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
            template=plotly_template,
            paper_bgcolor=plotly_paper_bg,
            plot_bgcolor=plotly_plot_bg,
            font=dict(color=plotly_font_color, family="Plus Jakarta Sans, sans-serif"),
            xaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            yaxis=dict(categoryorder='total ascending', title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            coloraxis=dict(colorbar=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color))),
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_org_bar, use_container_width=True, theme=None)
        st.caption("Highlights institutional skew: a handful of public bodies account for a major share of submissions.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cp2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {'#38bdf8' if is_dark else '#1d4ed8'}; margin-top: 0;'>2. Resolving Vague Catch-All Tags</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color: {'#cbd5e1' if is_dark else '#1e293b'}; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0.8rem;">
            In raw crowdsourced datasets, generic catch-all labels are pervasive.
            Unsupervised NLP disambiguates these generic tags into concrete technical clusters:
        </div>
        """, unsafe_allow_html=True)
        vague_df = df[df['Theme'].isin(['Smart Automation', 'Miscellaneous'])]
        if len(vague_df) > 0:
            regroup_counts = vague_df.groupby(['Theme', 'Theme_Title']).size().unstack(fill_value=0)
            render_styled_table(regroup_counts)
            st.caption("Demonstrates how NLP reveals specific domain clusters within vague labels.")
        else:
            top_themes = df['Theme'].value_counts().head(5).index.tolist()
            subset_df = df[df['Theme'].isin(top_themes)]
            if len(subset_df) > 0:
                regroup_counts = subset_df.groupby(['Theme', 'Theme_Title']).size().unstack(fill_value=0)
                render_styled_table(regroup_counts)
                st.caption("Cross-tabulation of discovered AI themes against top source themes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🚨 Responsible AI & Ethical Boundaries")
    col_eth1, col_eth2 = st.columns(2, gap="large")
    
    with col_eth1:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #f59e0b;">
            <div style="font-weight: 700; color: #fbbf24; font-size: 0.95rem; margin-bottom: 0.4rem;">⚠️ Exploratory Discovery vs. Rigid Dogma</div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: {'#cbd5e1' if is_dark else '#1e293b'};">
                Unsupervised clusters are mathematically discovered groupings based on lexical co-occurrence. 
                They serve as discovery lenses for policy makers and evaluators, not as permanent or exclusive real-world boundaries.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_eth2:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {'#38bdf8' if is_dark else '#2563eb'};">
            <div style="font-weight: 700; color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-size: 0.95rem; margin-bottom: 0.4rem;">ℹ️ Linguistic & Lexical Polysemy</div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: {'#cbd5e1' if is_dark else '#1e293b'};">
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
        fig_wcss.update_layout(
            template=plotly_template, paper_bgcolor=plotly_paper_bg, plot_bgcolor=plotly_plot_bg,
            font=dict(color=plotly_font_color), title_font=dict(color=plotly_font_color),
            xaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            yaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            height=260, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_wcss, use_container_width=True, theme=None)
        st.caption("Diminishing returns elbow inspection.")

    with col_ev2:
        fig_sil = px.line(eval_df, x='k', y='Silhouette_Score', markers=True, title="Silhouette Score Curve")
        fig_sil.update_layout(
            template=plotly_template, paper_bgcolor=plotly_paper_bg, plot_bgcolor=plotly_plot_bg,
            font=dict(color=plotly_font_color), title_font=dict(color=plotly_font_color),
            xaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            yaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            height=260, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_sil, use_container_width=True, theme=None)
        st.caption("Cohesion vs. separation (higher is better).")

    with col_ev3:
        fig_db = px.line(eval_df, x='k', y='Davies_Bouldin_Index', markers=True, title="Davies-Bouldin Index")
        fig_db.update_layout(
            template=plotly_template, paper_bgcolor=plotly_paper_bg, plot_bgcolor=plotly_plot_bg,
            font=dict(color=plotly_font_color), title_font=dict(color=plotly_font_color),
            xaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            yaxis=dict(title_font=dict(color=plotly_font_color), tickfont=dict(color=plotly_font_color), gridcolor=plotly_grid_color),
            height=260, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_db, use_container_width=True, theme=None)
        st.caption("Cluster similarity index (lower is better).")

    st.markdown('</div>', unsafe_allow_html=True)

    # Hierarchical Clustering Dendrogram
    col_h1, col_h2 = st.columns([1.3, 1], gap="large")
    
    with col_h1:
        st.markdown('<div class="graph-enlarge-container">', unsafe_allow_html=True)
        st.markdown('<div class="graph-caption-badge">Taxonomic Hierarchy</div>', unsafe_allow_html=True)
        st.markdown("#### 🌳 Agglomerative Hierarchical Linkage Dendrogram")
        fig_dend, ax_dend = plt.subplots(figsize=(10, 4.2))
        fig_dend.patch.set_facecolor(dend_bg)
        ax_dend.set_facecolor(dend_bg)
        
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
        ax_dend.tick_params(colors=dend_tick_color)
        for spine in ax_dend.spines.values():
            spine.set_color(dend_spine_color)
        st.pyplot(fig_dend)
        st.caption("Bottom-up hierarchical dendrogram showing how challenge themes merge at increasing distance thresholds.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_h2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {'#38bdf8' if is_dark else '#1d4ed8'}; margin-top: 0;'>🎯 Stability & Concordance</h4>", unsafe_allow_html=True)
        stability_res = results['stability']
        st.markdown(f"""
        <div style="color: {'#cbd5e1' if is_dark else '#1e293b'}; font-size: 0.9rem; line-height: 1.8;">
            <div>• <b>Mean Adjusted Rand Index (ARI):</b> <code style="color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-weight: 700;">{stability_res['mean_ari']:.4f}</code></div>
            <div>• <b>Stability Status:</b> <code style="color: {'#34d399' if is_dark else '#059669'}; font-weight: 700;">{"Very High (>0.6)" if stability_res['mean_ari'] > 0.6 else "Moderate"}</code></div>
            <div>• <b>Hierarchical vs K-Means ARI:</b> <code style="color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-weight: 700;">{results['hier_comp']['adjusted_rand_index']:.4f}</code></div>
            <div>• <b>Hierarchical vs K-Means NMI:</b> <code style="color: {'#38bdf8' if is_dark else '#1d4ed8'}; font-weight: 700;">{results['hier_comp']['normalized_mutual_info']:.4f}</code></div>
        </div>
        <div style="font-weight: 700; color: {'#e2e8f0' if is_dark else '#0f172a'}; margin-top: 0.8rem; margin-bottom: 0.4rem;">
            Contingency Cross-Tabulation:
        </div>
        """, unsafe_allow_html=True)
        contingency_df = results['hier_comp'].get('contingency_matrix', results['hier_comp'].get('contingency_table'))
        if contingency_df is not None:
            render_styled_table(contingency_df)
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
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: {'#38bdf8' if is_dark else '#1d4ed8'}; margin-top: 0;">📊 Clustered Dataset (.csv)</h4>
            <p style="color: {'#cbd5e1' if is_dark else '#1e293b'}; font-size: 0.9rem;">
                Includes all original problem statement fields plus:
            </p>
            <ul style="color: {'#94a3b8' if is_dark else '#334155'}; font-size: 0.86rem; line-height: 1.8;">
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
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: {'#38bdf8' if is_dark else '#1d4ed8'}; margin-top: 0;">📑 Executive Summary Report (.md)</h4>
            <p style="color: {'#cbd5e1' if is_dark else '#1e293b'}; font-size: 0.9rem; line-height: 1.6;">
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
    render_styled_table(df[preview_cols].head(15))

    st.markdown('</div>', unsafe_allow_html=True)
