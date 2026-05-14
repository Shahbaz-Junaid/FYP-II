import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy import stats
from ydata_profiling import ProfileReport
import joblib
from fpdf import FPDF


import tempfile
import os
import plotly.io as pio

# =========================================================
# === 1. PAGE CONFIGURATION & STYLING =====================
# =========================================================

st.set_page_config(
    page_title="DataMate AI - Your AI Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── ENHANCED CSS: Responsive + Animated ─────────────────────────────────────
st.markdown("""
<style>
    /* ================= GOOGLE FONT IMPORT ================= */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    /* ================= KEYFRAME ANIMATIONS ================= */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(52,152,219,0.4); }
        50%       { box-shadow: 0 0 0 8px rgba(52,152,219,0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-6px); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.8); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes borderPulse {
        0%, 100% { border-color: #3498db; }
        50%       { border-color: #2ecc71; }
    }
    @keyframes sidebarFadeIn {
        from { opacity: 0; transform: translateX(-15px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* ================= GLOBAL FONT ================= */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2535 100%) !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(174,214,241,0.2);
        color: #cdd9e5 !important;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600 !important;
        font-size: 14px;
        border-radius: 10px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
        padding: 11px 16px;
        text-align: left;
        margin-bottom: 6px;
        animation: sidebarFadeIn 0.4s ease both;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateX(6px);
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border: 1px solid #2980b9 !important;
        box-shadow: 0 4px 18px rgba(52,152,219,0.45);
    }
    section[data-testid="stSidebar"] .stButton button:active {
        transform: scale(0.97);
    }

    /* ================= MAIN HEADER ================= */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 22px 28px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 24px;
        font-family: 'Sora', sans-serif;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        animation: fadeInUp 0.6s ease both;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        animation: shimmer 3s infinite;
    }

    /* ================= SIDEBAR HEADER CARD ================= */
    .sidebar-header-card {
        background: linear-gradient(135deg, #3498db, #1a6dad) !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 12px !important;
        margin-bottom: 18px !important;
        text-align: center !important;
        box-shadow: 0 4px 20px rgba(52,152,219,0.4) !important;
        animation: pulse-glow 3s infinite;
    }

    /* ================= METRIC CARDS ================= */
    .metric-card {
        background: white;
        padding: 20px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
        margin: 10px 0;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.5s ease both;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 12px 32px rgba(52,152,219,0.18);
    }
    .metric-card p {
        animation: countUp 0.6s ease both;
    }

    /* ================= SECTION HEADER ================= */
    .section-header {
        font-family: 'Sora', sans-serif;
        font-size: 19px;
        font-weight: 700;
        color: #1a2535;
        margin: 22px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #3498db;
        animation: slideInLeft 0.5s ease both;
    }

    /* ================= CONTENT FRAME ================= */
    .content-frame {
        border: 1.5px solid #dee2e6;
        border-radius: 12px;
        padding: 22px;
        margin: 12px 0;
        background: white;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        animation: fadeInUp 0.5s ease both;
    }

    /* ================= CHART CONTAINER ================= */
    .chart-container {
        background: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 12px 0;
        transition: box-shadow 0.2s;
        animation: fadeInUp 0.5s ease both;
    }
    .chart-container:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    }

    /* ================= UPLOAD SECTION ================= */
    .upload-section {
        background: #f8f9fa;
        border: 2px dashed #AED6F1;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
        transition: border-color 0.3s, background 0.3s;
        animation: borderPulse 3s infinite;
    }
    .upload-section:hover {
        border-color: #3498db;
        background: #eaf4fb;
    }

    /* ================= WELCOME PAGE ================= */
    .welcome-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        animation: fadeInUp 0.7s ease both;
    }
    .welcome-title {
        font-family: 'Sora', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);   /* RESPONSIVE FONT */
        font-weight: 800;
        color: #0e1117;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }
    .welcome-brand {
        font-family: 'Sora', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);   /* RESPONSIVE FONT */
        font-weight: 800;
        background: linear-gradient(135deg, #3498db, #1abc9c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 1.2rem;
        animation: shimmer 4s infinite;
        background-size: 200% auto;
    }
    .welcome-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: clamp(1rem, 2.5vw, 1.4rem);  /* RESPONSIVE FONT */
        color: #555;
        font-weight: 400;
        margin-bottom: 2.2rem;
        line-height: 1.6;
    }

    /* ================= FEATURE CARDS ================= */
    .feature-card {
        background: white;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.07);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease both;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 14px 36px rgba(52,152,219,0.18);
        animation: float 2s ease-in-out infinite;
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 10px;
        display: block;
    }
    .feature-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 15px;
        color: #1a2535;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 13px;
        color: #7f8c8d;
        line-height: 1.5;
    }

    /* ================= ROBOT IMAGE: FLOATING ANIMATION ================= */
    .robot-container {
        animation: float 3s ease-in-out infinite;
    }

    /* ================= RESPONSIVE: MOBILE ================= */
    @media (max-width: 768px) {
        .main-header { font-size: 18px; padding: 16px; }
        .metric-card { padding: 14px 12px; }
        .content-frame { padding: 14px; }
        .welcome-title, .welcome-brand { font-size: 2rem !important; }
        .welcome-subtitle { font-size: 1rem !important; }
        .feature-card { margin-bottom: 12px; }
    }

    /* ================= GET STARTED BUTTON ================= */
    .stButton > button[kind="primary"] {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        border-radius: 50px;
        padding: 12px 32px;
        background: linear-gradient(135deg, #3498db, #1abc9c) !important;
        border: none !important;
        color: white !important;
        font-size: 16px;
        box-shadow: 0 6px 20px rgba(52,152,219,0.4);
        transition: all 0.3s ease;
        animation: pulse-glow 2.5s infinite;
    }
    .stButton > button[kind="primary"]:hover {
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 12px 32px rgba(52,152,219,0.5);
    }

    /* ================= ABOUT PAGE CARDS ================= */
    .founder-card {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-top: 3px solid #3498db;
        transition: transform 0.25s;
        animation: fadeInUp 0.6s ease both;
    }
    .founder-card:hover { transform: translateY(-4px); }

    /* ================= DOWNLOAD BUTTON ================= */
    .stDownloadButton > button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #27ae60, #2ecc71) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stDownloadButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(39,174,96,0.4) !important;
    }

    /* ================= STREAMLIT NATIVE METRIC ================= */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
        border-left: 3px solid #3498db;
        animation: fadeInUp 0.5s ease both;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Welcome'
if 'all_datasets' not in st.session_state:
    st.session_state.all_datasets = {}
if 'selected_file_name' not in st.session_state:
    st.session_state.selected_file_name = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'saved_charts' not in st.session_state:
    st.session_state.saved_charts = []


# =========================================================
# === 2. HELPER FUNCTIONS =================================
# =========================================================

def load_data_with_encoding(file, file_type, user_encoding=None):
    def try_encodings(read_function, file, encodings, **kwargs):
        for enc in encodings:
            try:
                file.seek(0)
                return read_function(file, encoding=enc, **kwargs)
            except (UnicodeDecodeError, TypeError, ValueError):
                continue
        raise ValueError("Unable to decode file.")

    try:
        file_type = file_type.lower()
        if file_type == "excel":
            file_type = file.name.split('.')[-1].lower()

        encodings_to_try = []
        if user_encoding:
            encodings_to_try.append(user_encoding)
        encodings_to_try += ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        if file_type == 'csv':
            return try_encodings(pd.read_csv, file, encodings_to_try, sep=None, engine='python', on_bad_lines='skip')
        elif file_type == 'txt':
            return try_encodings(pd.read_csv, file, encodings_to_try, sep=None, engine='python', on_bad_lines='skip')
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(file)
        elif file_type == 'json':
            try:
                file.seek(0)
                return pd.read_json(file)
            except ValueError:
                return try_encodings(pd.read_json, file, encodings_to_try, lines=True)
        elif file_type == 'xml':
            return try_encodings(pd.read_xml, file, encodings_to_try)
        elif file_type == 'parquet':
            return pd.read_parquet(file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


# =========================================================
# === HTML Report Generator ================================
# =========================================================

def generate_html_report(df, saved_charts):
    def compute_kpi_value(df, column, agg):
        if column == "__rows__":         return f"{len(df):,}"
        elif column == "__columns__":    return f"{len(df.columns):,}"
        elif column == "__numeric__":    return f"{len(df.select_dtypes(include=['int64', 'float64']).columns):,}"
        elif column == "__missing__":    return f"{int(df.isnull().sum().sum()):,}"
        elif column == "__duplicates__": return f"{int(df.duplicated().sum()):,}"
        if column not in df.columns: return "N/A"
        try:
            series = pd.to_numeric(df[column], errors='coerce')
            if agg == "Sum":      return f"{series.sum():,.2f}"
            elif agg == "Count":  return f"{series.count():,}"
            elif agg == "Mean":   return f"{series.mean():,.2f}"
            elif agg == "Median": return f"{series.median():,.2f}"
            elif agg == "Max":    return f"{series.max():,.2f}"
            elif agg == "Min":    return f"{series.min():,.2f}"
            elif agg == "Std":    return f"{series.std():,.2f}"
        except Exception:
            return str(df[column].count())

    kpi_configs = st.session_state.get("kpi_configs", [
        {"label": "Total Records",   "column": "__rows__",    "agg": "Count", "color": "#3498db"},
        {"label": "Total Columns",   "column": "__columns__", "agg": "Count", "color": "#3498db"},
        {"label": "Numeric Columns", "column": "__numeric__", "agg": "Count", "color": "#3498db"},
        {"label": "Missing Values",  "column": "__missing__", "agg": "Count", "color": "#3498db"},
    ])

    kpi_cards_html = ""
    for cfg in kpi_configs:
        label  = cfg.get("label",  "KPI")
        column = cfg.get("column", "__rows__")
        agg    = cfg.get("agg",    "Count")
        color  = cfg.get("color",  "#3498db")
        val    = compute_kpi_value(df, column, agg)
        val_color = "#e74c3c" if (column == "__missing__" and int(df.isnull().sum().sum()) > 0) else color

        kpi_cards_html += f"""
        <div class="metric-card" style="border-top: 4px solid {color};">
            <span class="metric-label">{label}</span>
            <p class="metric-value" style="color: {val_color};">{val}</p>
            <span class="metric-agg">{agg}</span>
        </div>"""

    charts_html = ""
    if saved_charts:
        for chart in saved_charts:
            chart_div = pio.to_html(chart['fig'], full_html=False, include_plotlyjs='cdn')
            user_desc = chart.get('description', '').strip()
            desc_html = f"""<div class="analysis-note"><span class="note-label">📝 Analysis Note:</span>{user_desc}</div>""" if user_desc else ""
            charts_html += f"""
            <div class="chart-container">
                <h3 class="chart-title">{chart['title']}</h3>
                {chart_div}
                {desc_html}
            </div>"""
    else:
        charts_html = "<p style='color:#7f8c8d; padding:20px;'>No visualizations were pinned to the dashboard.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #2c3e50; padding: 40px 30px; }}
        .report-header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px 40px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
        .report-header h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: 1px; }}
        .section-header {{ font-size: 1.25rem; font-weight: 700; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; margin: 35px 0 20px 0; }}
        .kpi-grid {{ display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 10px; }}
        .metric-card {{ background: white; padding: 22px 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); flex: 1 1 160px; min-width: 140px; text-align: center; }}
        .metric-value {{ font-size: 2rem; font-weight: 800; margin: 10px 0 4px 0; display: block; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
        .metric-agg {{ color: #bdc3c7; font-size: 0.72rem; text-transform: uppercase; }}
        .chart-container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); margin-bottom: 35px; }}
        .chart-title {{ font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin-bottom: 18px; padding-bottom: 10px; border-bottom: 1px solid #ecf0f1; }}
        .analysis-note {{ background-color: #eaf4fb; border-left: 5px solid #3498db; padding: 14px 18px; margin-top: 20px; border-radius: 6px; color: #2c3e50; font-size: 0.92rem; line-height: 1.6; }}
        .note-label {{ font-weight: 700; color: #2980b9; display: block; margin-bottom: 6px; font-size: 0.88rem; }}
        @media (max-width: 600px) {{ body {{ padding: 16px; }} .metric-card {{ min-width: 100%; }} }}
    </style>
</head>
<body>
    <div class="report-header"><h1>📊 Dashboard</h1></div>
    <div class="section-header">Key Performance Indicators</div>
    <div class="kpi-grid">{kpi_cards_html}</div>
    <div class="section-header">Visualizations &amp; Analysis</div>
    {charts_html}
</body>
</html>"""


# =========================================================
# === DASHBOARD PAGE ======================================
# =========================================================

def dashboard_page():
    st.markdown('<div class="main-header">📊 Dashboard</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload data in the 'Data Upload' section to view the dashboard.")
        return

    df = st.session_state.df

    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    all_columns = df.columns.tolist()

    if "kpi_configs" not in st.session_state:
        st.session_state.kpi_configs = [
            {"label": "Total Records",   "column": "__rows__",    "agg": "Count", "color": "#3498db"},
            {"label": "Total Columns",   "column": "__columns__", "agg": "Count", "color": "#3498db"},
            {"label": "Numeric Columns", "column": "__numeric__", "agg": "Count", "color": "#3498db"},
            {"label": "Missing Values",  "column": "__missing__", "agg": "Count", "color": "#3498db"},
        ]

    with st.expander("⚙️ Customize KPI Cards", expanded=False):
        st.caption("Set how many KPI cards to show and configure each one.")
        num_kpis = st.slider("How many KPI cards?", min_value=1, max_value=6,
                             value=len(st.session_state.kpi_configs), key="num_kpis_slider")

        while len(st.session_state.kpi_configs) < num_kpis:
            st.session_state.kpi_configs.append({
                "label": f"KPI {len(st.session_state.kpi_configs) + 1}",
                "column": all_columns[0] if all_columns else "__rows__",
                "agg": "Sum", "color": "#3498db"
            })
        st.session_state.kpi_configs = st.session_state.kpi_configs[:num_kpis]
        st.markdown("---")

        builtin_options = {
            "Total Rows": "__rows__", "Total Columns": "__columns__",
            "Numeric Columns": "__numeric__", "Missing Values": "__missing__",
            "Duplicate Rows": "__duplicates__",
        }
        agg_options = ["Sum", "Count", "Mean", "Median", "Max", "Min", "Std"]

        for idx in range(num_kpis):
            cfg = st.session_state.kpi_configs[idx]
            st.markdown(f"**Card {idx + 1}**")
            c1, c2, c3, c4 = st.columns([1.2, 1.8, 1.2, 0.8])

            with c1:
                new_label = st.text_input("Card Label", value=cfg.get("label", f"KPI {idx+1}"), key=f"kpi_label_{idx}")

            col_display_options = list(builtin_options.keys()) + all_columns
            col_value_options   = list(builtin_options.values()) + all_columns
            cur_col     = cfg.get("column", "__rows__")
            cur_col_idx = col_value_options.index(cur_col) if cur_col in col_value_options else 0

            with c2:
                selected_display = st.selectbox("Column / Metric", options=col_display_options,
                                                index=cur_col_idx, key=f"kpi_col_{idx}")
                selected_col = col_value_options[col_display_options.index(selected_display)]

            is_builtin = selected_col.startswith("__")
            with c3:
                if is_builtin:
                    st.selectbox("Aggregation", ["Count"], key=f"kpi_agg_{idx}", disabled=True)
                    selected_agg = "Count"
                else:
                    cur_agg = cfg.get("agg", "Sum")
                    agg_idx = agg_options.index(cur_agg) if cur_agg in agg_options else 0
                    selected_agg = st.selectbox("Aggregation", agg_options, index=agg_idx, key=f"kpi_agg_{idx}")

            with c4:
                selected_color = st.color_picker("Accent Color", value=cfg.get("color", "#3498db"), key=f"kpi_color_{idx}")

            st.session_state.kpi_configs[idx] = {
                "label": new_label, "column": selected_col, "agg": selected_agg, "color": selected_color
            }
            st.markdown("---")

    def compute_kpi_value(df, column, agg):
        if column == "__rows__":         return f"{len(df):,}"
        elif column == "__columns__":    return f"{len(df.columns):,}"
        elif column == "__numeric__":    return f"{len(df.select_dtypes(include=['int64', 'float64']).columns):,}"
        elif column == "__missing__":    return f"{int(df.isnull().sum().sum()):,}"
        elif column == "__duplicates__": return f"{int(df.duplicated().sum()):,}"
        if column not in df.columns: return "N/A"
        try:
            series = pd.to_numeric(df[column], errors='coerce')
            if agg == "Sum":      return f"{series.sum():,.2f}"
            elif agg == "Count":  return f"{series.count():,}"
            elif agg == "Mean":   return f"{series.mean():,.2f}"
            elif agg == "Median": return f"{series.median():,.2f}"
            elif agg == "Max":    return f"{series.max():,.2f}"
            elif agg == "Min":    return f"{series.min():,.2f}"
            elif agg == "Std":    return f"{series.std():,.2f}"
        except Exception:
            return str(df[column].count())

    num_cards = len(st.session_state.kpi_configs)
    kpi_cols  = st.columns(num_cards)

    for col_ui, cfg in zip(kpi_cols, st.session_state.kpi_configs):
        val   = compute_kpi_value(df, cfg["column"], cfg["agg"])
        label = cfg["label"]
        color = cfg.get("color", "#3498db")
        val_color = "#e74c3c" if (cfg["column"] == "__missing__" and int(df.isnull().sum().sum()) > 0) else color

        with col_ui:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3 style='margin: 0; color: #2c3e50; font-family: Sora, sans-serif;'>{label}</h3>
                <p style='font-size: 26px; font-weight: 800; color: {val_color}; margin: 10px 0; font-family: Sora, sans-serif;'>{val}</p>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.saved_charts:
        st.markdown('<div class="section-header">📌 Your Pinned Visualizations</div>', unsafe_allow_html=True)
        for i, chart_obj in enumerate(st.session_state.saved_charts):
            with st.container():
                st.subheader(f"{chart_obj['title']}")
                st.plotly_chart(chart_obj['fig'], use_container_width=True, key=f"saved_chart_{i}")
                current_desc = chart_obj.get('description', '')
                new_desc = st.text_area(
                    "📝 Add Analysis Note (Optional):",
                    value=current_desc,
                    placeholder="Type your insights here...",
                    key=f"desc_input_{i}", height=100
                )
                st.session_state.saved_charts[i]['description'] = new_desc
                if st.button(f"🗑️ Remove Chart {i+1}", key=f"del_{i}"):
                    st.session_state.saved_charts.pop(i)
                    st.rerun()
            st.markdown("---")

    st.markdown("### 📥 Export Dashboard")
    st.info("Download the entire dashboard (KPIs + Charts + Your Notes) as an offline HTML file.")
    html_report = generate_html_report(df, st.session_state.saved_charts)
    st.download_button(
        label="⬇️ Download Full HTML Report",
        data=html_report,
        file_name="data_analysis_report.html",
        mime="text/html"
    )


# =========================================================
# === DATA UPLOAD PAGE =====================================
# =========================================================

def data_upload_page():
    st.markdown('<div class="section-header">📁 Data Upload Management</div>', unsafe_allow_html=True)
    st.subheader("📥 Upload Datasets")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=['csv', 'xlsx', 'xls', 'json', 'txt', 'xml', 'parquet'],
        accept_multiple_files=True,
        help="Supported: CSV, Excel, JSON, TXT, XML, Parquet. Auto-encoding detection enabled.",
        label_visibility="collapsed"
    )

    if uploaded_files:
        for file in uploaded_files:
            file_key = file.name
            if file_key not in st.session_state.all_datasets:
                with st.spinner(f"Processing {file.name}..."):
                    ext = file.name.split('.')[-1].lower()
                    df = load_data_with_encoding(file, file_type=ext)
                    if df is not None:
                        st.session_state.all_datasets[file_key] = df
                        st.success(f"✅ Successfully loaded: {file.name}")
                    else:
                        st.error(f"❌ Failed to load {file.name}. Check format/encoding.")
        st.markdown("---")

    if st.session_state.all_datasets:
        st.markdown('<div class="content-frame">', unsafe_allow_html=True)
        st.subheader("📂 Select Active Dataset")

        file_options = list(st.session_state.all_datasets.keys())
        selected_file = st.selectbox(
            "Choose a dataset to analyze:", file_options,
            index=file_options.index(st.session_state.selected_file_name)
                  if st.session_state.selected_file_name in file_options else 0
        )

        if selected_file:
            st.session_state.selected_file_name = selected_file
            st.session_state.df = st.session_state.all_datasets[selected_file]
            current_df = st.session_state.df
            st.markdown(f"---\n**Previewing: {selected_file}**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(current_df):,}")
            with col2:
                st.metric("Columns", f"{len(current_df.columns):,}")
            with col3:
                mem_usage = current_df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory Usage", f"{mem_usage:.2f} MB")

            st.dataframe(current_df.head(10), use_container_width=True)

            with st.expander("View Data Types Info"):
                dtype_info = pd.DataFrame({
                    'Column': current_df.columns,
                    'Data Type': current_df.dtypes.astype(str),
                    'Non-Null Count': current_df.count(),
                    'Null Count': current_df.isnull().sum()
                })
                st.dataframe(dtype_info, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Generate Dataset Profiling Report")

            if st.button("🔍 Generate Profiling Report for This Dataset"):
                with st.spinner("Generating profiling report..."):
                    profile = ProfileReport(current_df, title=f"Profiling Report - {selected_file}", explorative=True)
                    profile.to_file("dataset_report.html")
                    with open("dataset_report.html", "rb") as f:
                        html_bytes = f.read()
                    st.success("✅ Profiling report generated successfully!")
                    st.download_button(
                        label="⬇️ Download Profiling Report",
                        data=html_bytes,
                        file_name=f"{selected_file}_report.html",
                        mime="text/html"
                    )

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No datasets uploaded yet. Use the section above to add data.")


# =========================================================
# === WELCOME PAGE =========================================
# =========================================================

def welcome_page():
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
        st.markdown("""
        <div class="welcome-title">Welcome to</div>
        <div class="welcome-brand">DataMate AI</div>
        <div class="welcome-subtitle">Your smart partner for<br>data insights & intelligence</div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Get Started", type="primary", use_container_width=False):
            st.session_state.current_page = "Data Upload"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        try:
            st.markdown('<div class="robot-container">', unsafe_allow_html=True)
            st.image("datamate_logo_50.png", width=350)
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="robot-container" style="
                display: flex; justify-content: center; align-items: center;
                height: 380px;
                background: radial-gradient(circle at 50% 40%, #dbeeff 10%, #f0f8ff 70%);
                border-radius: 20px; border: 1px solid #cde;">
                <div style="font-size: 130px; filter: drop-shadow(0 8px 24px rgba(52,152,219,0.3));">🤖</div>
            </div>
            """, unsafe_allow_html=True)

    # st.markdown("---")

    # ── Animated Feature Cards ────────────────────────────────────────────────
    features = [
        ("📊", "Smart Analytics", "Auto-generated KPIs & real-time insights from your data"),
        ("🧹", "Auto Cleaning",   "Fix missing values, duplicates & outliers instantly"),
        ("📈", "Visualization",   "Create beautiful interactive charts in seconds"),
        ("🤖", "ML Models",       "Train AI models with zero code, zero effort"),
    ]
    f_cols = st.columns(4)
    for col, (icon, title, desc) in zip(f_cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)



def data_preprocessing_page():
    """Comprehensive Data Preprocessing & ML Preprocessing - DataMate AI v2"""
    import pandas as pd
    import numpy as np
    import io
    from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                        MinMaxScaler, StandardScaler,
                                        RobustScaler, MaxAbsScaler,
                                        PowerTransformer, QuantileTransformer)
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                            f_classif, f_regression,
                                            mutual_info_classif, mutual_info_regression)
    from sklearn.decomposition import PCA
    from scipy import stats
    from scipy.stats import shapiro, skew, kurtosis
    import plotly.express as px
    import plotly.graph_objects as go
    import warnings
    warnings.filterwarnings("ignore")

    st.markdown('<div class="main-header">🔄 Data Preprocessing Center</div>',
                unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the 'Data Upload' section first.")
        return

    # ── Top Controls Bar ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="content-frame" style="background:linear-gradient(135deg,#f0f4f8,#ffffff);
         border-left:5px solid #3498db; padding:14px 20px; margin-bottom:16px;">
        <h5 style="margin:0; font-family:Sora,sans-serif; color:#1a2535;">
            ⚙️ Preprocessing Controls
        </h5>
    </div>
    """, unsafe_allow_html=True)

    ctl1, ctl2, ctl3, ctl4 = st.columns([1, 1, 1, 3])
    with ctl1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with ctl2:
        if st.button("⏮️ Reset to Original", use_container_width=True):
            if st.session_state.selected_file_name in st.session_state.all_datasets:
                st.session_state.df = st.session_state.all_datasets[
                    st.session_state.selected_file_name].copy()
                st.success("✅ Reverted to original dataset.")
                st.rerun()
    with ctl3:
        if st.button("📸 Save Snapshot", use_container_width=True,
                     help="Save current state as a named version"):
            if "snapshots" not in st.session_state:
                st.session_state.snapshots = {}
            snap_name = f"Snapshot_{len(st.session_state.snapshots)+1}"
            st.session_state.snapshots[snap_name] = st.session_state.df.copy()
            st.success(f"✅ Saved: {snap_name}")

    df = st.session_state.df.copy()
    st.info(
        f"📊 **{st.session_state.selected_file_name}** | "
        f"Rows: {len(df):,} | Columns: {len(df.columns)} | "
        f"Missing: {df.isnull().sum().sum():,} | "
        f"Duplicates: {df.duplicated().sum():,}"
    )

    # Snapshot restore
    if "snapshots" in st.session_state and st.session_state.snapshots:
        with st.expander("🔁 Restore from Snapshot"):
            snap_choice = st.selectbox("Choose snapshot",
                                       list(st.session_state.snapshots.keys()))
            if st.button("Restore This Snapshot"):
                st.session_state.df = st.session_state.snapshots[snap_choice].copy()
                st.success(f"✅ Restored: {snap_choice}")
                st.rerun()

    # ── SECTION TABS ─────────────────────────────────────────────────────────
    (
        tab_overview, tab_missing, tab_dupes, tab_cols,
        tab_types, tab_outliers, tab_incon,
        tab_encode, tab_scale, tab_feature,
        tab_ml, tab_export
    ) = st.tabs([
        "📋 Overview",
        "🔍 Missing Values",
        "♻️ Duplicates",
        "🗂️ Column Mgmt",
        "🔁 Data Types",
        "📉 Outliers",
        "✏️ Inconsistent Data",
        "🔢 Encoding",
        "📐 Scaling",
        "🎯 Feature Engineering",
        "🤖 ML Preprocessing",
        "📤 Export"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0 – OVERVIEW / DATA HEALTH REPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown('<div class="section-header">📋 Data Health Overview</div>',
                    unsafe_allow_html=True)

        numeric_cols  = df.select_dtypes(include=['int64','float64']).columns.tolist()
        cat_cols      = df.select_dtypes(include=['object','category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # KPI strip
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpis = [
            ("Rows",            f"{len(df):,}",                      "#3498db"),
            ("Columns",         f"{len(df.columns)}",                "#2ecc71"),
            ("Numeric",         f"{len(numeric_cols)}",              "#9b59b6"),
            ("Categorical",     f"{len(cat_cols)}",                  "#e67e22"),
            ("Missing Cells",   f"{df.isnull().sum().sum():,}",      "#e74c3c" if df.isnull().sum().sum() > 0 else "#2ecc71"),
            ("Duplicates",      f"{df.duplicated().sum():,}",        "#e74c3c" if df.duplicated().sum() > 0 else "#2ecc71"),
        ]
        for col_ui, (label, val, color) in zip([k1,k2,k3,k4,k5,k6], kpis):
            with col_ui:
                st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid {color}; text-align:center;">
                    <p style="font-size:11px;color:#7f8c8d;text-transform:uppercase;
                              letter-spacing:1px;margin:0;">{label}</p>
                    <p style="font-size:22px;font-weight:800;color:{color};
                              margin:6px 0 0 0;font-family:Sora,sans-serif;">{val}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Missing value heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Values per Column**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            }).sort_values('Missing', ascending=False)
            missing_df_show = missing_df[missing_df['Missing'] > 0]
            if len(missing_df_show) > 0:
                fig_miss = px.bar(missing_df_show, x='Column', y='Missing %',
                                  color='Missing %',
                                  color_continuous_scale='Reds',
                                  title="Missing % by Column",
                                  template="plotly_white")
                fig_miss.update_layout(height=300, margin=dict(t=40,b=40))
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ No missing values!")

        with col2:
            st.markdown("**Data Type Distribution**")
            dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
            dtype_counts.columns = ['Type', 'Count']
            fig_dtype = px.pie(dtype_counts, names='Type', values='Count',
                               hole=0.45, title="Column Types",
                               template="plotly_white",
                               color_discrete_sequence=px.colors.qualitative.Set2)
            fig_dtype.update_layout(height=300, margin=dict(t=40,b=40))
            st.plotly_chart(fig_dtype, use_container_width=True)

        st.markdown("---")

        # Full column profile
        st.markdown("**Full Column Profile**")
        profile_rows = []
        for c in df.columns:
            row = {
                "Column": c,
                "Type": str(df[c].dtype),
                "Non-Null": df[c].count(),
                "Null": df[c].isnull().sum(),
                "Null %": f"{df[c].isnull().mean()*100:.1f}%",
                "Unique": df[c].nunique(),
            }
            if df[c].dtype in ['int64','float64']:
                row["Min"]  = f"{df[c].min():.2f}"
                row["Max"]  = f"{df[c].max():.2f}"
                row["Mean"] = f"{df[c].mean():.2f}"
                row["Std"]  = f"{df[c].std():.2f}"
                sk = skew(df[c].dropna())
                row["Skew"] = f"{sk:.2f}"
            else:
                row["Min"] = row["Max"] = row["Mean"] = row["Std"] = row["Skew"] = "—"
            profile_rows.append(row)
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – MISSING VALUES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_missing:
        st.markdown('<div class="section-header">🔍 Handle Missing Values</div>',
                    unsafe_allow_html=True)

        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
            'Dtype': df.dtypes.astype(str).values
        }).sort_values('Missing', ascending=False)
        missing_summary = missing_summary[missing_summary['Missing'] > 0]

        if len(missing_summary) > 0:
            st.dataframe(missing_summary, use_container_width=True)
            st.markdown("---")

            # Strategy selector
            strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
                "🔧 Single Column", "🔧 Bulk Fill", "🧠 Advanced (KNN / Iterative)"
            ])

            with strategy_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fill_column = st.selectbox("Select Column",
                                               missing_summary['Column'].tolist(),
                                               key="fill_col")
                    fill_method = st.radio("Fill Method", [
                        "Drop Rows", "Mean", "Median", "Mode",
                        "Forward Fill", "Backward Fill",
                        "Interpolate (Linear)", "Custom Value"
                    ], key="fill_method")
                    fill_value = st.text_input("Custom value",
                                               key="fill_value") if fill_method == "Custom Value" else None

                    if st.button("✅ Apply Fill", key="apply_fill"):
                        try:
                            if fill_method == "Drop Rows":
                                df = df.dropna(subset=[fill_column])
                            elif fill_method == "Mean":
                                df[fill_column].fillna(df[fill_column].mean(), inplace=True)
                            elif fill_method == "Median":
                                df[fill_column].fillna(df[fill_column].median(), inplace=True)
                            elif fill_method == "Mode":
                                df[fill_column].fillna(df[fill_column].mode()[0], inplace=True)
                            elif fill_method == "Forward Fill":
                                df[fill_column].fillna(method='ffill', inplace=True)
                            elif fill_method == "Backward Fill":
                                df[fill_column].fillna(method='bfill', inplace=True)
                            elif fill_method == "Interpolate (Linear)":
                                df[fill_column] = df[fill_column].interpolate(method='linear')
                            elif fill_method == "Custom Value" and fill_value:
                                val = float(fill_value) if df[fill_column].dtype in ['int64','float64'] else fill_value
                                df[fill_column].fillna(val, inplace=True)
                            st.session_state.df = df
                            st.success("✅ Applied!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

                with col2:
                    st.markdown("**Drop Columns by Threshold**")
                    threshold = st.slider("Drop if missing % >", 0, 100, 50,
                                          key="missing_threshold")
                    cols_to_drop = missing_summary[
                        missing_summary['Missing %'] > threshold]['Column'].tolist()
                    if cols_to_drop:
                        for c in cols_to_drop:
                            st.write(f"  • {c}")
                        if st.button("Drop These Columns", key="drop_missing_cols"):
                            df = df.drop(columns=cols_to_drop)
                            st.session_state.df = df
                            st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                            st.rerun()
                    else:
                        st.info("No columns exceed the threshold")

            with strategy_tab2:
                st.markdown("**Bulk Fill — Apply same strategy to multiple columns**")
                bulk_cols = st.multiselect("Select columns to fill",
                    missing_summary['Column'].tolist(), key="bulk_cols")
                bulk_method = st.selectbox("Bulk fill method",
                    ["Mean", "Median", "Mode", "Forward Fill",
                     "Backward Fill", "Drop Rows"], key="bulk_method")

                if st.button("✅ Apply Bulk Fill", key="bulk_fill_btn") and bulk_cols:
                    for c in bulk_cols:
                        try:
                            if bulk_method == "Mean" and df[c].dtype in ['int64','float64']:
                                df[c].fillna(df[c].mean(), inplace=True)
                            elif bulk_method == "Median" and df[c].dtype in ['int64','float64']:
                                df[c].fillna(df[c].median(), inplace=True)
                            elif bulk_method == "Mode":
                                df[c].fillna(df[c].mode()[0], inplace=True)
                            elif bulk_method == "Forward Fill":
                                df[c].fillna(method='ffill', inplace=True)
                            elif bulk_method == "Backward Fill":
                                df[c].fillna(method='bfill', inplace=True)
                            elif bulk_method == "Drop Rows":
                                df = df.dropna(subset=[c])
                        except Exception:
                            pass
                    st.session_state.df = df
                    st.success(f"✅ Bulk filled {len(bulk_cols)} columns")
                    st.rerun()

            with strategy_tab3:
                st.markdown("**KNN Imputer** — fills numeric missing values using K nearest neighbours")
                num_missing_cols = [
                    c for c in missing_summary['Column']
                    if df[c].dtype in ['int64','float64']
                ]
                if num_missing_cols:
                    knn_cols = st.multiselect("Select numeric columns for KNN Imputation",
                                              num_missing_cols, key="knn_cols",
                                              default=num_missing_cols)
                    k_neighbors = st.slider("K Neighbors", 2, 20, 5, key="knn_k")
                    if st.button("✅ Apply KNN Imputer", key="knn_btn") and knn_cols:
                        imputer = KNNImputer(n_neighbors=k_neighbors)
                        df[knn_cols] = imputer.fit_transform(df[knn_cols])
                        st.session_state.df = df
                        st.success(f"✅ KNN imputed {len(knn_cols)} columns")
                        st.rerun()
                else:
                    st.info("No numeric columns with missing values for KNN.")
        else:
            st.success("✅ No missing values found! Your data is clean.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – DUPLICATES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_dupes:
        st.markdown('<div class="section-header">♻️ Handle Duplicates</div>',
                    unsafe_allow_html=True)

        dup_count = df.duplicated().sum()
        st.metric("Total Duplicate Rows", f"{dup_count:,}",
                  delta="Clean" if dup_count == 0 else "Found",
                  delta_color="normal" if dup_count == 0 else "inverse")

        col1, col2 = st.columns(2)
        with col1:
            if dup_count > 0:
                st.markdown("**Preview Duplicates**")
                st.dataframe(
                    df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(30),
                    use_container_width=True)
            else:
                st.success("✅ No duplicate rows found.")

        with col2:
            st.markdown("**Remove Options**")
            keep_option = st.radio("Keep",
                ["First occurrence", "Last occurrence", "Remove all"],
                key="keep_option")
            subset_cols = st.multiselect(
                "Subset columns (empty = all columns)",
                df.columns.tolist(), key="subset_cols")
            keep_map = {
                "First occurrence": "first",
                "Last occurrence":  "last",
                "Remove all":       False
            }
            if st.button("🗑️ Remove Duplicates", key="remove_duplicates"):
                orig = len(df)
                df = df.drop_duplicates(
                    subset=subset_cols if subset_cols else None,
                    keep=keep_map[keep_option])
                st.session_state.df = df
                st.success(f"✅ Removed {orig - len(df):,} duplicate rows")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – COLUMN MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cols:
        st.markdown('<div class="section-header">🗂️ Column Management</div>',
                    unsafe_allow_html=True)

        sub1, sub2, sub3 = st.tabs(["Rename / Drop", "Reorder", "Create New Column"])

        with sub1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Rename Column**")
                col_to_rename = st.selectbox("Column", df.columns.tolist(), key="rename_col")
                new_name = st.text_input("New name", value=col_to_rename, key="new_col_name")
                if st.button("Rename", key="rename_btn"):
                    df = df.rename(columns={col_to_rename: new_name})
                    st.session_state.df = df
                    st.success(f"✅ '{col_to_rename}' → '{new_name}'")
                    st.rerun()
            with col2:
                st.markdown("**Drop Columns**")
                cols_to_drop = st.multiselect("Select columns to drop",
                                              df.columns.tolist(), key="cols_to_drop")
                if st.button("🗑️ Drop Selected", key="drop_cols_btn") and cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    st.session_state.df = df
                    st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                    st.rerun()

            st.markdown("---")
            st.dataframe(pd.DataFrame({
                'Column': df.columns, 'Type': df.dtypes.astype(str),
                'Non-Null': df.count(), 'Null': df.isnull().sum(),
                'Unique': df.nunique()
            }), use_container_width=True)

        with sub2:
            st.markdown("**Reorder Columns** — drag and rearrange")
            ordered = st.multiselect(
                "Arrange columns in desired order:",
                options=df.columns.tolist(),
                default=df.columns.tolist(),
                key="col_reorder"
            )
            if st.button("✅ Apply Order", key="reorder_btn") and ordered:
                remaining = [c for c in df.columns if c not in ordered]
                df = df[ordered + remaining]
                st.session_state.df = df
                st.success("✅ Columns reordered")
                st.rerun()

        with sub3:
            st.markdown("**Create Derived Column**")
            new_col_name = st.text_input("New column name", key="new_derived_col")
            numeric_cols_d = df.select_dtypes(include=['int64','float64']).columns.tolist()
            col_a = st.selectbox("Column A", numeric_cols_d, key="derived_a")
            operation = st.selectbox("Operation",
                ["+", "−", "×", "÷", "log(A)", "sqrt(A)",
                 "A²", "A % B", "abs(A)"], key="derived_op")
            col_b = st.selectbox("Column B (if needed)", numeric_cols_d, key="derived_b")

            if st.button("✅ Create Column", key="create_derived_btn") and new_col_name:
                try:
                    if operation == "+":       df[new_col_name] = df[col_a] + df[col_b]
                    elif operation == "−":     df[new_col_name] = df[col_a] - df[col_b]
                    elif operation == "×":     df[new_col_name] = df[col_a] * df[col_b]
                    elif operation == "÷":     df[new_col_name] = df[col_a] / df[col_b].replace(0, np.nan)
                    elif operation == "log(A)":  df[new_col_name] = np.log1p(df[col_a].clip(lower=0))
                    elif operation == "sqrt(A)": df[new_col_name] = np.sqrt(df[col_a].clip(lower=0))
                    elif operation == "A²":      df[new_col_name] = df[col_a] ** 2
                    elif operation == "A % B":   df[new_col_name] = df[col_a] % df[col_b].replace(0, np.nan)
                    elif operation == "abs(A)":  df[new_col_name] = df[col_a].abs()
                    st.session_state.df = df
                    st.success(f"✅ Created column '{new_col_name}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – DATA TYPES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_types:
        st.markdown('<div class="section-header">🔁 Data Type Conversion</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Single Column Conversion**")
            col_to_convert = st.selectbox("Select Column",
                                          df.columns.tolist(), key="convert_col")
            st.caption(f"Current type: **{df[col_to_convert].dtype}** | "
                       f"Sample: `{df[col_to_convert].dropna().iloc[0] if len(df[col_to_convert].dropna()) > 0 else 'N/A'}`")
            new_type = st.selectbox("Convert To",
                ["int64","float64","string","category",
                 "datetime64[ns]","bool"], key="new_type")
            fmt_str = st.text_input("Datetime format (optional, e.g. %Y-%m-%d)",
                                    key="dt_fmt") if new_type == "datetime64[ns]" else None

            if st.button("✅ Convert", key="convert_btn"):
                try:
                    if new_type == "datetime64[ns]":
                        df[col_to_convert] = pd.to_datetime(
                            df[col_to_convert], format=fmt_str if fmt_str else None,
                            errors='coerce')
                    elif new_type == "bool":
                        df[col_to_convert] = df[col_to_convert].astype(bool)
                    else:
                        df[col_to_convert] = df[col_to_convert].astype(new_type)
                    st.session_state.df = df
                    st.success(f"✅ Converted to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

        with col2:
            st.markdown("**Bulk Type Conversion**")
            st.caption("Convert all object columns that look numeric/datetime automatically.")
            if st.button("🔍 Auto-Detect & Convert Numerics", key="auto_num"):
                converted = []
                for c in df.select_dtypes(include='object').columns:
                    converted_series = pd.to_numeric(df[c], errors='coerce')
                    if converted_series.notna().sum() / len(df) > 0.8:
                        df[c] = converted_series
                        converted.append(c)
                st.session_state.df = df
                st.success(f"✅ Converted: {converted if converted else 'None found'}")
                st.rerun()

            if st.button("🔍 Auto-Detect & Convert Datetimes", key="auto_dt"):
                converted = []
                for c in df.select_dtypes(include='object').columns:
                    try:
                        dt_series = pd.to_datetime(df[c], infer_datetime_format=True,
                                                   errors='coerce')
                        if dt_series.notna().sum() / len(df) > 0.7:
                            df[c] = dt_series
                            converted.append(c)
                    except Exception:
                        pass
                st.session_state.df = df
                st.success(f"✅ Converted: {converted if converted else 'None found'}")
                st.rerun()

            st.markdown("**Type Summary**")
            for dtype, count in df.dtypes.value_counts().items():
                st.markdown(f"""
                <div class="metric-card" style="padding:10px 14px;">
                    <span style="font-weight:700;color:#1a2535;">{dtype}</span>
                    <span style="color:#3498db;font-weight:600;float:right;">
                        {count} col(s)
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 – OUTLIERS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_outliers:
        st.markdown('<div class="section-header">📉 Outlier Detection & Treatment</div>',
                    unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns for outlier detection.")
        else:
            outlier_col = st.selectbox("Select Numeric Column", numeric_cols, key="outlier_col")

            # Distribution preview
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df[outlier_col].dropna(),
                                     name=outlier_col,
                                     marker_color='#3498db',
                                     boxmean='sd'))
            fig_box.update_layout(title=f"Distribution: {outlier_col}",
                                  template="plotly_white", height=280,
                                  margin=dict(t=40,b=20))
            st.plotly_chart(fig_box, use_container_width=True)

            # Stats
            s = df[outlier_col].dropna()
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            for col_ui, (lbl, val) in zip(
                [sc1,sc2,sc3,sc4,sc5],
                [("Mean", f"{s.mean():.2f}"),
                 ("Std",  f"{s.std():.2f}"),
                 ("Skew", f"{skew(s):.2f}"),
                 ("Kurt", f"{kurtosis(s):.2f}"),
                 ("Range",f"{s.max()-s.min():.2f}")]
            ):
                with col_ui:
                    st.metric(lbl, val)

            st.markdown("---")
            method_tab1, method_tab2, method_tab3, method_tab4 = st.tabs([
                "IQR Method", "Z-Score", "Percentile Clip", "Winsorization"
            ])

            with method_tab1:
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_mult")
                lb, ub = Q1 - multiplier * IQR, Q3 + multiplier * IQR
                mask = (df[outlier_col] < lb) | (df[outlier_col] > ub)
                outliers_iqr = df[mask]
                st.write(f"Bounds: [{lb:.2f}, {ub:.2f}] | Outliers: **{len(outliers_iqr)}**")
                if len(outliers_iqr) > 0:
                    with st.expander("Preview Outlier Rows"):
                        st.dataframe(outliers_iqr.head(20), use_container_width=True)
                action_iqr = st.radio("Treatment",
                    ["Remove Rows", "Cap to Bounds"], key="iqr_action")
                if st.button("✅ Apply IQR Treatment", key="iqr_btn"):
                    if action_iqr == "Remove Rows":
                        df = df[(df[outlier_col] >= lb) & (df[outlier_col] <= ub)]
                    else:
                        df[outlier_col] = df[outlier_col].clip(lower=lb, upper=ub)
                    st.session_state.df = df
                    st.success("✅ Applied IQR treatment")
                    st.rerun()

            with method_tab2:
                z_thresh = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z_thresh")
                z_scores = np.abs(stats.zscore(s))
                outlier_idx = s[z_scores > z_thresh].index
                outliers_z = df.loc[outlier_idx]
                st.write(f"Threshold: {z_thresh} | Outliers: **{len(outliers_z)}**")
                if len(outliers_z) > 0:
                    with st.expander("Preview Outlier Rows"):
                        st.dataframe(outliers_z.head(20), use_container_width=True)
                action_z = st.radio("Treatment",
                    ["Remove Rows", "Replace with Mean", "Replace with Median"],
                    key="z_action")
                if st.button("✅ Apply Z-Score Treatment", key="z_btn"):
                    if action_z == "Remove Rows":
                        df = df.drop(index=outlier_idx)
                    elif action_z == "Replace with Mean":
                        df.loc[outlier_idx, outlier_col] = s.mean()
                    else:
                        df.loc[outlier_idx, outlier_col] = s.median()
                    st.session_state.df = df
                    st.success("✅ Applied Z-Score treatment")
                    st.rerun()

            with method_tab3:
                p_low  = st.slider("Lower Percentile", 0, 10, 1, key="p_low")
                p_high = st.slider("Upper Percentile", 90, 100, 99, key="p_high")
                low_val  = s.quantile(p_low  / 100)
                high_val = s.quantile(p_high / 100)
                outside  = ((df[outlier_col] < low_val) | (df[outlier_col] > high_val)).sum()
                st.write(f"Clip range: [{low_val:.2f}, {high_val:.2f}] | "
                         f"Values outside: **{outside}**")
                if st.button("✅ Apply Percentile Clip", key="pct_btn"):
                    df[outlier_col] = df[outlier_col].clip(lower=low_val, upper=high_val)
                    st.session_state.df = df
                    st.success("✅ Clipped!")
                    st.rerun()

            with method_tab4:
                st.info("Winsorization replaces extreme values with the nearest non-extreme value.")
                w_low  = st.slider("Lower % to winsorize", 0.0, 10.0, 1.0, 0.5, key="w_low")
                w_high = st.slider("Upper % to winsorize", 0.0, 10.0, 1.0, 0.5, key="w_high")
                if st.button("✅ Apply Winsorization", key="wins_btn"):
                    from scipy.stats.mstats import winsorize
                    df[outlier_col] = winsorize(df[outlier_col],
                                                limits=[w_low/100, w_high/100])
                    st.session_state.df = df
                    st.success("✅ Winsorized!")
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 – INCONSISTENT DATA
    # ══════════════════════════════════════════════════════════════════════════
    with tab_incon:
        st.markdown('<div class="section-header">✏️ Fix Inconsistent Data</div>',
                    unsafe_allow_html=True)

        text_cols = df.select_dtypes(include='object').columns.tolist()

        sub_a, sub_b, sub_c, sub_d = st.tabs([
            "Whitespace & Case", "Value Replace / Regex",
            "Category Standardize", "Date Parsing"
        ])

        with sub_a:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Trim Whitespace**")
                trim_cols = st.multiselect("Select columns", text_cols, key="trim_cols")
                if st.button("✅ Trim", key="trim_btn") and trim_cols:
                    for c in trim_cols:
                        df[c] = df[c].astype(str).str.strip()
                    st.session_state.df = df
                    st.success("✅ Trimmed")
                    st.rerun()

            with col2:
                st.markdown("**Case Conversion**")
                if text_cols:
                    case_col = st.selectbox("Column", text_cols, key="case_col")
                    case_opt = st.radio("Convert to",
                        ["lowercase","UPPERCASE","Title Case","Sentence case"],
                        key="case_opt")
                    if st.button("✅ Convert Case", key="case_btn"):
                        if case_opt == "lowercase":
                            df[case_col] = df[case_col].str.lower()
                        elif case_opt == "UPPERCASE":
                            df[case_col] = df[case_col].str.upper()
                        elif case_opt == "Title Case":
                            df[case_col] = df[case_col].str.title()
                        else:
                            df[case_col] = df[case_col].str.capitalize()
                        st.session_state.df = df
                        st.success("✅ Applied")
                        st.rerun()

        with sub_b:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Find & Replace Value**")
                rep_col  = st.selectbox("Column", df.columns.tolist(), key="rep_col")
                old_val  = st.text_input("Find value", key="rep_old")
                new_val  = st.text_input("Replace with", key="rep_new")
                if st.button("✅ Replace", key="rep_btn") and old_val:
                    df[rep_col] = df[rep_col].astype(str).str.replace(
                        old_val, new_val, regex=False)
                    st.session_state.df = df
                    st.success(f"✅ Replaced '{old_val}' → '{new_val}'")
                    st.rerun()

            with col2:
                st.markdown("**Regex Replace**")
                regex_col     = st.selectbox("Column", text_cols, key="regex_col") if text_cols else None
                regex_pattern = st.text_input("Regex pattern (e.g. [^a-zA-Z0-9])", key="regex_pat")
                regex_replace = st.text_input("Replacement string", key="regex_rep")
                if st.button("✅ Apply Regex", key="regex_btn") and regex_col and regex_pattern:
                    try:
                        df[regex_col] = df[regex_col].astype(str).str.replace(
                            regex_pattern, regex_replace, regex=True)
                        st.session_state.df = df
                        st.success("✅ Regex applied")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

        with sub_c:
            st.markdown("**Category Value Mapping**")
            if text_cols:
                map_col = st.selectbox("Column", text_cols, key="map_col")
                unique_vals = df[map_col].dropna().unique().tolist()
                st.caption(f"Unique values ({len(unique_vals)}): {unique_vals[:15]}")

                st.markdown("Enter new label for each value (leave blank to keep):")
                mapping = {}
                chunk_cols = st.columns(3)
                for i, v in enumerate(unique_vals[:15]):
                    with chunk_cols[i % 3]:
                        mapped = st.text_input(f"`{v}`", key=f"map_{map_col}_{i}",
                                               placeholder=str(v))
                        if mapped.strip():
                            mapping[v] = mapped.strip()

                if st.button("✅ Apply Mapping", key="apply_map_btn") and mapping:
                    df[map_col] = df[map_col].replace(mapping)
                    st.session_state.df = df
                    st.success(f"✅ Mapped {len(mapping)} values in '{map_col}'")
                    st.rerun()

        with sub_d:
            st.markdown("**Parse & Extract Date Parts**")
            dt_cols = df.select_dtypes(include='datetime64').columns.tolist()
            obj_cols_d = df.select_dtypes(include='object').columns.tolist()
            all_date_candidates = dt_cols + obj_cols_d

            if all_date_candidates:
                date_src = st.selectbox("Date column", all_date_candidates, key="date_src")
                parts = st.multiselect("Extract parts",
                    ["Year","Month","Day","Hour","Minute",
                     "DayOfWeek","Quarter","WeekOfYear"],
                    default=["Year","Month","Day"], key="date_parts")

                if st.button("✅ Extract Date Parts", key="date_ext_btn") and parts:
                    try:
                        series = pd.to_datetime(df[date_src], errors='coerce')
                        if "Year"       in parts: df[f"{date_src}_year"]    = series.dt.year
                        if "Month"      in parts: df[f"{date_src}_month"]   = series.dt.month
                        if "Day"        in parts: df[f"{date_src}_day"]     = series.dt.day
                        if "Hour"       in parts: df[f"{date_src}_hour"]    = series.dt.hour
                        if "Minute"     in parts: df[f"{date_src}_minute"]  = series.dt.minute
                        if "DayOfWeek"  in parts: df[f"{date_src}_dow"]     = series.dt.dayofweek
                        if "Quarter"    in parts: df[f"{date_src}_quarter"] = series.dt.quarter
                        if "WeekOfYear" in parts: df[f"{date_src}_week"]    = series.dt.isocalendar().week.astype(int)
                        st.session_state.df = df
                        st.success(f"✅ Extracted {len(parts)} date features")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 – ENCODING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_encode:
        st.markdown('<div class="section-header">🔢 Categorical Encoding</div>',
                    unsafe_allow_html=True)

        cat_cols_enc = df.select_dtypes(include=['object','category']).columns.tolist()

        if not cat_cols_enc:
            st.info("No categorical columns found.")
        else:
            enc_tab1, enc_tab2, enc_tab3, enc_tab4, enc_tab5 = st.tabs([
                "Label Encoding", "One-Hot Encoding",
                "Ordinal Encoding", "Binary Encoding", "Target Encoding"
            ])

            with enc_tab1:
                le_col = st.selectbox("Column", cat_cols_enc, key="le_col")
                if st.button("✅ Apply Label Encoding", key="le_btn"):
                    le = LabelEncoder()
                    df[f"{le_col}_label_enc"] = le.fit_transform(df[le_col].astype(str))
                    st.session_state.df = df
                    st.success(f"✅ Created '{le_col}_label_enc'")
                    st.dataframe(pd.DataFrame({
                        'Original': le.classes_,
                        'Encoded': range(len(le.classes_))
                    }), use_container_width=True)
                    st.rerun()

            with enc_tab2:
                ohe_col  = st.selectbox("Column", cat_cols_enc, key="ohe_col")
                drop_first = st.checkbox("Drop first (avoid dummy trap)", value=True, key="drop_first")
                unique_count = df[ohe_col].nunique()
                st.write(f"Unique values: **{unique_count}**")
                if unique_count > 15:
                    st.warning("⚠️ High cardinality. Consider Label or Binary encoding instead.")
                if st.button("✅ Apply One-Hot Encoding", key="ohe_btn"):
                    ohe_df = pd.get_dummies(df[[ohe_col]], prefix=ohe_col,
                                            drop_first=drop_first)
                    df = pd.concat([df, ohe_df], axis=1)
                    st.session_state.df = df
                    st.success(f"✅ Created {len(ohe_df.columns)} OHE columns")
                    st.rerun()

            with enc_tab3:
                ord_col = st.selectbox("Column", cat_cols_enc, key="ord_col")
                unique_vals_ord = df[ord_col].dropna().unique().tolist()
                st.caption("Drag to set order (low → high):")
                ordered_cats = st.multiselect(
                    "Ordered categories (first = lowest rank):",
                    options=unique_vals_ord,
                    default=unique_vals_ord, key="ord_cats")
                if st.button("✅ Apply Ordinal Encoding", key="ord_btn") and ordered_cats:
                    cat_map = {v: i for i, v in enumerate(ordered_cats)}
                    df[f"{ord_col}_ordinal"] = df[ord_col].map(cat_map)
                    st.session_state.df = df
                    st.success(f"✅ Created '{ord_col}_ordinal'")
                    st.rerun()

            with enc_tab4:
                bin_col = st.selectbox("Column", cat_cols_enc, key="bin_col")
                if st.button("✅ Apply Binary Encoding", key="bin_btn"):
                    le_b = LabelEncoder()
                    le_encoded = le_b.fit_transform(df[bin_col].astype(str))
                    n_bits = max(1, int(np.ceil(np.log2(len(le_b.classes_) + 1))))
                    for bit in range(n_bits):
                        df[f"{bin_col}_bin_{bit}"] = (le_encoded >> bit) & 1
                    st.session_state.df = df
                    st.success(f"✅ Binary encoded into {n_bits} bit columns")
                    st.rerun()

            with enc_tab5:
                st.info("Target encoding replaces category with mean of the target variable.")
                tgt_enc_col = st.selectbox("Feature Column", cat_cols_enc, key="tgt_enc_col")
                numeric_for_target = df.select_dtypes(include=['int64','float64']).columns.tolist()
                if numeric_for_target:
                    target_col_enc = st.selectbox("Target Column (numeric)",
                                                   numeric_for_target, key="target_col_enc")
                    smoothing = st.slider("Smoothing factor", 0.0, 10.0, 1.0, key="te_smooth")
                    if st.button("✅ Apply Target Encoding", key="te_btn"):
                        global_mean = df[target_col_enc].mean()
                        stats_df = df.groupby(tgt_enc_col)[target_col_enc].agg(['mean','count'])
                        smoothed = (stats_df['count'] * stats_df['mean'] + smoothing * global_mean) \
                                   / (stats_df['count'] + smoothing)
                        df[f"{tgt_enc_col}_target_enc"] = df[tgt_enc_col].map(smoothed)
                        st.session_state.df = df
                        st.success(f"✅ Created '{tgt_enc_col}_target_enc'")
                        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 – SCALING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_scale:
        st.markdown('<div class="section-header">📐 Feature Scaling</div>',
                    unsafe_allow_html=True)

        numeric_cols_sc = df.select_dtypes(include=['int64','float64']).columns.tolist()

        if not numeric_cols_sc:
            st.info("No numeric columns available.")
        else:
            scale_tab1, scale_tab2, scale_tab3, scale_tab4, scale_tab5 = st.tabs([
                "Min-Max Norm.", "Standard (Z)", "Robust Scaler",
                "MaxAbs Scaler", "Power / Quantile"
            ])

            def apply_scaler(scaler_obj, cols, suffix):
                df_temp = df.copy()
                df_temp[cols] = scaler_obj.fit_transform(df_temp[cols])
                for c in cols:
                    df[f"{c}_{suffix}"] = df_temp[c]
                st.session_state.df = df
                st.success(f"✅ Applied to {len(cols)} columns (suffix: _{suffix})")
                st.rerun()

            with scale_tab1:
                st.caption("Scales each feature to [0, 1] range. Sensitive to outliers.")
                norm_cols = st.multiselect("Columns", numeric_cols_sc, key="norm_cols",
                                           default=numeric_cols_sc[:3])
                feat_range = st.slider("Feature range", 0, 1, (0, 1), key="feat_range")
                in_place = st.checkbox("Replace original columns", key="norm_inplace")
                if st.button("✅ Apply Min-Max", key="norm_btn") and norm_cols:
                    scaler = MinMaxScaler(feature_range=feat_range)
                    if in_place:
                        df[norm_cols] = scaler.fit_transform(df[norm_cols])
                        st.session_state.df = df
                        st.success("✅ Normalized in-place")
                        st.rerun()
                    else:
                        apply_scaler(scaler, norm_cols, "norm")

            with scale_tab2:
                st.caption("Zero mean, unit variance. Best for normally distributed data.")
                std_cols = st.multiselect("Columns", numeric_cols_sc, key="std_cols",
                                          default=numeric_cols_sc[:3])
                in_place_std = st.checkbox("Replace original", key="std_inplace")
                if st.button("✅ Apply Standardization", key="std_btn") and std_cols:
                    scaler = StandardScaler()
                    if in_place_std:
                        df[std_cols] = scaler.fit_transform(df[std_cols])
                        st.session_state.df = df
                        st.success("✅ Standardized in-place")
                        st.rerun()
                    else:
                        apply_scaler(scaler, std_cols, "std")

            with scale_tab3:
                st.caption("Uses median & IQR. **Robust to outliers.** Best when outliers exist.")
                rob_cols = st.multiselect("Columns", numeric_cols_sc, key="rob_cols",
                                          default=numeric_cols_sc[:3])
                q_range  = st.slider("Quantile range (%)", 5, 45, (25, 75), key="rob_q")
                if st.button("✅ Apply Robust Scaler", key="rob_btn") and rob_cols:
                    apply_scaler(RobustScaler(quantile_range=q_range), rob_cols, "robust")

            with scale_tab4:
                st.caption("Scales by max absolute value. Keeps sparsity. Range [-1, 1].")
                mabs_cols = st.multiselect("Columns", numeric_cols_sc, key="mabs_cols",
                                           default=numeric_cols_sc[:3])
                if st.button("✅ Apply MaxAbs", key="mabs_btn") and mabs_cols:
                    apply_scaler(MaxAbsScaler(), mabs_cols, "maxabs")

            with scale_tab5:
                t_type = st.radio("Transformer type",
                    ["Yeo-Johnson Power", "Box-Cox Power", "Quantile (Uniform)",
                     "Quantile (Normal)"], key="pwr_type")
                pwr_cols = st.multiselect("Columns", numeric_cols_sc, key="pwr_cols",
                                          default=numeric_cols_sc[:3])
                if st.button("✅ Apply Transformer", key="pwr_btn") and pwr_cols:
                    try:
                        if t_type == "Yeo-Johnson Power":
                            apply_scaler(PowerTransformer(method='yeo-johnson'), pwr_cols, "yeojohn")
                        elif t_type == "Box-Cox Power":
                            apply_scaler(PowerTransformer(method='box-cox'), pwr_cols, "boxcox")
                        elif t_type == "Quantile (Uniform)":
                            apply_scaler(QuantileTransformer(output_distribution='uniform',
                                                             random_state=42), pwr_cols, "q_uni")
                        elif t_type == "Quantile (Normal)":
                            apply_scaler(QuantileTransformer(output_distribution='normal',
                                                             random_state=42), pwr_cols, "q_norm")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9 – FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_feature:
        st.markdown('<div class="section-header">🎯 Feature Engineering</div>',
                    unsafe_allow_html=True)

        numeric_cols_fe = df.select_dtypes(include=['int64','float64']).columns.tolist()

        fe_tab1, fe_tab2, fe_tab3 = st.tabs([
            "Feature Selection", "Dimensionality Reduction (PCA)",
            "Interaction & Polynomial Features"
        ])

        with fe_tab1:
            st.markdown("**Variance Threshold** — remove near-zero variance features")
            if numeric_cols_fe:
                vt_threshold = st.slider("Min variance threshold", 0.0, 1.0, 0.01, 0.01,
                                         key="vt_thresh")
                sel = VarianceThreshold(threshold=vt_threshold)
                try:
                    sel.fit(df[numeric_cols_fe].fillna(0))
                    dropped_vt = [c for c, s in zip(numeric_cols_fe, sel.get_support()) if not s]
                    kept_vt    = [c for c, s in zip(numeric_cols_fe, sel.get_support()) if s]
                    st.write(f"Will drop {len(dropped_vt)} columns: {dropped_vt}")
                    st.write(f"Will keep {len(kept_vt)} columns")
                    if st.button("✅ Remove Low Variance Features", key="vt_btn") and dropped_vt:
                        df = df.drop(columns=dropped_vt)
                        st.session_state.df = df
                        st.success(f"✅ Dropped {len(dropped_vt)} features")
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

                st.markdown("---")
                st.markdown("**SelectKBest** — supervised feature selection")
                target_fs = st.selectbox("Target column", df.columns.tolist(), key="fs_target")
                k_best    = st.slider("K best features", 1,
                                      min(20, len(numeric_cols_fe)), 5, key="k_best")
                fs_type   = st.radio("Problem type", ["Regression","Classification"], key="fs_type")
                if st.button("✅ Select K Best Features", key="kbest_btn"):
                    try:
                        feature_cols_fs = [c for c in numeric_cols_fe if c != target_fs]
                        X_fs = df[feature_cols_fs].fillna(0)
                        y_fs = df[target_fs].fillna(0)
                        score_fn = f_regression if fs_type == "Regression" else f_classif
                        selector = SelectKBest(score_func=score_fn, k=k_best)
                        selector.fit(X_fs, y_fs)
                        scores_df = pd.DataFrame({
                            'Feature': feature_cols_fs,
                            'Score':   selector.scores_,
                            'P-Value': selector.pvalues_
                        }).sort_values('Score', ascending=False)
                        st.dataframe(scores_df, use_container_width=True)
                        selected = scores_df.head(k_best)['Feature'].tolist()
                        st.success(f"Top {k_best} features: {selected}")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

        with fe_tab2:
            st.markdown("**PCA — Principal Component Analysis**")
            if len(numeric_cols_fe) >= 2:
                pca_cols   = st.multiselect("Select features for PCA",
                                            numeric_cols_fe,
                                            default=numeric_cols_fe[:min(10, len(numeric_cols_fe))],
                                            key="pca_cols")
                n_comp     = st.slider("Number of components", 1,
                                       min(len(pca_cols), 20) if pca_cols else 2,
                                       min(2, len(pca_cols)) if pca_cols else 2,
                                       key="pca_n")
                keep_orig  = st.checkbox("Keep original columns", value=True, key="pca_keep")

                if st.button("✅ Apply PCA", key="pca_btn") and pca_cols and n_comp <= len(pca_cols):
                    try:
                        X_pca = df[pca_cols].fillna(0)
                        X_scaled = StandardScaler().fit_transform(X_pca)
                        pca = PCA(n_components=n_comp, random_state=42)
                        pca_result = pca.fit_transform(X_scaled)
                        for i in range(n_comp):
                            df[f"PCA_{i+1}"] = pca_result[:, i]
                        if not keep_orig:
                            df = df.drop(columns=pca_cols)
                        st.session_state.df = df
                        evr = pca.explained_variance_ratio_
                        st.success(f"✅ PCA applied. Explained variance: "
                                   f"{[f'{v:.1%}' for v in evr]}")
                        # Scree plot
                        fig_scree = px.bar(x=[f"PC{i+1}" for i in range(n_comp)],
                                           y=evr * 100,
                                           labels={'x':'Component','y':'Explained Variance %'},
                                           title="PCA Scree Plot",
                                           template="plotly_white",
                                           color=evr * 100,
                                           color_continuous_scale='Blues')
                        fig_scree.update_layout(height=300, margin=dict(t=40,b=20))
                        st.plotly_chart(fig_scree, use_container_width=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("Need at least 2 numeric columns for PCA.")

        with fe_tab3:
            st.markdown("**Polynomial & Interaction Features**")
            if len(numeric_cols_fe) >= 2:
                poly_col_a = st.selectbox("Column A", numeric_cols_fe, key="poly_a")
                poly_col_b = st.selectbox("Column B", numeric_cols_fe, key="poly_b")
                poly_ops   = st.multiselect("Generate features",
                    ["A × B", "A + B", "A − B", "A²", "B²",
                     "A³", "log(A)", "log(B)", "√A", "√B",
                     "A / (B+1)", "(A−B)²"],
                    default=["A × B","A²","log(A)"], key="poly_ops")

                if st.button("✅ Generate Features", key="poly_btn") and poly_ops:
                    created = []
                    for op in poly_ops:
                        try:
                            a, b = df[poly_col_a], df[poly_col_b]
                            name_a, name_b = poly_col_a[:8], poly_col_b[:8]
                            if op == "A × B":    df[f"{name_a}x{name_b}"]   = a * b;  created.append(f"{name_a}x{name_b}")
                            elif op == "A + B":  df[f"{name_a}+{name_b}"]   = a + b;  created.append(f"{name_a}+{name_b}")
                            elif op == "A − B":  df[f"{name_a}-{name_b}"]   = a - b;  created.append(f"{name_a}-{name_b}")
                            elif op == "A²":     df[f"{name_a}_sq"]         = a ** 2; created.append(f"{name_a}_sq")
                            elif op == "B²":     df[f"{name_b}_sq"]         = b ** 2; created.append(f"{name_b}_sq")
                            elif op == "A³":     df[f"{name_a}_cu"]         = a ** 3; created.append(f"{name_a}_cu")
                            elif op == "log(A)": df[f"log_{name_a}"]        = np.log1p(a.clip(0)); created.append(f"log_{name_a}")
                            elif op == "log(B)": df[f"log_{name_b}"]        = np.log1p(b.clip(0)); created.append(f"log_{name_b}")
                            elif op == "√A":     df[f"sqrt_{name_a}"]       = np.sqrt(a.clip(0)); created.append(f"sqrt_{name_a}")
                            elif op == "√B":     df[f"sqrt_{name_b}"]       = np.sqrt(b.clip(0)); created.append(f"sqrt_{name_b}")
                            elif op == "A / (B+1)": df[f"{name_a}_div_{name_b}"] = a / (b + 1); created.append(f"{name_a}_div_{name_b}")
                            elif op == "(A−B)²": df[f"({name_a}-{name_b})_sq"]   = (a - b) ** 2; created.append(f"({name_a}-{name_b})_sq")
                        except Exception:
                            pass
                    st.session_state.df = df
                    st.success(f"✅ Created: {created}")
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 10 – ML PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ml:
        st.markdown('<div class="section-header">🤖 ML-Specific Preprocessing</div>',
                    unsafe_allow_html=True)

        ml_tab1, ml_tab2, ml_tab3, ml_tab4, ml_tab5 = st.tabs([
            "Train-Test Split", "Class Imbalance (SMOTE)",
            "Correlation Filter", "Normality Test",
            "Final ML Readiness Check"
        ])

        with ml_tab1:
            st.markdown("**Train / Validation / Test Split**")
            numeric_cols_ml = df.select_dtypes(include=['int64','float64']).columns.tolist()
            all_cols_ml     = df.columns.tolist()

            target_ml = st.selectbox("Target column (Y)", all_cols_ml, key="ml_target",
                                     index=len(all_cols_ml)-1)
            features_ml = st.multiselect("Feature columns (X)",
                [c for c in all_cols_ml if c != target_ml],
                default=[c for c in all_cols_ml if c != target_ml][:10],
                key="ml_features")

            use_val = st.checkbox("Include Validation Set", value=False, key="use_val")
            test_pct = st.slider("Test %", 5, 40, 20, key="test_pct")
            val_pct  = st.slider("Validation % (of remaining)", 5, 30, 15,
                                 key="val_pct") if use_val else 0
            random_seed = st.number_input("Random Seed", 0, 9999, 42, key="rand_seed")
            stratify_split = st.checkbox("Stratify split (classification only)",
                                         value=False, key="strat_split")

            if st.button("✅ Create Split", key="split_btn") and features_ml:
                from sklearn.model_selection import train_test_split
                try:
                    X = df[features_ml].select_dtypes(include=['int64','float64']).fillna(0)
                    y = df[target_ml]
                    strat = y if stratify_split else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_pct/100,
                        random_state=random_seed, stratify=strat)
                    if use_val:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train, y_train, test_size=val_pct/100,
                            random_state=random_seed)
                        st.session_state['ml_X_val'] = X_val
                        st.session_state['ml_y_val'] = y_val

                    st.session_state['ml_X_train'] = X_train
                    st.session_state['ml_X_test']  = X_test
                    st.session_state['ml_y_train'] = y_train
                    st.session_state['ml_y_test']  = y_test

                    total = len(X)
                    st.success("✅ Split complete!")
                    s1, s2, s3 = st.columns(3)
                    with s1: st.metric("Train", f"{len(X_train):,} ({len(X_train)/total:.0%})")
                    with s2: st.metric("Test",  f"{len(X_test):,}  ({len(X_test)/total:.0%})")
                    if use_val:
                        with s3: st.metric("Val", f"{len(X_val):,} ({len(X_val)/total:.0%})")
                except Exception as e:
                    st.error(f"❌ {str(e)}")

        with ml_tab2:
            st.markdown("**Class Imbalance Handling**")
            cat_target_cols = df.select_dtypes(include=['object','category']).columns.tolist()
            int_cols_imb    = df.select_dtypes(include=['int64']).columns.tolist()
            imb_target = st.selectbox("Target column (classification)",
                                      cat_target_cols + int_cols_imb, key="imb_target")

            if imb_target:
                vc = df[imb_target].value_counts()
                st.dataframe(vc.reset_index().rename(
                    columns={'index': 'Class', imb_target: 'Count'}),
                    use_container_width=True)
                fig_imb = px.bar(vc.reset_index(), x='index', y=imb_target,
                                 title="Class Distribution",
                                 template="plotly_white",
                                 color='index',
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig_imb.update_layout(height=280, margin=dict(t=40,b=20))
                st.plotly_chart(fig_imb, use_container_width=True)

                imb_method = st.selectbox("Resampling method",
                    ["Random Oversampling", "Random Undersampling",
                     "SMOTE (requires imbalanced-learn)"], key="imb_method")

                if st.button("✅ Apply Resampling", key="imb_btn"):
                    try:
                        feat_cols_imb = df.select_dtypes(include=['int64','float64']).columns
                        feat_cols_imb = [c for c in feat_cols_imb if c != imb_target]
                        X_imb = df[feat_cols_imb].fillna(0)
                        y_imb = df[imb_target]

                        if imb_method == "Random Oversampling":
                            max_count = vc.max()
                            parts = [df]
                            for cls, cnt in vc.items():
                                if cnt < max_count:
                                    oversampled = df[df[imb_target] == cls].sample(
                                        max_count - cnt, replace=True, random_state=42)
                                    parts.append(oversampled)
                            df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
                            st.session_state.df = df
                            st.success(f"✅ Oversampled to {len(df):,} rows")

                        elif imb_method == "Random Undersampling":
                            min_count = vc.min()
                            parts = [df[df[imb_target] == cls].sample(
                                min_count, random_state=42) for cls in vc.index]
                            df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
                            st.session_state.df = df
                            st.success(f"✅ Undersampled to {len(df):,} rows")

                        elif imb_method == "SMOTE (requires imbalanced-learn)":
                            from imblearn.over_sampling import SMOTE
                            le_imb = LabelEncoder()
                            y_enc  = le_imb.fit_transform(y_imb)
                            sm     = SMOTE(random_state=42)
                            X_res, y_res = sm.fit_resample(X_imb, y_enc)
                            df_res = pd.DataFrame(X_res, columns=feat_cols_imb)
                            df_res[imb_target] = le_imb.inverse_transform(y_res)
                            st.session_state.df = df_res
                            st.success(f"✅ SMOTE applied: {len(df_res):,} rows")
                        st.rerun()
                    except ImportError:
                        st.error("❌ Install imbalanced-learn: `pip install imbalanced-learn`")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

        with ml_tab3:
            st.markdown("**Correlation Filter** — remove highly correlated features")
            numeric_cols_corr = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if len(numeric_cols_corr) >= 2:
                corr_thresh = st.slider("Drop if |correlation| >", 0.5, 1.0, 0.9, 0.05,
                                        key="corr_thresh")
                corr_matrix = df[numeric_cols_corr].corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop_corr = [c for c in upper.columns if any(upper[c] > corr_thresh)]

                fig_corr = px.imshow(corr_matrix, text_auto=".2f",
                                     color_continuous_scale='RdBu_r',
                                     title="Correlation Heatmap",
                                     template="plotly_white",
                                     aspect="auto")
                fig_corr.update_layout(height=400, margin=dict(t=40,b=20))
                st.plotly_chart(fig_corr, use_container_width=True)

                if to_drop_corr:
                    st.warning(f"Highly correlated columns to drop: {to_drop_corr}")
                    if st.button("🗑️ Drop Correlated Columns", key="drop_corr_btn"):
                        df = df.drop(columns=to_drop_corr)
                        st.session_state.df = df
                        st.success(f"✅ Dropped {len(to_drop_corr)} columns")
                        st.rerun()
                else:
                    st.success("✅ No highly correlated pairs above threshold.")

        with ml_tab4:
            st.markdown("**Normality Tests** — check if features follow a normal distribution")
            numeric_cols_nt = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if numeric_cols_nt:
                nt_col = st.selectbox("Select column", numeric_cols_nt, key="nt_col")
                s_nt   = df[nt_col].dropna()

                nc1, nc2 = st.columns(2)
                with nc1:
                    fig_hist = px.histogram(df, x=nt_col, nbins=40,
                                            title=f"Distribution: {nt_col}",
                                            template="plotly_white",
                                            color_discrete_sequence=['#3498db'])
                    fig_hist.update_layout(height=260, margin=dict(t=40,b=20))
                    st.plotly_chart(fig_hist, use_container_width=True)
                with nc2:
                    fig_qq = go.Figure()
                    qq = stats.probplot(s_nt, dist="norm")
                    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1],
                                                mode='markers', name='Data',
                                                marker=dict(color='#3498db', size=4)))
                    x_line = np.array([qq[0][0].min(), qq[0][0].max()])
                    fig_qq.add_trace(go.Scatter(x=x_line,
                                                y=qq[1][1] + qq[1][0] * x_line,
                                                mode='lines', name='Normal',
                                                line=dict(color='red', dash='dash')))
                    fig_qq.update_layout(title="Q-Q Plot", template="plotly_white",
                                         height=260, margin=dict(t=40,b=20))
                    st.plotly_chart(fig_qq, use_container_width=True)

                try:
                    stat_sw, p_sw = shapiro(s_nt.sample(min(5000, len(s_nt)),
                                                         random_state=42))
                    stat_ks, p_ks = stats.kstest(s_nt, 'norm',
                                                   args=(s_nt.mean(), s_nt.std()))
                    t1, t2, t3, t4 = st.columns(4)
                    with t1: st.metric("Shapiro W",      f"{stat_sw:.4f}")
                    with t2: st.metric("Shapiro p-value", f"{p_sw:.4f}",
                                       delta="Normal" if p_sw > 0.05 else "Non-Normal",
                                       delta_color="normal" if p_sw > 0.05 else "inverse")
                    with t3: st.metric("KS Stat",        f"{stat_ks:.4f}")
                    with t4: st.metric("KS p-value",      f"{p_ks:.4f}",
                                       delta="Normal" if p_ks > 0.05 else "Non-Normal",
                                       delta_color="normal" if p_ks > 0.05 else "inverse")
                    sk_val = skew(s_nt)
                    ku_val = kurtosis(s_nt)
                    st.info(f"**Skewness:** {sk_val:.3f} "
                            f"({'Right-skewed' if sk_val > 0.5 else 'Left-skewed' if sk_val < -0.5 else 'Approx. symmetric'}) | "
                            f"**Kurtosis:** {ku_val:.3f} "
                            f"({'Leptokurtic (heavy tails)' if ku_val > 3 else 'Platykurtic (light tails)' if ku_val < 3 else 'Mesokurtic (normal)'})")
                except Exception as e:
                    st.warning(f"Could not run normality test: {str(e)}")

        with ml_tab5:
            st.markdown('<div class="section-header">✅ ML Readiness Check</div>',
                        unsafe_allow_html=True)

            checks = []
            missing_total = df.isnull().sum().sum()
            checks.append(("Missing Values",
                            "✅ None" if missing_total == 0 else f"❌ {missing_total:,} cells",
                            missing_total == 0))

            dup_total = df.duplicated().sum()
            checks.append(("Duplicate Rows",
                            "✅ None" if dup_total == 0 else f"⚠️ {dup_total:,} rows",
                            dup_total == 0))

            cat_remaining = df.select_dtypes(include='object').columns.tolist()
            checks.append(("Categorical Encoding",
                            "✅ All encoded" if not cat_remaining
                            else f"⚠️ {len(cat_remaining)} unencoded: {cat_remaining[:5]}",
                            not cat_remaining))

            numeric_cols_check = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if numeric_cols_check:
                ranges = df[numeric_cols_check].max() - df[numeric_cols_check].min()
                large_range = ranges[ranges > 1000].index.tolist()
                checks.append(("Feature Scaling",
                                "✅ Ranges OK" if not large_range
                                else f"⚠️ Large ranges in: {large_range[:5]}",
                                not large_range))
            else:
                checks.append(("Feature Scaling", "⚠️ No numeric columns", False))

            inf_count = np.isinf(df.select_dtypes(include=['int64','float64']).values).sum()
            checks.append(("Infinite Values",
                            "✅ None" if inf_count == 0 else f"❌ {inf_count} inf values",
                            inf_count == 0))

            score = sum(1 for _, _, ok in checks if ok)
            total = len(checks)

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f0f4f8,#fff);
                 padding:20px; border-radius:14px; margin-bottom:20px;
                 border-left:5px solid {'#2ecc71' if score == total else '#e67e22'};">
                <h3 style="margin:0; font-family:Sora,sans-serif; color:#1a2535;">
                    ML Readiness Score:
                    <span style="color:{'#2ecc71' if score==total else '#e67e22'};">
                        {score}/{total}
                    </span>
                </h3>
            </div>
            """, unsafe_allow_html=True)

            for name, msg, ok in checks:
                color = "#2ecc71" if ok else "#e74c3c" if "❌" in msg else "#e67e22"
                st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid {color};
                     padding:12px 16px; margin-bottom:8px;">
                    <strong style="color:#1a2535;">{name}</strong>
                    <span style="float:right; color:{color};">{msg}</span>
                </div>
                """, unsafe_allow_html=True)

            if score == total:
                st.success("🎉 Your dataset is ML-ready! Head to the ML Algorithm page.")
            else:
                st.warning("⚠️ Address the flagged issues above before training models.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 11 – PREVIEW & EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown('<div class="section-header">📤 Preview & Export</div>',
                    unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)
        with e1: st.metric("Rows",       f"{len(df):,}")
        with e2: st.metric("Columns",    f"{len(df.columns):,}")
        with e3:
            mp = (df.isnull().sum().sum() / (len(df)*len(df.columns))*100) if len(df) > 0 else 0
            st.metric("Missing %", f"{mp:.2f}%")
        with e4: st.metric("Duplicates", f"{df.duplicated().sum()}")

        st.markdown("---")
        st.markdown("**Preview (first 50 rows)**")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown("---")

        st.markdown("**Export Options**")
        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                data=csv_bytes,
                file_name=f"cleaned_{st.session_state.selected_file_name}.csv",
                mime="text/csv", use_container_width=True
            )

        with ex2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                "📥 Download Excel",
                data=excel_buf.getvalue(),
                file_name=f"cleaned_{st.session_state.selected_file_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with ex3:
            json_bytes = df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                "📥 Download JSON",
                data=json_bytes,
                file_name=f"cleaned_{st.session_state.selected_file_name}.json",
                mime="application/json", use_container_width=True
            )







# =========================================================
# === ABOUT PAGE ===========================================
# =========================================================

def about_page():
    st.markdown('<div class="main-header">ℹ️ About – DataMate AI</div>', unsafe_allow_html=True)

    st.subheader("Overview")
    st.write("""
    **DataMate AI** is a streamlined and intelligent data analysis platform built with Python and Streamlit.
    It enables users to upload datasets, clean and preprocess data, generate interactive visualizations,
    build machine learning models, and export reports — all within a single unified interface.
    """)
    st.write("---")

    st.subheader("What This Platform Offers")
    st.markdown("""
    ### **1. Data Upload**
    Supports CSV, Excel, JSON, TXT, XML, Parquet with automatic encoding detection.

    ### **2. Data Preprocessing**
    Missing value handling, duplicate removal, outlier detection, encoding, scaling, and more.

    ### **3. Interactive Visualizations**
    Plotly-powered EDA with univariate, bivariate, and multivariate charts.

    ### **4. Dashboard**
    Custom KPI cards, pinned charts, and HTML export.

    ### **5. Machine Learning**
    Guided regression & classification with feature selection and prediction interface.

    ### **6. Export Center**
    Generate professional HTML/PDF reports with automatic layout.

    ### **7. History & Status**
    Track uploads, preprocessing steps, and model training for full transparency.
    """)
    st.write("---")

    st.subheader("Co-Founders")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="founder-card">
            <h3 style="margin-bottom:8px; font-family: Sora, sans-serif;">Shahbaz Junaid</h3>
            <p style="margin:0; font-size:15px;">📧 <b>Email:</b> shahbazjunaid55@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="founder-card">
            <h3 style="margin-bottom:8px; font-family: Sora, sans-serif;">Rizwan Ullah</h3>
            <p style="margin:0; font-size:15px;">📧 <b>Email:</b> sulima9876@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.info("For support or feedback, feel free to contact any co-founder listed above.")


# =========================================================
# === SIDEBAR NAVIGATION ==================================
# =========================================================

with st.sidebar:
    st.markdown("""
    <div class='sidebar-header-card'>
        <h1 style='color: white; margin: 0; font-size: 22px; font-family: Sora, sans-serif; letter-spacing: 0.5px;'>
             DataMate AI
        </h1>
        <p style='color: rgba(255,255,255,0.7); margin: 6px 0 0 0; font-size: 12px;'></p>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        'Dashboard':          '📊',
        'Data Upload':        '📁',
        'Data Preprocessing': '🔄',
        'Visualization':      '📈',
        'ML Algorithm':       '🤖',
        'Export':             '💾',
        'History':            '🕒',
        'About':              'ℹ️'
    }

    for page_name, icon in pages.items():
        target = 'Welcome' if page_name == 'Home' else page_name
        if st.button(f"{icon}  {page_name}", key=page_name, use_container_width=True):
            st.session_state.current_page = target


# =========================================================
# === PAGE ROUTING ========================================
# =========================================================

def route_pages():
    if st.session_state.current_page == 'Welcome':
        welcome_page()
    elif st.session_state.current_page == 'Dashboard':
        dashboard_page()
    elif st.session_state.current_page == 'Data Upload':
        data_upload_page()
    elif st.session_state.current_page == 'Data Preprocessing':
        data_preprocessing_page()    
    elif st.session_state.current_page == 'About':
        about_page()
    else:
        # Placeholder for pages under construction
        st.markdown(f'<div class="main-header">{st.session_state.current_page}</div>', unsafe_allow_html=True)
        st.info(f"🚧 **{st.session_state.current_page}** page is coming soon! Stay tuned.")

st.markdown("<div style='text-align: center; color: #7f8c8d; padding: 20px; font-size: 12px;'></div>", unsafe_allow_html=True)

route_pages()