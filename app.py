import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Color Palette (Soft Light Theme) ───────────────────────────────────────────
PRIMARY     = "#D63B6E"   # vivid pink-rose  — buttons, active states
SECONDARY   = "#B02D5A"   # deep rose        — headings
ACCENT      = "#7B4FA6"   # medium purple    — accents
BG_MAIN     = "#FFF8FB"   # near-white blush — page background
BG_CARD     = "#FFFFFF"   # pure white cards
BG_CARD2    = "#FFF0F6"   # very light pink  — alt card
BENIGN_C    = "#1E8C5A"   # forest green     — benign
MALIGNANT_C = "#D63030"   # clear red        — malignant
TEXT_DARK   = "#1C1020"   # near-black text  — all body copy
TEXT_MID    = "#5A3A50"   # dark mauve       — sub-labels
BORDER_C    = "#F0B8D0"   # soft pink border

# aliases so chart code stays consistent
TEXT_LIGHT  = TEXT_DARK
TEXT_MUTED  = TEXT_MID
BG_DARK     = BG_CARD

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG_MAIN};
    color: {TEXT_DARK};
}}
.stApp {{
    background: linear-gradient(150deg, #FFF8FB 0%, #FFF0F6 55%, #FBF4FF 100%);
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FDE8F2 0%, #F8DCEB 100%);
    border-right: 2px solid {BORDER_C};
}}
[data-testid="stSidebar"] * {{
    color: {TEXT_DARK} !important;
}}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {SECONDARY} !important;
}}

/* ── Headings ── */
h1 {{
    font-family: 'Playfair Display', serif !important;
    color: {SECONDARY} !important;
    letter-spacing: 0.4px;
}}
h2 {{
    font-family: 'Playfair Display', serif !important;
    color: {TEXT_DARK} !important;
}}
h3, h4 {{
    font-family: 'DM Sans', sans-serif !important;
    color: {SECONDARY} !important;
    font-weight: 600;
}}
p, li, label, span {{
    color: {TEXT_DARK} !important;
}}

/* ── Metric Cards ── */
[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1.5px solid {BORDER_C};
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 3px 14px rgba(180,50,100,0.10);
}}
[data-testid="stMetric"] label {{
    color: {TEXT_MID} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {SECONDARY} !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: #FDE8F2;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1.5px solid {BORDER_C};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {TEXT_MID};
    font-weight: 500;
    padding: 8px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT}) !important;
    color: white !important;
    font-weight: 600 !important;
}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {{ accent-color: {PRIMARY}; }}

/* ── Button ── */
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(180,50,100,0.28);
    width: 100%;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(180,50,100,0.42);
}}

/* ── Selectbox ── */
.stSelectbox > div > div {{
    background: {BG_CARD} !important;
    border: 1.5px solid {BORDER_C} !important;
    border-radius: 8px !important;
    color: {TEXT_DARK} !important;
}}

/* ── Dataframe / Table ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_C};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Divider ── */
hr {{ border-color: {BORDER_C}; }}

/* ── Plotly transparent bg ── */
.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

/* ── Prediction result boxes ── */
.prediction-box-benign {{
    background: linear-gradient(135deg, #E8F8F0, #F5FFF9);
    border: 2px solid {BENIGN_C};
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 4px 18px rgba(30,140,90,0.12);
}}
.prediction-box-malignant {{
    background: linear-gradient(135deg, #FDF0F0, #FFF5F5);
    border: 2px solid {MALIGNANT_C};
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 4px 18px rgba(214,48,48,0.12);
}}

/* ── Ribbon header ── */
.ribbon-header {{
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT}, {PRIMARY});
    background-size: 200% 100%;
    animation: shimmer 3s infinite;
    padding: 22px 30px;
    border-radius: 16px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(180,50,100,0.22);
}}
@keyframes shimmer {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* ── Sidebar stat cards ── */
.stat-card {{
    background: {BG_CARD};
    border: 1.5px solid {BORDER_C};
    border-radius: 12px;
    padding: 16px 18px;
    margin: 8px 0;
    box-shadow: 0 2px 10px rgba(180,50,100,0.08);
}}
.stat-card p {{ color: {TEXT_DARK} !important; }}
</style>
""", unsafe_allow_html=True)


# ─── Load / Train Model ──────────────────────────────────────────────────────────

# ─── Load Resources (Model + Data) ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Load dataset
    df = pd.read_csv("breast_cancer.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    # Load saved model & scaler
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Train-test split (ONLY for metrics, not training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Transform test data
    X_test_s = scaler.transform(X_test)

    # Predictions for dashboard
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    return model, scaler, X, y, X_train, X_test, y_train, y_test, y_pred, y_proba, df


# Load everything
model, scaler, X, y, X_train, X_test, y_train, y_test, y_pred, y_proba, df = load_resources()

# Feature list
features = list(X.columns)

# Feature groups
MEAN_FEATURES  = [f for f in features if f.startswith("mean")]
ERROR_FEATURES = [f for f in features if f.endswith("error")]
WORST_FEATURES = [f for f in features if f.startswith("worst")]


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <span style='font-size:48px;'>🎗️</span>
        <h2 style='font-family:Playfair Display,serif; color:#B02D5A; margin:8px 0 4px;'>BreastScan AI</h2>
        <p style='color:#5A3A50; font-size:13px; margin:0;'>Powered by Decision Tree</p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["🏠 Dashboard", "🔬 Prediction", "📊 Data Explorer", "📈 Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"""
    <div class='stat-card'>
        <p style='color:#C9A8BE; font-size:12px; margin:0 0 4px;'>MODEL ACCURACY</p>
        <p style='font-family:Playfair Display,serif; font-size:2rem; color:#E8A0BF; margin:0;'>{acc*100:.1f}%</p>
    </div>
    <div class='stat-card'>
        <p style='color:#C9A8BE; font-size:12px; margin:0 0 4px;'>TOTAL SAMPLES</p>
        <p style='font-family:Playfair Display,serif; font-size:2rem; color:#E8A0BF; margin:0;'>{len(df)}</p>
    </div>
    <div class='stat-card'>
        <p style='color:#C9A8BE; font-size:12px; margin:0 0 4px;'>FEATURES USED</p>
        <p style='font-family:Playfair Display,serif; font-size:2rem; color:#E8A0BF; margin:0;'>{len(features)}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:30px; padding: 14px; background:rgba(192,57,43,0.1);
    border-radius:10px; border:1px solid rgba(192,57,43,0.3);'>
        <p style='color:#C9A8BE; font-size:11px; margin:0; text-align:center;'>
        ⚠️ For educational use only.<br>Always consult a medical professional.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class='ribbon-header'>
        <h1 style='margin:0; color:white; font-size:2.4rem;'>🎗️ Breast Cancer Prediction Dashboard</h1>
        <p style='color:rgba(255,255,255,0.8); margin:8px 0 0; font-size:15px;'>
            Early detection saves lives — AI-powered diagnostic assistance
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    benign_count    = int((y == 1).sum())
    malignant_count = int((y == 0).sum())
    train_acc = accuracy_score(y_train, model.predict(scaler.transform(X_train)))

    with col1:
        st.metric("🎯 Test Accuracy",    f"{acc*100:.1f}%")
    with col2:
        st.metric("🩺 Training Accuracy", f"{train_acc*100:.1f}%")
    with col3:
        st.metric("✅ Benign Cases",      f"{benign_count}")
    with col4:
        st.metric("⚠️ Malignant Cases",   f"{malignant_count}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Distribution Pie + Class Distribution Bar
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🥧 Diagnosis Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=["Malignant", "Benign"],
            values=[malignant_count, benign_count],
            hole=0.55,
            marker=dict(colors=[MALIGNANT_C, BENIGN_C],
                        line=dict(color=BG_DARK, width=3)),
            textfont=dict(color="white", size=14),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        fig_pie.add_annotation(text=f"<b>{len(df)}</b><br>Total",
                               x=0.5, y=0.5, font=dict(size=18, color=TEXT_LIGHT),
                               showarrow=False)
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_LIGHT)),
            margin=dict(t=10, b=10, l=10, r=10), height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### 📊 Feature Group Averages by Class")
        mean_cols = MEAN_FEATURES[:6]
        benign_means    = df[df.target==1][mean_cols].mean().values
        malignant_means = df[df.target==0][mean_cols].mean().values
        labels_short = [c.replace("mean ","") for c in mean_cols]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Benign",    x=labels_short, y=benign_means,
                                  marker_color=BENIGN_C, opacity=0.85))
        fig_bar.add_trace(go.Bar(name="Malignant", x=labels_short, y=malignant_means,
                                  marker_color=MALIGNANT_C, opacity=0.85))
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            margin=dict(t=10, b=10, l=10, r=10), height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 3: Correlation Heatmap
    st.markdown("#### 🔥 Feature Correlation Heatmap (Mean Features)")
    corr = df[MEAN_FEATURES + ["target"]].corr()
    fig_hm = px.imshow(
        corr,
        color_continuous_scale=[[0,"#FFF0F6"],[0.5,"#7B4FA6"],[1,"#D63B6E"]],
        text_auto=".2f",
        aspect="auto"
    )
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT, size=10),
        coloraxis_colorbar=dict(tickfont=dict(color=TEXT_LIGHT)),
        margin=dict(t=10, b=10, l=10, r=10), height=420
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Prediction":
    st.markdown("## 🔬 Predict Breast Cancer Diagnosis")
    st.markdown(
        "<p style='color:#C9A8BE;'>Adjust the feature sliders below and click Predict to get a diagnosis.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Input sliders in 3 tabs for cleanliness
    tab_m, tab_e, tab_w = st.tabs(["📐 Mean Features", "📏 Error Features", "⚠️ Worst Features"])

    input_vals = {}

    with tab_m:
        cols = st.columns(3)
        for i, feat in enumerate(MEAN_FEATURES):
            mn, mx, med = float(df[feat].min()), float(df[feat].max()), float(df[feat].median())
            input_vals[feat] = cols[i % 3].slider(feat, mn, mx, med, key=f"m_{feat}")

    with tab_e:
        cols = st.columns(3)
        for i, feat in enumerate(ERROR_FEATURES):
            mn, mx, med = float(df[feat].min()), float(df[feat].max()), float(df[feat].median())
            input_vals[feat] = cols[i % 3].slider(feat, mn, mx, med, key=f"e_{feat}")

    with tab_w:
        cols = st.columns(3)
        for i, feat in enumerate(WORST_FEATURES):
            mn, mx, med = float(df[feat].min()), float(df[feat].max()), float(df[feat].median())
            input_vals[feat] = cols[i % 3].slider(feat, mn, mx, med, key=f"w_{feat}")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Run Prediction")

    if predict_btn:
        input_df  = pd.DataFrame([input_vals])[features]
        scaled    = scaler.transform(input_df)
        pred      = model.predict(scaled)[0]
        proba     = model.predict_proba(scaled)[0]
        conf      = proba[pred] * 100

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class='prediction-box-benign'>
                    <div style='font-size:56px;'>✅</div>
                    <h2 style='color:{BENIGN_C}; font-family:Playfair Display,serif;'>Benign</h2>
                    <p style='color:#C9A8BE; font-size:15px;'>The tumor appears to be <b>non-cancerous</b>.</p>
                    <p style='color:{BENIGN_C}; font-size:28px; font-weight:700; margin:8px 0 0;'>{conf:.1f}% Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-box-malignant'>
                    <div style='font-size:56px;'>⚠️</div>
                    <h2 style='color:{MALIGNANT_C}; font-family:Playfair Display,serif;'>Malignant</h2>
                    <p style='color:#C9A8BE; font-size:15px;'>The tumor may be <b>cancerous</b>. Please consult a specialist.</p>
                    <p style='color:{MALIGNANT_C}; font-size:28px; font-weight:700; margin:8px 0 0;'>{conf:.1f}% Confidence</p>
                </div>
                """, unsafe_allow_html=True)

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[0] * 100,
                title={"text": "Malignancy Probability (%)", "font": {"color": TEXT_LIGHT, "size": 14}},
                number={"suffix": "%", "font": {"color": MALIGNANT_C, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED},
                    "bar": {"color": MALIGNANT_C if pred == 0 else BENIGN_C},
                    "bgcolor": BG_CARD2,
                    "borderwidth": 1,
                    "bordercolor": "rgba(192,57,43,0.3)",
                    "steps": [
                        {"range": [0, 40],   "color": "rgba(39,174,96,0.2)"},
                        {"range": [40, 70],  "color": "rgba(243,156,18,0.2)"},
                        {"range": [70, 100], "color": "rgba(231,76,60,0.2)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": 50}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_LIGHT),
                margin=dict(t=20, b=20, l=20, r=20), height=300
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Probability bar
        fig_prob = go.Figure(go.Bar(
            x=["Malignant", "Benign"],
            y=[proba[0]*100, proba[1]*100],
            marker=dict(
                color=[MALIGNANT_C, BENIGN_C],
                line=dict(color="rgba(255,255,255,0.1)", width=1)
            ),
            text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
            textposition="outside",
            textfont=dict(color=TEXT_LIGHT, size=14)
        ))
        fig_prob.update_layout(
            title=dict(text="Class Probability Breakdown", font=dict(color=TEXT_LIGHT)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT),
            yaxis=dict(range=[0, 115], gridcolor="rgba(0,0,0,0.07)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            margin=dict(t=40, b=10, l=10, r=10), height=280
        )
        st.plotly_chart(fig_prob, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")
    st.markdown("<p style='color:#C9A8BE;'>Explore the breast cancer dataset interactively.</p>",
                unsafe_allow_html=True)

    # Raw data
    with st.expander("📋 View Raw Dataset", expanded=False):
        df_show = df.copy()
        df_show["Diagnosis"] = df_show["target"].map({1: "✅ Benign", 0: "⚠️ Malignant"})
        st.dataframe(df_show.drop("target", axis=1), use_container_width=True, height=350)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📦 Box Plot — Feature by Class")
        box_feat = st.selectbox("Select Feature", features, index=0, key="box_feat")
        fig_box = px.box(
            df, x="target", y=box_feat,
            color="target",
            color_discrete_map={0: MALIGNANT_C, 1: BENIGN_C},
            labels={"target": "Diagnosis (0=Malignant, 1=Benign)"},
            points="outliers"
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            margin=dict(t=10, b=10, l=10, r=10), height=350
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown("#### 🔵 Scatter Plot — Feature vs Feature")
        feat_x = st.selectbox("X Axis", features, index=0, key="sc_x")
        feat_y = st.selectbox("Y Axis", features, index=2, key="sc_y")
        fig_sc = px.scatter(
            df, x=feat_x, y=feat_y, color="target",
            color_discrete_map={0: MALIGNANT_C, 1: BENIGN_C},
            opacity=0.7, size_max=8,
            labels={"target": "0=Malignant / 1=Benign"}
        )
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_LIGHT)),
            xaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            margin=dict(t=10, b=10, l=10, r=10), height=350
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Histogram
    st.markdown("#### 📉 Feature Distribution")
    hist_feat = st.selectbox("Select Feature for Histogram", features, index=0, key="hist_f")
    fig_hist = px.histogram(
        df, x=hist_feat, color="target",
        color_discrete_map={0: MALIGNANT_C, 1: BENIGN_C},
        barmode="overlay", opacity=0.75, nbins=40,
        labels={"target": "0=Malignant / 1=Benign"}
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_LIGHT)),
        xaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
        margin=dict(t=10, b=10, l=10, r=10), height=320
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistical Summary
    st.markdown("#### 📐 Descriptive Statistics")
    tab_b, tab_mal = st.tabs(["✅ Benign", "⚠️ Malignant"])
    with tab_b:
        st.dataframe(df[df.target==1].drop("target",axis=1).describe().round(3),
                     use_container_width=True)
    with tab_mal:
        st.dataframe(df[df.target==0].drop("target",axis=1).describe().round(3),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.markdown("## 📈 Model Performance & Insights")
    st.markdown("<p style='color:#C9A8BE;'>Detailed evaluation of the Decision Tree Classifier.</p>",
                unsafe_allow_html=True)

    # ── Metrics Row ──
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["Malignant","Benign"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc*100:.1f}%")
    c2.metric("Precision (Benign)",  f"{report['Benign']['precision']*100:.1f}%")
    c3.metric("Recall (Benign)",     f"{report['Benign']['recall']*100:.1f}%")
    c4.metric("F1-Score (Benign)",   f"{report['Benign']['f1-score']*100:.1f}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns(2)

    # ── Confusion Matrix ──
    with row1_c1:
        st.markdown("#### 🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Malignant", "Benign"],
            y=["Malignant", "Benign"],
            text_auto=True,
            color_continuous_scale=[[0, BG_CARD],[0.5, ACCENT],[1, PRIMARY]]
        )
        fig_cm.update_traces(textfont=dict(size=22, color="white"))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10), height=340
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curve ──
    with row1_c2:
        st.markdown("#### 📉 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Decision Tree (AUC = {roc_auc:.3f})",
            line=dict(color=PRIMARY, width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            name="Random Baseline",
            line=dict(color=TEXT_MUTED, width=1.5, dash="dash")
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_LIGHT)),
            xaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
            margin=dict(t=10, b=10, l=10, r=10), height=340
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature Importance ──
    st.markdown("#### 🏆 Top Feature Importances")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(15)

    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[[0, ACCENT],[0.5, PRIMARY],[1, "#FF6B6B"]]
    )
    fig_imp.update_layout(
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0.07)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT),
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10), height=440
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Classification Report Table ──
    st.markdown("#### 📋 Full Classification Report")
    cr_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(cr_df, use_container_width=True)