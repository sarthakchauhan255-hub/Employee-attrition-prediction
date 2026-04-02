"""
app.py — AttritionIQ  ·  Premium Dark Glassmorphism Edition
──────────────────────────────────────────────────────────────
Run:  streamlit run app.py
Pre:  python model.py  (generates ./model_artifacts/)
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AttritionIQ",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════
C = dict(
    bg        = "#080c14",
    surface   = "#0d1220",
    glass     = "rgba(255,255,255,0.035)",
    glass2    = "rgba(255,255,255,0.06)",
    border    = "rgba(255,255,255,0.08)",
    border2   = "rgba(255,255,255,0.14)",
    cyan      = "#00e5ff",
    violet    = "#8b5cf6",
    rose      = "#f43f5e",
    emerald   = "#10b981",
    amber     = "#f59e0b",
    text      = "#f1f5f9",
    text2     = "#94a3b8",
    text3     = "#475569",
)

# ══════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Fira+Code:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif !important;
    background: {C['bg']} !important;
    color: {C['text']};
}}

/* App shell with ambient glow */
.stApp {{
    background:
        radial-gradient(ellipse 70% 50% at 5% 0%,  rgba(139,92,246,0.07) 0%, transparent 65%),
        radial-gradient(ellipse 60% 40% at 95% 100%, rgba(0,229,255,0.05) 0%, transparent 65%),
        {C['bg']};
    background-attachment: fixed;
    min-height: 100vh;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,rgba(11,15,26,0.97) 0%,rgba(8,12,20,0.98) 100%) !important;
    border-right: 1px solid {C['border']} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:rgba(139,92,246,0.45); border-radius:99px; }}

/* ── Glass cards ── */
.glass {{
    background: {C['glass']};
    border: 1px solid {C['border']};
    border-radius: 20px;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.38), 0 1px 0 rgba(255,255,255,0.055) inset;
    position: relative;
    overflow: hidden;
    transition: transform .25s cubic-bezier(.34,1.56,.64,1), box-shadow .25s ease, border-color .25s ease;
}}
.glass::after {{
    content:'';
    position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);
}}
.glass:hover {{
    transform: translateY(-3px);
    box-shadow: 0 18px 50px rgba(0,0,0,0.48), 0 0 0 1px {C['border2']};
    border-color: {C['border2']};
}}
.glass-cyan   {{ border-color:rgba(0,229,255,0.2);   box-shadow:0 8px 32px rgba(0,0,0,0.36),0 0 40px rgba(0,229,255,0.04); }}
.glass-violet {{ border-color:rgba(139,92,246,0.2);  box-shadow:0 8px 32px rgba(0,0,0,0.36),0 0 40px rgba(139,92,246,0.05); }}
.glass-rose   {{ border-color:rgba(244,63,94,0.2);   box-shadow:0 8px 32px rgba(0,0,0,0.36),0 0 40px rgba(244,63,94,0.04); }}
.glass-emerald{{ border-color:rgba(16,185,129,0.2);  box-shadow:0 8px 32px rgba(0,0,0,0.36),0 0 40px rgba(16,185,129,0.04); }}
.glass-cyan::before    {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,{C['cyan']},transparent);    opacity:.55; border-radius:20px 20px 0 0; }}
.glass-violet::before  {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,{C['violet']},transparent);  opacity:.55; border-radius:20px 20px 0 0; }}
.glass-rose::before    {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,{C['rose']},transparent);    opacity:.55; border-radius:20px 20px 0 0; }}
.glass-emerald::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,{C['emerald']},transparent); opacity:.55; border-radius:20px 20px 0 0; }}

/* ── Stat/KPI cards ── */
.stat-grid {{
    display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem;
}}
.stat {{
    background:{C['glass']}; border:1px solid {C['border']}; border-radius:18px;
    padding:1.4rem 1.6rem; backdrop-filter:blur(24px); position:relative;
    overflow:hidden; transition:all .3s cubic-bezier(.34,1.56,.64,1); cursor:default;
}}
.stat:hover {{ transform:translateY(-4px) scale(1.015); }}
.stat-icon {{
    font-size:2.1rem; margin-bottom:.5rem; display:block;
    filter:drop-shadow(0 0 14px currentColor);
    animation:float 3.2s ease-in-out infinite;
}}
.stat:nth-child(2) .stat-icon {{ animation-delay:.45s; }}
.stat:nth-child(3) .stat-icon {{ animation-delay:.9s; }}
.stat:nth-child(4) .stat-icon {{ animation-delay:1.35s; }}
@keyframes float {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-5px)}} }}
.stat-value {{
    font-family:'Outfit',sans-serif; font-size:2.3rem; font-weight:900;
    line-height:1; letter-spacing:-.04em;
}}
.stat-label {{ font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.13em; color:{C['text3']}; margin-top:.35rem; }}
.stat-sub   {{ font-size:.74rem; color:{C['text2']}; margin-top:.2rem; }}
.stat-c .stat-value{{ color:{C['cyan']};    }} .stat-c {{ border-color:rgba(0,229,255,0.15);   }} .stat-c:hover  {{ border-color:rgba(0,229,255,0.4);   box-shadow:0 16px 42px rgba(0,0,0,0.42),0 0 28px rgba(0,229,255,0.1); }}
.stat-v .stat-value{{ color:#a78bfa;         }} .stat-v {{ border-color:rgba(139,92,246,0.15);  }} .stat-v:hover  {{ border-color:rgba(139,92,246,0.4);  box-shadow:0 16px 42px rgba(0,0,0,0.42),0 0 28px rgba(139,92,246,0.1); }}
.stat-e .stat-value{{ color:{C['emerald']};  }} .stat-e {{ border-color:rgba(16,185,129,0.15);  }} .stat-e:hover  {{ border-color:rgba(16,185,129,0.4);  box-shadow:0 16px 42px rgba(0,0,0,0.42),0 0 28px rgba(16,185,129,0.1); }}
.stat-r .stat-value{{ color:{C['rose']};     }} .stat-r {{ border-color:rgba(244,63,94,0.15);   }} .stat-r:hover  {{ border-color:rgba(244,63,94,0.4);   box-shadow:0 16px 42px rgba(0,0,0,0.42),0 0 28px rgba(244,63,94,0.1); }}

/* ── Typography ── */
.hero-title {{
    font-family:'Outfit',sans-serif; font-size:clamp(2rem,4vw,3.2rem);
    font-weight:900; letter-spacing:-.04em; line-height:1.05;
    background:linear-gradient(135deg,#fff 0%,#fff 40%,{C['cyan']} 72%,{C['violet']} 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin-bottom:.4rem;
}}
.hero-sub  {{ font-size:.94rem; color:{C['text3']}; font-weight:400; letter-spacing:.02em; margin-bottom:2rem; }}
.sec-head  {{
    font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.18em;
    color:{C['text3']}; margin-bottom:.75rem; display:flex; align-items:center; gap:.5rem;
}}
.sec-head::before {{
    content:''; display:inline-block; width:16px; height:1px;
    background:{C['cyan']}; box-shadow:0 0 6px {C['cyan']};
}}

/* ── Metric pills ── */
.metric-row {{ display:flex; flex-wrap:wrap; gap:.6rem; margin-bottom:1.5rem; }}
.mpill {{
    background:{C['glass']}; border:1px solid {C['border']}; border-radius:99px;
    padding:.42rem 1rem; font-size:.78rem; font-weight:600; backdrop-filter:blur(12px);
    white-space:nowrap; transition:all .2s ease;
}}
.mpill:hover {{ transform:translateY(-1px); border-color:{C['border2']}; }}
.mpill-c {{ color:{C['cyan']};    border-color:rgba(0,229,255,0.2);   background:rgba(0,229,255,0.06); }}
.mpill-v {{ color:#a78bfa;         border-color:rgba(139,92,246,0.2);  background:rgba(139,92,246,0.06); }}
.mpill-e {{ color:{C['emerald']};  border-color:rgba(16,185,129,0.2);  background:rgba(16,185,129,0.06); }}
.mpill-r {{ color:{C['rose']};     border-color:rgba(244,63,94,0.2);   background:rgba(244,63,94,0.06); }}
.mpill-a {{ color:{C['amber']};    border-color:rgba(245,158,11,0.2);  background:rgba(245,158,11,0.06); }}

/* ── Prediction result ── */
.pred-result {{ border-radius:20px; padding:2.1rem; text-align:center; position:relative; overflow:hidden; }}
.pred-high {{
    background:radial-gradient(ellipse at center,rgba(244,63,94,0.12) 0%,transparent 70%);
    border:1px solid rgba(244,63,94,0.3);
    box-shadow:0 0 60px rgba(244,63,94,0.07), inset 0 1px 0 rgba(255,255,255,0.04);
}}
.pred-low  {{
    background:radial-gradient(ellipse at center,rgba(16,185,129,0.12) 0%,transparent 70%);
    border:1px solid rgba(16,185,129,0.3);
    box-shadow:0 0 60px rgba(16,185,129,0.07), inset 0 1px 0 rgba(255,255,255,0.04);
}}
.pred-icon {{
    font-size:3.2rem; display:block; margin-bottom:.8rem;
    animation:pulse-i 2.2s ease-in-out infinite; filter:drop-shadow(0 0 22px currentColor);
}}
@keyframes pulse-i {{ 0%,100%{{transform:scale(1)}} 50%{{transform:scale(1.1)}} }}
.pred-title {{ font-family:'Outfit',sans-serif; font-size:1.5rem; font-weight:900; letter-spacing:-.02em; margin-bottom:.45rem; }}
.pred-high .pred-title{{ color:{C['rose']};    }}
.pred-low  .pred-title{{ color:{C['emerald']}; }}
.pred-body {{ font-size:.82rem; color:{C['text3']}; line-height:1.65; }}

/* Risk bar */
.risk-bar-label  {{ display:flex; justify-content:space-between; font-size:.68rem; color:{C['text3']}; margin-bottom:.5rem; text-transform:uppercase; letter-spacing:.1em; }}
.risk-bar-track  {{ height:8px; background:rgba(255,255,255,0.055); border-radius:99px; overflow:hidden; position:relative; }}
.risk-bar-fill   {{ height:100%; border-radius:99px; position:relative; transition:width 1s cubic-bezier(.4,0,.2,1); }}
.risk-bar-fill::after {{ content:''; position:absolute; top:0; right:0; width:14px; height:100%; background:white; border-radius:99px; opacity:.45; filter:blur(4px); }}

/* Form section headings */
.fsec-title {{
    font-size:.67rem; font-weight:700; text-transform:uppercase; letter-spacing:.18em;
    color:{C['text3']}; margin-bottom:1.1rem; display:flex; align-items:center; gap:.5rem;
}}
.fsec-icon {{ font-size:1rem; animation:float 3s ease-in-out infinite; }}

/* Sidebar stat row */
.sstat {{
    display:flex; justify-content:space-between; align-items:center;
    padding:.5rem .7rem; border-radius:10px; margin-bottom:.3rem;
    background:rgba(255,255,255,0.025); border:1px solid {C['border']};
    transition:all .2s ease;
}}
.sstat:hover {{ background:rgba(255,255,255,0.04); border-color:{C['border2']}; }}
.sstat-lbl {{ font-size:.68rem; color:{C['text3']}; text-transform:uppercase; letter-spacing:.1em; font-weight:600; }}
.sstat-val {{ font-family:'Outfit',sans-serif; font-weight:800; font-size:.9rem; }}
.sstat-val.c {{ color:{C['cyan']};    text-shadow:0 0 12px {C['cyan']};    }}
.sstat-val.v {{ color:#a78bfa;         text-shadow:0 0 12px #a78bfa;         }}
.sstat-val.r {{ color:{C['rose']};     text-shadow:0 0 12px {C['rose']};     }}
.sstat-val.e {{ color:{C['emerald']};  text-shadow:0 0 12px {C['emerald']};  }}

/* Streamlit widget overrides */
[data-testid="stSlider"]>div>div>div>div {{ background:linear-gradient(90deg,{C['violet']},{C['cyan']}) !important; }}
[data-testid="stSlider"]>div>div>div>div>div {{ background:white !important; box-shadow:0 0 14px {C['cyan']} !important; }}
.stSelectbox [data-baseweb="select"]>div {{ background:rgba(255,255,255,0.04) !important; border:1px solid {C['border2']} !important; border-radius:10px !important; color:{C['text']} !important; font-family:'Outfit',sans-serif !important; }}
.stSelectbox [data-baseweb="select"]>div:focus-within {{ border-color:rgba(0,229,255,0.4) !important; box-shadow:0 0 0 2px rgba(0,229,255,0.1) !important; }}
[data-testid="stNumberInput"] input {{ background:rgba(255,255,255,0.04) !important; border:1px solid {C['border2']} !important; border-radius:10px !important; color:{C['text']} !important; font-family:'Fira Code',monospace !important; }}
[data-testid="stNumberInput"] input:focus {{ border-color:rgba(0,229,255,0.4) !important; box-shadow:0 0 0 2px rgba(0,229,255,0.1) !important; }}
.stButton>button {{ width:100%; background:linear-gradient(135deg,{C['violet']},{C['cyan']}) !important; color:white !important; border:none !important; border-radius:12px !important; padding:.8rem 2rem !important; font-family:'Outfit',sans-serif !important; font-size:.9rem !important; font-weight:700 !important; letter-spacing:.08em !important; text-transform:uppercase !important; box-shadow:0 4px 24px rgba(139,92,246,0.3) !important; transition:all .25s ease !important; }}
.stButton>button:hover {{ transform:translateY(-2px) !important; box-shadow:0 8px 36px rgba(139,92,246,0.48) !important; filter:brightness(1.1) !important; }}
[data-testid="stMetric"] {{ background:{C['glass']} !important; border:1px solid {C['border']} !important; border-radius:14px !important; padding:1rem 1.2rem !important; backdrop-filter:blur(16px) !important; }}
[data-testid="stMetricValue"] {{ font-family:'Outfit',sans-serif !important; font-size:1.85rem !important; font-weight:900 !important; color:{C['cyan']} !important; letter-spacing:-.03em !important; }}
[data-testid="stMetricLabel"] {{ font-size:.65rem !important; text-transform:uppercase !important; letter-spacing:.12em !important; color:{C['text3']} !important; font-family:'Outfit',sans-serif !important; font-weight:700 !important; }}
[data-testid="stDataFrame"] {{ border-radius:14px !important; overflow:hidden; border:1px solid {C['border']} !important; }}
[data-testid="stFileUploader"] {{ background:{C['glass']} !important; border:1.5px dashed {C['border2']} !important; border-radius:16px !important; padding:1rem !important; backdrop-filter:blur(16px) !important; transition:border-color .2s ease !important; }}
[data-testid="stFileUploader"]:hover {{ border-color:rgba(0,229,255,0.3) !important; }}
#MainMenu, footer, header {{ visibility:hidden; }}
[data-testid="stForm"] {{ background:transparent !important; border:none !important; }}
.block-container {{ padding-top:2rem !important; padding-bottom:3rem !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════
def mpl_dark():
    plt.rcParams.update({
        "figure.facecolor"  : "#0d1220",
        "axes.facecolor"    : "#0d1220",
        "axes.edgecolor"    : "rgba(255,255,255,0.06)",
        "axes.labelcolor"   : "#64748b",
        "axes.titlecolor"   : "#f1f5f9",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "xtick.color"       : "#475569",
        "ytick.color"       : "#475569",
        "grid.color"        : "rgba(255,255,255,0.05)",
        "grid.linestyle"    : "-",
        "grid.linewidth"    : 0.5,
        "text.color"        : "#f1f5f9",
        "legend.facecolor"  : "#111827",
        "legend.edgecolor"  : "rgba(255,255,255,0.08)",
        "legend.labelcolor" : "#94a3b8",
        "axes.labelsize"    : 9,
        "xtick.labelsize"   : 8,
        "ytick.labelsize"   : 8,
        "axes.titlesize"    : 11,
        "axes.titleweight"  : "bold",
        "axes.axisbelow"    : True,
        "figure.dpi"        : 120,
    })

mpl_dark()
CMAP_DUAL   = LinearSegmentedColormap.from_list("d", [C["rose"],"#1a1f30",C["cyan"]])
CMAP_VIOLET = LinearSegmentedColormap.from_list("v", ["#080c14",C["violet"]])


# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
ARTIFACT_DIR     = "model_artifacts"
DATA_PATH        = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CATEGORICAL_COLS = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
DROP_COLS        = ["EmployeeCount","Over18","StandardHours","EmployeeNumber"]
OPTIONS = dict(
    BusinessTravel = ["Non-Travel","Travel_Rarely","Travel_Frequently"],
    Department     = ["Human Resources","Research & Development","Sales"],
    EducationField = ["Human Resources","Life Sciences","Marketing","Medical","Other","Technical Degree"],
    Gender         = ["Female","Male"],
    JobRole        = ["Healthcare Representative","Human Resources","Laboratory Technician",
                      "Manager","Manufacturing Director","Research Director",
                      "Research Scientist","Sales Executive","Sales Representative"],
    MaritalStatus  = ["Divorced","Married","Single"],
    OverTime       = ["No","Yes"],
)


# ══════════════════════════════════════════════════════════
# AUTO-TRAIN  (runs on Streamlit Cloud where no pkl exists)
# ══════════════════════════════════════════════════════════
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, classification_report)
from sklearn.utils.class_weight import compute_sample_weight

@st.cache_resource(show_spinner=False)
def get_model_and_data():
    """
    Load pre-trained artifacts if they exist (local dev),
    otherwise train the model on-the-fly (Streamlit Cloud).
    Results are cached so training only happens once per session.
    """
    needed = ["model.pkl","label_encoders.pkl","feature_names.pkl","metrics.pkl"]
    artifacts_ready = all(os.path.exists(os.path.join(ARTIFACT_DIR,f)) for f in needed)

    # ── Load CSV ──────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        return None, None, None, None, None

    df_raw = pd.read_csv(DATA_PATH)

    # ── Try loading pre-built artifacts ───────────────────
    if artifacts_ready:
        def _p(n):
            with open(os.path.join(ARTIFACT_DIR,n),"rb") as f: return pickle.load(f)
        return (df_raw,
                _p("model.pkl"),
                _p("label_encoders.pkl"),
                _p("feature_names.pkl"),
                _p("metrics.pkl"))

    # ── Train from scratch (Streamlit Cloud path) ─────────
    df = df_raw.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    sw = compute_sample_weight("balanced", y_train)

    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sw)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cv_scores    = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    metrics = dict(
        accuracy   = accuracy_score(y_test, y_pred),
        precision  = precision_score(y_test, y_pred),
        recall     = recall_score(y_test, y_pred),
        f1         = f1_score(y_test, y_pred),
        roc_auc    = roc_auc_score(y_test, y_prob),
        cm         = confusion_matrix(y_test, y_pred),
        fpr=fpr, tpr=tpr,
        report     = classification_report(y_test, y_pred, output_dict=True),
        cv_mean    = cv_scores.mean(),
        cv_std     = cv_scores.std(),
        feature_names = feature_names,
        importances   = model.feature_importances_,
    )

    # Save for subsequent runs if writable
    try:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        for fname, obj in [("model.pkl",model),("label_encoders.pkl",label_encoders),
                            ("feature_names.pkl",feature_names),("metrics.pkl",metrics)]:
            with open(os.path.join(ARTIFACT_DIR,fname),"wb") as f:
                pickle.dump(obj, f)
    except Exception:
        pass  # read-only filesystem on Cloud — that's fine, cache keeps it in memory

    return df_raw, model, label_encoders, feature_names, metrics


# ── Spinner shown only on first load ──────────────────────
spinner_ph = st.empty()
with spinner_ph.container():
    st.markdown(f"""
    <div style='min-height:60vh;display:flex;flex-direction:column;
                align-items:center;justify-content:center;gap:1rem;text-align:center;'>
      <div style='font-size:3rem;'>⚗️</div>
      <div style='font-family:"Outfit",sans-serif;font-size:1.4rem;font-weight:800;
                  color:{C["cyan"]};'>Initialising AttritionIQ…</div>
      <div style='color:{C["text3"]};font-size:.85rem;'>
        Training model on first launch. This takes ~20 seconds.</div>
    </div>
    """, unsafe_allow_html=True)

df_raw, model, label_encoders, feature_names, metrics = get_model_and_data()
spinner_ph.empty()   # remove the spinner once done

if df_raw is None:
    st.error("Dataset CSV not found. Ensure `WA_Fn-UseC_-HR-Employee-Attrition.csv` "
             "is committed to your repository.")
    st.stop()


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:1.5rem 1.2rem 1rem;border-bottom:1px solid {C["border"]};margin-bottom:.8rem;'>
      <div style='font-family:"Outfit",sans-serif;font-size:1.45rem;font-weight:900;
                  letter-spacing:-.04em;background:linear-gradient(135deg,#fff,{C["cyan"]});
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
        AttritionIQ
      </div>
      <div style='font-size:.67rem;color:{C["text3"]};text-transform:uppercase;
                  letter-spacing:.16em;margin-top:.15rem;'>HR Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("",
        ["⬡  Dashboard","◈  Analytics","◉  Model Insights","◎  Predict","▦  Batch Scan"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Live model stats
    st.markdown('<div class="sec-head" style="padding-left:.5rem;">Live Stats</div>', unsafe_allow_html=True)
    stat_rows = [
        (f"{metrics['accuracy']*100:.1f}%", "Accuracy",       "c"),
        (f"{metrics['roc_auc']*100:.1f}%",  "ROC-AUC",        "v"),
        (f"{(df_raw['Attrition']=='Yes').mean()*100:.1f}%", "Attrition Rate","r"),
        (f"{len(df_raw):,}",                "Employees",      "e"),
    ]
    for val, lbl, cls in stat_rows:
        st.markdown(f"""
        <div class="sstat">
          <span class="sstat-lbl">{lbl}</span>
          <span class="sstat-val {cls}">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head" style="padding-left:.5rem;">Model Config</div>', unsafe_allow_html=True)
    cfg = [("Algorithm","Gradient Boosting"),("Estimators","300"),
           ("Learning Rate","0.05"),("Max Depth","4"),("Features","30")]
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;padding:.32rem 0;'
        f'border-bottom:1px solid {C["border"]};font-size:.71rem;">'
        f'<span style="color:{C["text3"]};">{k}</span>'
        f'<span style="font-family:\'Fira Code\',monospace;color:{C["text2"]};">{v}</span></div>'
        for k,v in cfg
    ])
    st.markdown(f'<div style="padding:.15rem 0 0;">{rows_html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════
def sec_head(label):
    st.markdown(f'<div class="sec-head">{label}</div>', unsafe_allow_html=True)

def new_fig(w=8, h=4):
    return plt.subplots(figsize=(w, h))

DIVIDER = "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0;'>"


# ══════════════════════════════════════════════════════════════════
# PAGE ①  DASHBOARD
# ══════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown('<div class="hero-title">Workforce Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-time attrition risk analytics powered by gradient boosting</div>', unsafe_allow_html=True)

    total   = len(df_raw)
    left    = (df_raw["Attrition"]=="Yes").sum()
    stayed  = total-left
    avg_inc = int(df_raw["MonthlyIncome"].mean())

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat stat-c">
        <span class="stat-icon" style="color:{C['cyan']};">🧑‍💼</span>
        <div class="stat-value">{total:,}</div>
        <div class="stat-label">Total Employees</div>
        <div class="stat-sub">in dataset</div>
      </div>
      <div class="stat stat-r">
        <span class="stat-icon" style="color:{C['rose']};">📉</span>
        <div class="stat-value">{left}</div>
        <div class="stat-label">Attrition Cases</div>
        <div class="stat-sub">{left/total*100:.1f}% of workforce</div>
      </div>
      <div class="stat stat-e">
        <span class="stat-icon" style="color:{C['emerald']};">🏢</span>
        <div class="stat-value">{stayed/total*100:.0f}%</div>
        <div class="stat-label">Retention Rate</div>
        <div class="stat-sub">{stayed:,} employees retained</div>
      </div>
      <div class="stat stat-v">
        <span class="stat-icon" style="color:#a78bfa;">💰</span>
        <div class="stat-value">${avg_inc:,}</div>
        <div class="stat-label">Avg Monthly Income</div>
        <div class="stat-sub">across all roles</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.7])
    with c1:
        sec_head("Attrition Split")
        f1, ax1 = new_fig(4.5, 4.2)
        vals   = df_raw["Attrition"].value_counts()
        cols_p = [C["emerald"], C["rose"]]
        wedges, _, autotexts = ax1.pie(
            vals.values, labels=None, autopct="%1.1f%%", colors=cols_p,
            startangle=90, wedgeprops=dict(width=0.52,edgecolor="#0d1220",linewidth=3),
            pctdistance=0.76)
        for at in autotexts: at.set(color=C["text"],fontsize=13,fontweight="bold")
        ax1.legend([mpatches.Patch(color=c,label=l) for c,l in zip(cols_p,vals.index)],
                   loc="lower center",ncol=2,frameon=False,fontsize=9,labelcolor=C["text2"])
        ax1.set_title("Overall Attrition",pad=14)
        st.pyplot(f1,use_container_width=True); plt.close()

    with c2:
        sec_head("Attrition Rate by Department")
        dept = (df_raw.groupby("Department")["Attrition"]
                .apply(lambda x:(x=="Yes").mean()*100).sort_values())
        f2, ax2 = new_fig(7, 4.2)
        bp = [C["cyan"],C["violet"],C["rose"]]
        bars = ax2.barh(dept.index, dept.values, height=0.46, color=bp[:len(dept)], edgecolor="none")
        for bar,v in zip(bars,dept.values):
            ax2.text(v+0.3,bar.get_y()+bar.get_height()/2,
                     f"{v:.1f}%",va="center",fontsize=9,color=C["text2"])
        ax2.set_xlabel("Attrition Rate (%)"); ax2.set_xlim(0,dept.max()*1.3)
        ax2.set_title("Department Breakdown",pad=10); ax2.grid(axis="x",alpha=0.35)
        st.pyplot(f2,use_container_width=True); plt.close()

    c3, c4 = st.columns(2)
    with c3:
        sec_head("Monthly Income vs Attrition")
        f3, ax3 = new_fig(5.5, 3.8)
        try:
            from scipy.stats import gaussian_kde
            for lbl, col in [("No",C["cyan"]),("Yes",C["rose"])]:
                d = df_raw[df_raw["Attrition"]==lbl]["MonthlyIncome"]
                d.plot(kind="kde",ax=ax3,color=col,linewidth=2.2,label=lbl)
                xs = np.linspace(d.min(),d.max(),200)
                ax3.fill_between(xs,gaussian_kde(d)(xs),alpha=0.08,color=col)
        except ImportError:
            for lbl, col in [("No",C["cyan"]),("Yes",C["rose"])]:
                df_raw[df_raw["Attrition"]==lbl]["MonthlyIncome"].plot(
                    kind="kde",ax=ax3,color=col,linewidth=2.2,label=lbl)
        ax3.set_xlabel("Monthly Income ($)"); ax3.legend(title="Attrition",framealpha=0.3)
        ax3.set_title("Income Distribution",pad=10)
        st.pyplot(f3,use_container_width=True); plt.close()

    with c4:
        sec_head("Overtime vs Attrition")
        f4, ax4 = new_fig(5.5, 3.8)
        ot_pct = (df_raw.groupby(["OverTime","Attrition"]).size()
                  .unstack(fill_value=0).pipe(lambda d:d.div(d.sum(axis=1),axis=0)*100))
        x=np.arange(len(ot_pct)); w=0.38
        ax4.bar(x-w/2,ot_pct["No"], w,color=C["emerald"],alpha=0.85,label="Stay")
        ax4.bar(x+w/2,ot_pct["Yes"],w,color=C["rose"],   alpha=0.85,label="Leave")
        ax4.set_xticks(x); ax4.set_xticklabels(ot_pct.index,fontsize=9)
        ax4.set_ylabel("Percentage (%)"); ax4.legend(framealpha=0.3)
        ax4.set_title("Overtime Impact",pad=10)
        st.pyplot(f4,use_container_width=True); plt.close()

    st.markdown(DIVIDER, unsafe_allow_html=True)
    sec_head("Dataset Preview")
    st.dataframe(df_raw.head(8), use_container_width=True, height=255)


# ══════════════════════════════════════════════════════════════════
# PAGE ②  ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="hero-title">Deep Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Multivariate exploration of workforce attrition drivers</div>', unsafe_allow_html=True)

    sec_head("Age Distribution by Attrition Status")
    fa, axa = new_fig(11, 3.6)
    bins = np.linspace(df_raw["Age"].min(),df_raw["Age"].max(),28)
    for lbl,col,al in [("No",C["cyan"],0.5),("Yes",C["rose"],0.76)]:
        axa.hist(df_raw[df_raw["Attrition"]==lbl]["Age"],
                 bins=bins,color=col,alpha=al,label=f"Attrition: {lbl}",edgecolor="none")
    axa.set_xlabel("Age"); axa.set_ylabel("Count"); axa.legend(framealpha=0.3)
    axa.set_title("Age Distribution",pad=10)
    st.pyplot(fa,use_container_width=True); plt.close()

    c1, c2 = st.columns(2)
    with c1:
        sec_head("Job Role Attrition Rate")
        jr = (df_raw.groupby("JobRole")["Attrition"]
              .apply(lambda x:(x=="Yes").mean()*100).sort_values(ascending=True))
        fb, axb = new_fig(5.8, 5)
        bars = axb.barh(jr.index,jr.values,
                        color=plt.cm.cool(np.linspace(0.25,0.9,len(jr))),
                        edgecolor="none",height=0.58)
        for bar,v in zip(bars,jr.values):
            axb.text(v+0.15,bar.get_y()+bar.get_height()/2,
                     f"{v:.1f}%",va="center",fontsize=8,color=C["text2"])
        axb.set_xlabel("Attrition Rate (%)"); axb.set_xlim(0,jr.max()*1.3)
        axb.set_title("By Job Role",pad=10)
        st.pyplot(fb,use_container_width=True); plt.close()

    with c2:
        sec_head("Satisfaction Scores — Stay vs Leave")
        sat   = ["JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance"]
        means = df_raw.groupby("Attrition")[sat].mean()
        fc, axc = new_fig(5.8, 5)
        x=np.arange(len(sat)); w=0.38; short=["Job","Env","Rel.","WLB"]
        axc.bar(x-w/2,means.loc["No"], w,color=C["cyan"], alpha=0.85,label="Stay")
        axc.bar(x+w/2,means.loc["Yes"],w,color=C["rose"], alpha=0.85,label="Leave")
        axc.set_xticks(x); axc.set_xticklabels(short,rotation=10,ha="right")
        axc.set_ylabel("Mean Score (1–4)"); axc.legend(framealpha=0.3)
        axc.set_title("Satisfaction Comparison",pad=10)
        st.pyplot(fc,use_container_width=True); plt.close()

    sec_head("Pearson Correlation — Numeric Features")
    num = df_raw.select_dtypes(include=np.number)
    fd, axd = new_fig(14, 9)
    mask = np.triu(np.ones_like(num.corr(),dtype=bool))
    sns.heatmap(num.corr(),mask=mask,cmap=CMAP_DUAL,ax=axd,
                linewidths=0.25,linecolor="rgba(255,255,255,0.03)",
                annot=False,center=0,cbar_kws={"shrink":0.72})
    axd.tick_params(labelsize=7.5); axd.set_title("Feature Correlation Matrix",pad=12)
    st.pyplot(fd,use_container_width=True); plt.close()

    sec_head("Tenure vs Monthly Income")
    fe, axe = new_fig(11, 4)
    for lbl,col,m in [("No",C["cyan"],"o"),("Yes",C["rose"],"^")]:
        s = df_raw[df_raw["Attrition"]==lbl]
        axe.scatter(s["YearsAtCompany"],s["MonthlyIncome"],
                    alpha=0.3,s=18,c=col,marker=m,edgecolors="none",label=f"Attrition: {lbl}")
    axe.set_xlabel("Years at Company"); axe.set_ylabel("Monthly Income ($)")
    axe.set_title("Tenure vs Income",pad=10); axe.legend(framealpha=0.3)
    st.pyplot(fe,use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════
# PAGE ③  MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown('<div class="hero-title">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Performance evaluation, ROC analysis, and feature explainability</div>', unsafe_allow_html=True)

    pills_data = [
        ("✦ Accuracy",  f"{metrics['accuracy']*100:.2f}%",  "c"),
        ("✦ Precision", f"{metrics['precision']*100:.2f}%", "v"),
        ("✦ Recall",    f"{metrics['recall']*100:.2f}%",    "a"),
        ("✦ F1-Score",  f"{metrics['f1']*100:.2f}%",        "e"),
        ("✦ ROC-AUC",   f"{metrics['roc_auc']*100:.2f}%",   "r"),
        ("✦ CV Acc.",   f"{metrics.get('cv_mean',0)*100:.2f}%","c"),
    ]
    pills_html = "".join([f'<span class="mpill mpill-{cls}">{lbl} &nbsp;<b>{val}</b></span>'
                          for lbl,val,cls in pills_data])
    st.markdown(f'<div class="metric-row">{pills_html}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec_head("Confusion Matrix")
        f1, ax1 = new_fig(5.2, 4.5)
        sns.heatmap(metrics["cm"],annot=True,fmt="d",cmap=CMAP_VIOLET,ax=ax1,
                    xticklabels=["Stay","Leave"],yticklabels=["Stay","Leave"],
                    linewidths=3,linecolor="#080c14",
                    annot_kws={"fontsize":18,"fontweight":"bold","color":"white"})
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix — Test Set",pad=12)
        st.pyplot(f1,use_container_width=True); plt.close()

    with c2:
        sec_head("ROC Curve")
        f2, ax2 = new_fig(5.2, 4.5)
        fpr,tpr  = metrics["fpr"],metrics["tpr"]
        auc_val  = metrics["roc_auc"]
        ax2.plot(fpr,tpr,color=C["cyan"],lw=2.5,label=f"GBM  (AUC = {auc_val:.3f})")
        ax2.fill_between(fpr,tpr,alpha=0.07,color=C["cyan"])
        ax2.plot([0,1],[0,1],"--",color=C["text3"],lw=1.2,label="Random")
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.set_title("Receiver Operating Characteristic",pad=12)
        ax2.legend(loc="lower right",framealpha=0.3)
        st.pyplot(f2,use_container_width=True); plt.close()

    sec_head("Feature Importances — Top 20")
    feat  = pd.Series(metrics["importances"],index=metrics["feature_names"])
    top20 = feat.sort_values(ascending=True).tail(20)
    f3, ax3 = new_fig(11, 6)
    bars = ax3.barh(top20.index,top20.values,
                    color=plt.cm.plasma(np.linspace(0.2,0.95,len(top20))),
                    edgecolor="none",height=0.62)
    for bar,v in zip(bars,top20.values):
        ax3.text(v+0.0005,bar.get_y()+bar.get_height()/2,
                 f"{v:.3f}",va="center",fontsize=8,color=C["text2"])
    ax3.set_xlabel("Importance Score"); ax3.set_xlim(0,top20.max()*1.18)
    ax3.set_title("GBM Feature Importances",pad=12)
    st.pyplot(f3,use_container_width=True); plt.close()

    sec_head("Classification Report")
    row_map = {"0":"Stay","1":"Leave","macro avg":"Macro Avg","weighted avg":"Weighted Avg"}
    rpt_df  = (pd.DataFrame({row_map[k]:v for k,v in metrics["report"].items() if k in row_map})
               .T[["precision","recall","f1-score","support"]].round(3))
    rpt_df["support"] = rpt_df["support"].astype(int)
    st.dataframe(rpt_df.style.background_gradient(cmap="Blues",
                 subset=["precision","recall","f1-score"]),
                 use_container_width=True,height=195)

    cv_m = metrics.get("cv_mean",0); cv_s = metrics.get("cv_std",0)
    if cv_m:
        st.markdown(f"""
        <div class="glass glass-cyan" style="padding:1.4rem 1.8rem;margin-top:1rem;">
          <div class="sec-head" style="margin-bottom:.6rem;">5-Fold Cross-Validation</div>
          <span style='font-family:"Outfit",sans-serif;font-size:2.3rem;font-weight:900;
                       color:{C["cyan"]};letter-spacing:-.04em;
                       text-shadow:0 0 30px {C["cyan"]};'>{cv_m*100:.2f}%</span>
          <span style='font-size:.88rem;color:{C["text3"]};margin-left:.5rem;'>
            ± {cv_s*100:.2f}%</span>
          <div style='color:{C["text3"]};font-size:.77rem;margin-top:.4rem;'>
            Mean accuracy across 5 stratified folds on training data
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE ④  PREDICT
# ══════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown('<div class="hero-title">Attrition Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter employee details for an instant, model-driven risk score</div>', unsafe_allow_html=True)

    with st.form("pred_form"):
        # Personal
        st.markdown(f'<div class="fsec-title"><span class="fsec-icon">🪪</span> Personal Information</div>', unsafe_allow_html=True)
        p1,p2,p3,p4 = st.columns(4)
        age   = p1.slider("Age",18,65,35)
        gender= p2.selectbox("Gender",OPTIONS["Gender"])
        ms    = p3.selectbox("Marital Status",OPTIONS["MaritalStatus"])
        dist  = p4.slider("Distance from Home (km)",1,30,5)
        p5,p6 = st.columns(2)
        edu   = p5.selectbox("Education Level  (1=Below College … 5=Doctor)",[1,2,3,4,5],index=2)
        edu_f = p6.selectbox("Education Field",OPTIONS["EducationField"])

        st.markdown(DIVIDER, unsafe_allow_html=True)

        # Job
        st.markdown(f'<div class="fsec-title"><span class="fsec-icon">💼</span> Job Profile</div>', unsafe_allow_html=True)
        j1,j2,j3,j4 = st.columns(4)
        dept  = j1.selectbox("Department",OPTIONS["Department"])
        jrole = j2.selectbox("Job Role",OPTIONS["JobRole"])
        jlevel= j3.selectbox("Job Level",[1,2,3,4,5],index=1)
        jinv  = j4.selectbox("Job Involvement (1–4)",[1,2,3,4],index=2)
        j5,j6 = st.columns(2)
        btrv  = j5.selectbox("Business Travel",OPTIONS["BusinessTravel"])
        ot    = j6.selectbox("Over Time",OPTIONS["OverTime"])

        st.markdown(DIVIDER, unsafe_allow_html=True)

        # Compensation
        st.markdown(f'<div class="fsec-title"><span class="fsec-icon">💰</span> Compensation &amp; Tenure</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        mi  = c1.number_input("Monthly Income ($)",1009,20000,5000,100)
        dr  = c2.number_input("Daily Rate ($)",100,1500,800,50)
        hr_ = c3.number_input("Hourly Rate ($)",30,100,65,5)
        mr  = c4.number_input("Monthly Rate ($)",2094,27000,14000,200)
        c5,c6,c7,c8 = st.columns(4)
        twy = c5.slider("Total Working Years",0,40,10)
        yac = c6.slider("Years at Company",0,40,5)
        yir = c7.slider("Years in Role",0,18,3)
        ysp = c8.slider("Years Since Promo",0,15,2)
        c9,c10,c11 = st.columns(3)
        ywm = c9.slider("Years w/ Manager",0,17,3)
        ncw = c10.slider("Num Companies",0,9,2)
        psh = c11.slider("% Salary Hike",11,25,15)

        st.markdown(DIVIDER, unsafe_allow_html=True)

        # Satisfaction
        st.markdown(f'<div class="fsec-title"><span class="fsec-icon">😊</span> Satisfaction &amp; Wellness</div>', unsafe_allow_html=True)
        s1,s2,s3,s4,s5,s6 = st.columns(6)
        js  = s1.selectbox("Job Sat.", [1,2,3,4],index=2,key="js")
        es  = s2.selectbox("Env Sat.", [1,2,3,4],index=2,key="es")
        rs  = s3.selectbox("Rel. Sat.",[1,2,3,4],index=2,key="rs")
        wlb = s4.selectbox("Work-Life",[1,2,3,4],index=2,key="wlb")
        prat= s5.selectbox("Perf. Rtg",[1,2,3,4],index=2,key="pr")
        tr  = s6.slider("Training/yr",0,6,3)
        _a,_b = st.columns(2)
        so  = _a.selectbox("Stock Option (0–3)",[0,1,2,3],index=1)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  Analyse Attrition Risk")

    if submitted:
        inp = {
            "Age":age,"BusinessTravel":btrv,"DailyRate":dr,"Department":dept,
            "DistanceFromHome":dist,"Education":edu,"EducationField":edu_f,
            "EnvironmentSatisfaction":es,"Gender":gender,"HourlyRate":hr_,
            "JobInvolvement":jinv,"JobLevel":jlevel,"JobRole":jrole,"JobSatisfaction":js,
            "MaritalStatus":ms,"MonthlyIncome":mi,"MonthlyRate":mr,
            "NumCompaniesWorked":ncw,"OverTime":ot,"PercentSalaryHike":psh,
            "PerformanceRating":prat,"RelationshipSatisfaction":rs,"StockOptionLevel":so,
            "TotalWorkingYears":twy,"TrainingTimesLastYear":tr,"WorkLifeBalance":wlb,
            "YearsAtCompany":yac,"YearsInCurrentRole":yir,"YearsSinceLastPromotion":ysp,
            "YearsWithCurrManager":ywm,
        }
        idf = pd.DataFrame([inp])
        for col in CATEGORICAL_COLS:
            idf[col] = label_encoders[col].transform(idf[col])
        idf  = idf[feature_names]
        pred = model.predict(idf)[0]
        prob = model.predict_proba(idf)[0][1]

        st.markdown(DIVIDER, unsafe_allow_html=True)
        sec_head("Risk Assessment Result")

        r1, r2, r3 = st.columns([1.3,1,1])
        with r1:
            if pred==1:
                st.markdown(f"""
                <div class="pred-result pred-high">
                  <span class="pred-icon" style="color:{C['rose']};">⚠️</span>
                  <div class="pred-title">HIGH ATTRITION RISK</div>
                  <div class="pred-body">This profile indicates elevated flight risk.<br>
                  Immediate retention action is recommended.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-result pred-low">
                  <span class="pred-icon" style="color:{C['emerald']};">✅</span>
                  <div class="pred-title">LOW ATTRITION RISK</div>
                  <div class="pred-body">This employee profile appears stable.<br>
                  Continue standard engagement practices.</div>
                </div>""", unsafe_allow_html=True)

        with r2:
            st.metric("Attrition Probability", f"{prob*100:.1f}%")
            bc = C["rose"] if prob>0.5 else C["emerald"]
            st.markdown(f"""
            <div style="margin-top:.8rem;">
              <div class="risk-bar-label">
                <span>Low</span><span>Risk Score</span><span>High</span>
              </div>
              <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width:{prob*100:.1f}%;
                  background:linear-gradient(90deg,{C['emerald']},{bc});"></div>
              </div>
              <div style="text-align:right;font-family:'Fira Code',monospace;
                          font-size:.75rem;color:{bc};margin-top:.4rem;
                          text-shadow:0 0 10px {bc};">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            fg, axg = plt.subplots(figsize=(3.6,3.2),subplot_kw=dict(aspect="equal"))
            fg.patch.set_facecolor("#0d1220"); axg.set_facecolor("#0d1220")
            t = np.linspace(np.pi,0,300)
            axg.plot(np.cos(t),np.sin(t),color="rgba(255,255,255,0.055)",lw=14,solid_capstyle="round")
            fe = np.pi-prob*np.pi
            tf = np.linspace(np.pi,fe,300)
            gc = C["rose"] if prob>0.5 else C["emerald"]
            axg.plot(np.cos(tf),np.sin(tf),color=gc,lw=14,solid_capstyle="round")
            axg.annotate("",xy=(.58*np.cos(fe),.58*np.sin(fe)),xytext=(0,0),
                          arrowprops=dict(arrowstyle="-|>",color=C["text"],lw=2,mutation_scale=14))
            axg.text(0,-.28,f"{prob*100:.0f}%",ha="center",va="center",
                     fontsize=20,fontweight="bold",color=gc)
            axg.text(0,-.54,"Risk Score",ha="center",va="center",fontsize=8,color=C["text3"])
            axg.set_xlim(-1.2,1.2); axg.set_ylim(-.7,1.2); axg.axis("off")
            st.pyplot(fg,use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════
# PAGE ⑤  BATCH SCAN
# ══════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown('<div class="hero-title">Batch Attrition Scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a workforce CSV to score every employee in seconds</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass glass-violet" style="padding:1.4rem 1.8rem;margin-bottom:1.5rem;">
      <div class="sec-head" style="margin-bottom:.6rem;">📋 Instructions</div>
      <div style='color:{C["text2"]};font-size:.87rem;line-height:1.85;'>
        Upload a CSV with the same schema as the training dataset (excluding
        <code style='color:{C["cyan"]};background:rgba(0,229,255,0.08);
        padding:.1rem .35rem;border-radius:4px;'>Attrition</code>). The model appends
        <code style='color:{C["violet"]};'>Predicted_Attrition</code> and
        <code style='color:{C["violet"]};'>Attrition_Probability_%</code> to every row.
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV here", type=["csv"])

    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            st.success(f"✓  Loaded **{len(batch):,}** employee records")
            bp = batch.copy()
            for col in DROP_COLS+["Attrition"]:
                if col in bp.columns: bp.drop(columns=[col],inplace=True)
            for col in CATEGORICAL_COLS:
                if col in bp.columns: bp[col] = label_encoders[col].transform(bp[col])
            bp    = bp[feature_names]
            preds = model.predict(bp)
            probs = model.predict_proba(bp)[:,1]
            res   = batch.copy()
            res["Predicted_Attrition"]     = ["Yes" if p==1 else "No" for p in preds]
            res["Attrition_Probability_%"] = (probs*100).round(2)
            yes_n = (preds==1).sum(); no_n = len(preds)-yes_n

            st.markdown(f"""
            <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-top:1.2rem;">
              <div class="stat stat-c">
                <span class="stat-icon" style="color:{C['cyan']};">🔍</span>
                <div class="stat-value">{len(res):,}</div>
                <div class="stat-label">Employees Scanned</div>
              </div>
              <div class="stat stat-r">
                <span class="stat-icon" style="color:{C['rose']};">⚠️</span>
                <div class="stat-value">{yes_n}</div>
                <div class="stat-label">At Risk</div>
                <div class="stat-sub">{yes_n/len(res)*100:.1f}%</div>
              </div>
              <div class="stat stat-e">
                <span class="stat-icon" style="color:{C['emerald']};">✅</span>
                <div class="stat-value">{no_n}</div>
                <div class="stat-label">Predicted to Stay</div>
                <div class="stat-sub">{no_n/len(res)*100:.1f}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            sec_head("Risk Score Distribution")
            fh, axh = new_fig(11,3.5)
            _,bins_h,patches_h = axh.hist(probs*100,bins=30,edgecolor="none")
            for patch,left in zip(patches_h,bins_h[:-1]):
                patch.set_facecolor(C["rose"] if left>=50 else C["emerald"]); patch.set_alpha(0.8)
            axh.axvline(50,color=C["amber"],ls="--",lw=1.8,label="50% threshold")
            axh.set_xlabel("Attrition Probability (%)"); axh.set_ylabel("Count")
            axh.set_title("Distribution of Predicted Risk Scores",pad=10); axh.legend(framealpha=0.3)
            st.pyplot(fh,use_container_width=True); plt.close()

            sec_head("Results Preview — first 15 rows")
            show = ["Predicted_Attrition","Attrition_Probability_%"]+[
                c for c in res.columns if c not in ["Predicted_Attrition","Attrition_Probability_%"]]
            st.dataframe(res[show].head(15),use_container_width=True,height=340)
            st.download_button("⬇️  Download Full Results CSV",
                               res.to_csv(index=False).encode("utf-8"),
                               "attrition_scan.csv","text/csv")
        except Exception as e:
            st.error(f"Processing error: {e}")
    else:
        st.markdown(f"""
        <div style='border:1.5px dashed rgba(255,255,255,0.1);border-radius:20px;
                    padding:4rem 2rem;text-align:center;color:{C["text3"]};
                    margin-top:1rem;background:rgba(255,255,255,0.015);backdrop-filter:blur(10px);'>
          <div style='font-size:3.5rem;margin-bottom:1rem;animation:float 3s ease-in-out infinite;'>📂</div>
          <div style='font-size:1rem;font-weight:600;color:{C["text2"]};'>
            Drag &amp; drop your workforce CSV above</div>
          <div style='font-size:.8rem;margin-top:.5rem;'>Supports files with up to 50,000 rows</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div style='margin-top:4rem;padding:1.4rem 0;
            border-top:1px solid {C["border"]};
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:.5rem;'>
  <div style='display:flex;align-items:center;gap:1rem;'>
    <span style='font-family:"Outfit",sans-serif;font-weight:900;font-size:1rem;
                 background:linear-gradient(135deg,#fff,{C["cyan"]});
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;'>AttritionIQ</span>
    <span style='color:{C["text3"]};font-size:.72rem;'>
      Gradient Boosting · IBM HR Analytics · 30 Features</span>
  </div>
  <div style='font-family:"Fira Code",monospace;font-size:.7rem;color:{C["text3"]};'>
    accuracy: {metrics["accuracy"]*100:.2f}% &nbsp;·&nbsp; roc-auc: {metrics["roc_auc"]*100:.2f}%
  </div>
</div>
""", unsafe_allow_html=True)
