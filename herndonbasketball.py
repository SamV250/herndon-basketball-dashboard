import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------
# App Config
# ----------------------------------
st.set_page_config(
    page_title="Herndon Basketball Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom theme
st.markdown(
    """
    <style>
    :root {
        --herndon-red: #B22222;
        --herndon-black: #000000;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stMetric {
        color: var(--herndon-red) !important;
    }
    .stButton>button {
        background-color: var(--herndon-red);
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #8B0000;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--herndon-red);
        color: white;
        border-radius: 8px 8px 0 0;
        margin-right: 4px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #8B0000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------
# Smart helpers
# ----------------------------------

@st.cache_data(show_spinner=False)
def load_any_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """
    Try CSV first, then Excel. Works for any schema.
    """
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception:
        # fallback
        uploaded_file.seek(0)
        try:
            return pd.read_excel(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Look at a dataframe and figure out:
    - numeric columns
    - categorical/string columns
    - possible ID/name columns
    - shot-style data
    - basketball-ish data (four factors / box score) if present
    """
    if df is None or df.empty:
        return {
            "numeric": [],
            "categorical": [],
            "text": [],
            "id_like": [],
            "looks_like_shots": False,
            "looks_like_basketball_team": False,
            "looks_like_basketball_player": False,
        }

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == object]

    # text-ish columns (long strings)
    text_cols = []
    for c in cat_cols:
        avg_len = df[c].dropna().astype(str).str.len().mean() if not df[c].dropna().empty else 0
        if avg_len and avg_len > 30:
            text_cols.append(c)

    # id-like columns: object but not too many uniques
    id_like = []
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        if 1 < nunique < len(df) * 0.8:
            id_like.append(c)

    # detect shots
    lower_cols = [c.lower() for c in df.columns]
    looks_like_shots = all(x in lower_cols for x in ["x", "y"]) and any(
        r in lower_cols for r in ["result", "made", "miss"]
    )

    # detect basketball-ish
    ff_candidates = {"EFG_O", "EFG_D", "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD"}
    looks_like_basketball_team = len(ff_candidates.intersection(set(df.columns))) >= 2

    player_candidates = {"PTS", "AST", "TRB", "REB", "MP", "FGA", "FG", "3PA"}
    looks_like_basketball_player = len(player_candidates.intersection(set(df.columns))) >= 2

    return {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "text": text_cols,
        "id_like": id_like,
        "looks_like_shots": looks_like_shots,
        "looks_like_basketball_team": looks_like_basketball_team,
        "looks_like_basketball_player": looks_like_basketball_player,
    }


def render_categorical_cards(df: pd.DataFrame, schema: dict, title: str = "Roster / Info"):
    st.subheader(title)
    cat_cols = schema["categorical"]
    id_like = schema["id_like"]

    if not cat_cols:
        st.info("No categorical columns detected.")
        return

    # choose a main label column
    if id_like:
        label_col = st.selectbox("Label column", id_like, index=0)
    else:
        label_col = st.selectbox("Label column", cat_cols, index=0)

    other_cols = [c for c in cat_cols if c != label_col]

    for _, row in df.iterrows():
        with st.expander(str(row.get(label_col, "Item")), expanded=False):
            for c in other_cols:
                val = row.get(c, "")
                if pd.isna(val) or val == "":
                    continue
                st.markdown(f"**{c}:** {val}")


def render_numeric_analysis(df: pd.DataFrame, schema: dict, title: str = "Stats Analysis"):
    st.subheader(title)
    num_cols = schema["numeric"]
    if len(num_cols) < 1:
        st.info("No numeric columns found — nothing to analyze.")
        return

    st.write("**Summary stats**")
    st.dataframe(df[num_cols].describe().T, use_container_width=True)

    if len(num_cols) >= 2:
        st.write("**Correlation heatmap**")
        corr = df[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

    # optional clustering if enough numeric vars
    if len(num_cols) >= 3:
        st.write("**Quick clustering (KMeans)**")
        k = st.slider("Number of clusters", 2, min(8, len(df)), 3)
        X = df[num_cols].fillna(0.0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=k, n_init=20, random_state=7)
        labels = km.fit_predict(Xs)
        df_clustered = df.copy()
        df_clustered["Cluster"] = labels

        x_axis = st.selectbox("X axis", num_cols, index=0)
        y_axis = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1))
        fig2 = px.scatter(df_clustered, x=x_axis, y=y_axis, color="Cluster", hover_data=df.columns)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df_clustered, use_container_width=True)

        csv = df_clustered.to_csv(index=False).encode()
        st.download_button("Download clustered data (CSV)", csv, file_name="clustered_data.csv")


def render_shot_chart(df: pd.DataFrame):
    st.subheader("Shot Chart (auto-detected)")
    # try flexible column detection
    cols_lower = {c.lower(): c for c in df.columns}
    x_col = cols_lower.get("x", None)
    y_col = cols_lower.get("y", None)
    result_col = cols_lower.get("result", None) or cols_lower.get("made", None) or cols_lower.get("miss", None)

    if x_col is None or y_col is None:
        st.info("Shot-like data detected but missing x/y columns.")
        return

    court_w, court_h = 50, 47
    fig = go.Figure()
    # outer
    fig.add_shape(type="rect", x0=0, y0=0, x1=court_w, y1=court_h, line=dict(width=2))
    # rim
    theta = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(
        go.Scatter(x=25 + 0.75*np.cos(theta), y=5.25 + 0.75*np.sin(theta), mode="lines", showlegend=False)
    )
    # restricted
    th = np.linspace(0, np.pi, 200)
    fig.add_trace(go.Scatter(x=25 + 4*np.cos(th), y=5.25 + 4*np.sin(th), mode="lines", showlegend=False))

    if result_col:
        made = df[df[result_col].astype(str).str.lower().isin(["make", "made", "1", "true"])]
        missed = df[~df.index.isin(made.index)]
        if not made.empty:
            fig.add_trace(go.Scatter(
                x=made[x_col], y=made[y_col], mode="markers", name="Make",
                marker=dict(symbol="circle", size=7, line=dict(width=1))
            ))
        if not missed.empty:
            fig.add_trace(go.Scatter(
                x=missed[x_col], y=missed[y_col], mode="markers", name="Miss",
                marker=dict(symbol="x", size=8)
            ))
    else:
        # just plot locations
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="markers", name="Shots"))

    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, court_h])
    fig.update_xaxes(range=[0, court_w])
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Shot density heatmap")
    fig_hm = px.density_heatmap(df, x=x_col, y=y_col, nbinsx=25, nbinsy=25)
    fig_hm.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_hm, use_container_width=True)


def render_basketballish_team(df: pd.DataFrame):
    """
    Keep your original team four-factor look, but only for columns that exist.
    """
    st.subheader("Team Four Factors (auto)")
    candidates = ["EFG_O","EFG_D","TOR","TORD","ORB","DRB","FTR","FTRD"]
    available = [c for c in candidates if c in df.columns]
    if not available:
        st.info("Detected basketball-style data, but four-factor columns are missing.")
        return

    agg = df[available].mean(numeric_only=True)
    col1, col2 = st.columns(2)
    with col1:
        for c in ["EFG_O","TOR","ORB","FTR"]:
            if c in agg:
                st.metric(c, f"{agg[c]*100:.1f}%" if agg[c] < 1.5 else f"{agg[c]:.2f}")
    with col2:
        for c in ["EFG_D","TORD","DRB","FTRD"]:
            if c in agg:
                st.metric(c, f"{agg[c]*100:.1f}%" if agg[c] < 1.5 else f"{agg[c]:.2f}")

    r = agg.to_frame(name="Team").reset_index(names="Factor")
    fig = px.line_polar(r, r="Team", theta="Factor", line_close=True)
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)


def build_quick_report(df: pd.DataFrame, schema: dict) -> str:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_cols": schema["numeric"],
        "categorical_cols": schema["categorical"],
    }
    if schema["numeric"]:
        report["numeric_summary"] = df[schema["numeric"]].describe().round(3).to_dict()
    return json.dumps(report, indent=2)


# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.image("herndonbasketball.jpg", use_container_width=True)
st.sidebar.title("🏀 Herndon Basketball")
st.sidebar.caption("Upload ANY CSV/XLSX — the app will figure it out.")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# ----------------------------------
# Load data
# ----------------------------------
if uploaded is not None:
    df = load_any_file(uploaded)
else:
    # tiny demo default
    df = pd.DataFrame({
        "Player": ["Ryan", "Charlie", "David"],
        "Strengths": ["Shooter", "Handles, IQ", "Rim protection"],
        "Weaknesses": ["Needs to talk", "Left hand", "Perimeter D"],
        "Notes": ["Great in transition", "Leader", "Keep confident"]
    })

schema = detect_schema(df)

# ----------------------------------
# Header
# ----------------------------------
col_logo, col_title = st.columns([1,6])
with col_logo:
    st.markdown("## 🏀")
with col_title:
    st.title("Herndon Basketball Analytics Dashboard")
    st.caption("Schema-aware: categorical → cards, numeric → analysis, shots → chart")

# ----------------------------------
# Tabs
# ----------------------------------
T1, T2, T3, T4 = st.tabs([
    "Data Preview",
    "Categorical / Roster View",
    "Numeric / Analytics",
    "Report",
])

with T1:
    st.header("Data Preview")
    st.dataframe(df, use_container_width=True)
    st.write(f"**Detected numeric columns:** {schema['numeric']}")
    st.write(f"**Detected categorical columns:** {schema['categorical']}")
    if schema["looks_like_shots"]:
        st.success("Shot-like data detected — check Numeric/Analytics tab or upload specific shots.")

with T2:
    if schema["categorical"]:
        render_categorical_cards(df, schema, title="Roster / Scouting Notes")
    else:
        st.info("No categorical data to display.")

with T3:
    # priority: if shots -> show chart first
    if schema["looks_like_shots"]:
        render_shot_chart(df)
        st.markdown("---")
    # if basketball-ish -> show team panel
    if schema["looks_like_basketball_team"]:
        render_basketballish_team(df)
        st.markdown("---")

    # generic numeric analysis
    render_numeric_analysis(df, schema, title="Auto Stats Analysis")

with T4:
    st.header("Quick JSON Report")
    report = build_quick_report(df, schema)
    st.code(report, language="json")
    st.download_button("Download report.json", report, file_name="herndon_auto_report.json")

st.markdown("---")
st.caption("Tip: save as app.py and run with `streamlit run app.py`. This version is NOT hardcoded to specific column names.")
