"""
=============================================================================
NETWORK ATTACK DETECTION SYSTEM — STARLINK TELEMETRY
Master's Thesis: "Development of Network Attack Detection Systems Using ML"
Kazakh-British Technical University (KBTU)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NADS | Network Attack Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — STRICT ACADEMIC DARK THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* BASE */
.stApp { background-color: #080C10; color: #CDD9E5; font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { color: #E6EDF3 !important; font-weight: 600 !important; letter-spacing: 0.03em; }
p, li { color: #CDD9E5 !important; }

/* HEADER BAND */
.header-band {
    background: linear-gradient(90deg, #080C10 0%, #0D1117 50%, #080C10 100%);
    border-bottom: 1px solid #21262D;
    padding: 12px 0 8px 0;
    margin-bottom: 24px;
}
.system-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #58A6FF;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.system-subtitle { font-size: 0.78rem; color: #8B949E; letter-spacing: 0.05em; }

/* METRICS */
[data-testid="stMetric"] {
    background-color: #0D1117 !important;
    border: 1px solid #21262D !important;
    border-radius: 6px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #E6EDF3 !important; font-family: 'JetBrains Mono', monospace !important; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { background-color: #080C10 !important; border-bottom: 1px solid #21262D !important; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8B949E !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.1em; padding: 8px 20px !important; border-radius: 4px 4px 0 0 !important; }
.stTabs [aria-selected="true"] { color: #58A6FF !important; border-bottom: 2px solid #58A6FF !important; background-color: rgba(88,166,255,0.05) !important; }

/* SIDEBAR */
[data-testid="stSidebar"] { background-color: #0D1117 !important; border-right: 1px solid #21262D; }
[data-testid="stSidebar"] * { color: #CDD9E5 !important; }

/* STATUS BADGES */
.badge-stable   { background:#0D2E1C; color:#3FB950; border:1px solid #238636; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-attack   { background:#2D0F0F; color:#F85149; border:1px solid #8B2A2A; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-jamming  { background:#2B1D0A; color:#E3B341; border:1px solid #9E6A03; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-congestion { background:#0F1E2D; color:#58A6FF; border:1px solid #1F6FEB; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; }

/* ALERT PANEL */
.alert-critical {
    border-left: 3px solid #F85149;
    background: rgba(248,81,73,0.07);
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #F85149;
}
.alert-info {
    border-left: 3px solid #58A6FF;
    background: rgba(88,166,255,0.06);
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    font-size: 0.82rem;
    color: #58A6FF;
}
.xai-feature {
    display: flex; justify-content: space-between; align-items: center;
    background: #0D1117; border: 1px solid #21262D;
    padding: 10px 14px; border-radius: 6px; margin: 5px 0;
}
.xai-bar { height: 6px; border-radius: 3px; margin-top: 4px; }
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #8B949E;
    text-transform: uppercase; letter-spacing: 0.15em;
    border-bottom: 1px solid #21262D; padding-bottom: 6px; margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
COL_TS          = "Timestamp"
COL_DL          = "health.ut.downlink_throughput"
COL_UL          = "health.ut.uplink_throughput"
COL_PING        = "health.ut.pop_ping_latency_ms_avg"
COL_DROP        = "health.ut.pop_ping_drop_rate_avg"
COL_SNR         = "health.ut.rx_avg_snr"
COL_OBSTR       = "user1.obstruction_map.stats.percent_time"

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,1)",
    font=dict(family="Inter, sans-serif", color="#CDD9E5", size=11),
    xaxis=dict(gridcolor="#21262D", linecolor="#30363D", zeroline=False),
    yaxis=dict(gridcolor="#21262D", linecolor="#30363D", zeroline=False),
    legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#21262D", borderwidth=1)
)

STATUS_COLORS = {
    "STABLE":             "#3FB950",
    "CYBER ATTACK":       "#F85149",
    "SIGNAL JAMMING":     "#E3B341",
    "NETWORK CONGESTION": "#58A6FF",
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_telemetry(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=[COL_TS])
    df[COL_TS] = pd.to_datetime(df[COL_TS], utc=True).dt.tz_localize(None)
    df = df.sort_values(COL_TS).reset_index(drop=True)
    return df

@st.cache_data
def load_security_logs(file) -> pd.DataFrame:
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()
        # Normalize datetime column
        dt_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if dt_col:
            df["_ts"] = pd.to_datetime(df[dt_col[0]], errors="coerce", utc=False)
        else:
            df["_ts"] = pd.NaT
        return df
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Z-scores (statistical anomaly indicators) ──────────────
    df["z_ping"]  = np.abs(stats.zscore(df[COL_PING].fillna(df[COL_PING].median())))
    df["z_drop"]  = np.abs(stats.zscore(df[COL_DROP].fillna(0)))
    df["z_snr"]   = np.abs(stats.zscore(df[COL_SNR].fillna(df[COL_SNR].median())))

    # ── Rolling baselines (1-hour window) ──────────────────────
    df["dl_rolling_mean"] = df[COL_DL].rolling(6, min_periods=1).mean()
    df["dl_deviation"]    = (df[COL_DL] - df["dl_rolling_mean"]) / (df["dl_rolling_mean"].replace(0, np.nan))
    df["dl_deviation"]    = df["dl_deviation"].fillna(0)

    # ── SNR-Throughput divergence (Correlation Score) ──────────
    # High SNR + low downlink = suspicious (attack masks itself behind clean signal)
    snr_norm  = (df[COL_SNR] - df[COL_SNR].min()) / (df[COL_SNR].max() - df[COL_SNR].min() + 1e-9)
    dl_norm   = (df[COL_DL]  - df[COL_DL].min())  / (df[COL_DL].max()  - df[COL_DL].min()  + 1e-9)
    df["correlation_score"] = snr_norm * (1 - dl_norm)  # 0→benign, 1→highly suspicious

    # ── Obstruction indicator ────────────────────────────────────
    df["obstr_flag"] = (df[COL_OBSTR] > 0).astype(int)

    return df

# ─────────────────────────────────────────────────────────────
# DATA FUSION — telemetry ✕ security logs
# ─────────────────────────────────────────────────────────────
def fuse_with_logs(df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nms_illegal_rip"] = False
    df["nms_severity"]    = "NONE"

    if logs.empty or "_ts" not in logs.columns:
        return df

    # Detect rows with illegal RIP events
    desc_col = [c for c in logs.columns if "description" in c.lower() or "event" in c.lower()]
    if not desc_col:
        return df
    desc_col = desc_col[0]

    rip_logs = logs[logs[desc_col].str.contains("Illegal unicast RIP|Illegal RIP", case=False, na=False)].copy()
    if rip_logs.empty:
        return df

    rip_logs = rip_logs.dropna(subset=["_ts"])

    # Merge-asof: match each telemetry row to nearest log within ±30 min
    df_sorted  = df.sort_values(COL_TS)
    rip_sorted = rip_logs.sort_values("_ts")

    merged = pd.merge_asof(
        df_sorted,
        rip_sorted[["_ts", desc_col]].rename(columns={"_ts": "_log_ts", desc_col: "_log_event"}),
        left_on=COL_TS, right_on="_log_ts",
        tolerance=pd.Timedelta("30min"),
        direction="nearest"
    )
    mask = merged["_log_event"].notna()
    df_sorted["nms_illegal_rip"] = mask.values
    df_sorted["nms_severity"]    = merged["_log_event"].where(mask, "NONE").values

    return df_sorted.sort_index()

# ─────────────────────────────────────────────────────────────
# ML DETECTION
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [COL_PING, COL_DROP, COL_SNR, COL_DL,
                "z_ping", "z_drop", "correlation_score", "dl_deviation"]
FEATURE_LABELS = {
    COL_PING:             "Ping Latency (ms)",
    COL_DROP:             "Packet Drop Rate",
    COL_SNR:              "Signal SNR",
    COL_DL:               "Downlink Throughput",
    "z_ping":             "Z-score: Latency",
    "z_drop":             "Z-score: Packet Loss",
    "correlation_score":  "SNR-Throughput Divergence",
    "dl_deviation":       "Throughput Deviation",
}

def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURE_COLS].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.06, n_estimators=200, random_state=42)
    df["if_score"]  = model.fit_predict(X_scaled)           # -1 = anomaly
    df["if_raw"]    = model.decision_function(X_scaled)     # lower = more anomalous
    return df, model, scaler

def classify_incidents(row) -> str:
    """
    Rule-based classifier on top of IsolationForest anomaly flag.
    Priority: CYBER ATTACK > SIGNAL JAMMING > NETWORK CONGESTION > STABLE
    """
    is_anomaly = row["if_score"] == -1

    if not is_anomaly:
        return "STABLE"

    # Cyber Attack: high packet loss + NMS illegal-RIP event
    if row[COL_DROP] > 0.05 and row.get("nms_illegal_rip", False):
        return "CYBER ATTACK"

    # Cyber Attack fallback: high correlation score (SNR fine, throughput tanked)
    if row["correlation_score"] > 0.65 and row["z_drop"] > 2.5:
        return "CYBER ATTACK"

    # Signal Jamming: SNR collapse
    if row[COL_SNR] < 0.5 or row["z_snr"] > 2.5:
        return "SIGNAL JAMMING"

    # Network Congestion: high ping, clean logs
    if row["z_ping"] > 2.0:
        return "NETWORK CONGESTION"

    return "NETWORK CONGESTION"

def full_analysis(df: pd.DataFrame, logs: pd.DataFrame) -> tuple:
    df = engineer_features(df)
    df = fuse_with_logs(df, logs)
    df, model, scaler = run_isolation_forest(df)
    df["status"] = df.apply(classify_incidents, axis=1)
    return df, model, scaler

# ─────────────────────────────────────────────────────────────
# ATTACK SIMULATION
# ─────────────────────────────────────────────────────────────
def simulate_advanced_attack(df: pd.DataFrame, logs: pd.DataFrame) -> tuple:
    """
    Inject synthetic DoS attack signature into the last 12 rows:
    - Packet drop rate spikes to 15-20%
    - Latency increases 2-3×
    - Throughput collapses
    - Synthetic 'Illegal unicast RIP message' log entries injected
    """
    df_sim = df.copy()
    n = len(df_sim)
    attack_idx = range(max(0, n - 12), n)

    rng = np.random.default_rng(seed=42)
    df_sim.loc[df_sim.index[attack_idx], COL_DROP] = rng.uniform(0.12, 0.20, len(attack_idx))
    df_sim.loc[df_sim.index[attack_idx], COL_PING] = df_sim[COL_PING].median() * rng.uniform(2.1, 3.0, len(attack_idx))
    df_sim.loc[df_sim.index[attack_idx], COL_DL]   = df_sim[COL_DL].median()   * rng.uniform(0.05, 0.15, len(attack_idx))

    # Synthesize RIP log rows
    attack_times = df_sim[COL_TS].iloc[list(attack_idx)].values
    syn_logs = pd.DataFrame({
        "_ts": attack_times,
        "Event Description": ["Illegal unicast RIP message from 10.0.0.1" for _ in attack_times],
        "Severity": ["CRITICAL" for _ in attack_times],
        "Element Name": ["SIM-INJECT" for _ in attack_times],
    })
    if logs.empty:
        merged_logs = syn_logs.rename(columns={"Event Description": "Event Description"})
    else:
        merged_logs = pd.concat([logs, syn_logs], ignore_index=True)

    return df_sim, merged_logs

# ─────────────────────────────────────────────────────────────
# XAI — Explainable AI feature attribution
# ─────────────────────────────────────────────────────────────
def explain_incident(row: pd.Series, scaler: StandardScaler) -> dict:
    """
    Approximate feature importance via scaled deviation from normal mean.
    Returns dict {feature_label: contribution_score (0-1)}
    """
    vals = row[FEATURE_COLS].fillna(0).values.reshape(1, -1)
    scaled = scaler.transform(vals)[0]
    # Contribution = absolute deviation from zero (normalized center)
    raw_contrib = np.abs(scaled)
    total = raw_contrib.sum() + 1e-9
    normed = raw_contrib / total
    return {FEATURE_LABELS[f]: float(normed[i]) for i, f in enumerate(FEATURE_COLS)}

# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────
def ts_plot_with_anomalies(df: pd.DataFrame, y_col: str, y_label: str, title: str) -> go.Figure:
    fig = go.Figure()

    # Normal data
    normal = df[df["status"] == "STABLE"]
    fig.add_trace(go.Scatter(
        x=normal[COL_TS], y=normal[y_col],
        mode="lines", name="STABLE",
        line=dict(color="#3FB950", width=1.2)
    ))

    # Anomaly points
    for status, color in STATUS_COLORS.items():
        if status == "STABLE":
            continue
        sub = df[df["status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[COL_TS], y=sub[y_col],
            mode="markers", name=status,
            marker=dict(color=color, size=7, symbol="circle-open", line=dict(width=2))
        ))

    # Shade attack windows
    attack_mask = df["status"] == "CYBER ATTACK"
    if attack_mask.any():
        a_times = df[attack_mask][COL_TS]
        for t in a_times:
            fig.add_vrect(
                x0=t - pd.Timedelta("20min"),
                x1=t + pd.Timedelta("20min"),
                fillcolor="rgba(248,81,73,0.08)",
                layer="below", line_width=0
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#E6EDF3")),
        yaxis_title=y_label,
        height=280,
        margin=dict(l=40, r=20, t=40, b=30),
        **PLOTLY_TEMPLATE
    )
    return fig

def status_distribution_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    fig = px.pie(
        counts, names="status", values="count",
        color="status",
        color_discrete_map=STATUS_COLORS,
        hole=0.55,
    )
    fig.update_traces(textinfo="label+percent", textfont_color="#E6EDF3")
    fig.update_layout(
        showlegend=False, height=240,
        margin=dict(l=10, r=10, t=20, b=10),
        **PLOTLY_TEMPLATE
    )
    return fig

def xai_bar_chart(contributions: dict, incident_status: str) -> go.Figure:
    feats = list(contributions.keys())
    vals  = list(contributions.values())
    color = STATUS_COLORS.get(incident_status, "#58A6FF")

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(
            color=[color] * len(vals),
            opacity=[0.4 + 0.6 * v for v in vals]
        ),
        text=[f"{v*100:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#CDD9E5", size=10)
    ))
    fig.update_layout(
        title=dict(text="Feature Attribution (XAI)", font=dict(size=12, color="#E6EDF3")),
        xaxis=dict(title="Relative Contribution", tickformat=".0%"),
        height=320,
        margin=dict(l=160, r=60, t=40, b=30),
        **PLOTLY_TEMPLATE
    )
    return fig

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
  <span class="system-title">⬡ NADS &nbsp;|&nbsp; Network Attack Detection System</span><br>
  <span class="system-subtitle">KBTU Master's Thesis &nbsp;·&nbsp; Starlink Telemetry Analytics &nbsp;·&nbsp; ML-Powered Threat Classification</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)
    csv_files  = st.file_uploader("Telemetry CSV (30-day export)", type="csv",  accept_multiple_files=True)
    xlsx_file  = st.file_uploader("NMS Security Log (Export .xlsx)", type=["xlsx", "xls"])

    st.markdown('<div class="section-label" style="margin-top:20px">Simulation Controls</div>', unsafe_allow_html=True)
    run_sim = st.button("⚡ INJECT SYNTHETIC ATTACK", use_container_width=True,
                        help="Simulates DoS: spikes packet drop to 15-20%, injects Illegal RIP log entries")
    if run_sim:
        st.session_state["simulation_active"] = True
    if st.button("↺ RESET SIMULATION", use_container_width=True):
        st.session_state["simulation_active"] = False

    st.markdown('<div class="section-label" style="margin-top:20px">Detection Parameters</div>', unsafe_allow_html=True)
    contamination = st.slider("IsolationForest Contamination", 0.02, 0.15, 0.06, 0.01,
                              help="Expected fraction of anomalies in data")
    z_threshold   = st.slider("Z-score Alert Threshold", 1.5, 4.0, 2.5, 0.1)

    st.markdown("---")
    st.markdown('<span style="font-size:0.7rem;color:#8B949E">Supervisor: Zuhra Abdiakhmetova<br>KBTU · 2025</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────
if not csv_files:
    st.markdown("""
    <div class="alert-info">
    ⬡ &nbsp; Upload one or two 30-day telemetry CSV files and optionally the NMS security log (.xlsx) to begin analysis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Expected Column Schema")
    schema = pd.DataFrame({
        "Column": [COL_TS, COL_DL, COL_UL, COL_PING, COL_DROP, COL_SNR, COL_OBSTR],
        "Type":   ["datetime", "float", "float", "float", "float (0-1)", "float", "float (0-1)"],
        "Description": [
            "UTC timestamp", "Downlink throughput (Mbps)", "Uplink throughput (Mbps)",
            "Avg POP ping latency (ms)", "Avg packet drop rate",
            "Avg receive SNR", "Obstruction % time"
        ]
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.stop()

# ─── Load & process ───────────────────────────────────────────
region_labels = ["REGION A", "REGION B"]
all_dfs   = []
all_names = []
logs_df   = load_security_logs(xlsx_file) if xlsx_file else pd.DataFrame()

for i, f in enumerate(csv_files[:2]):
    raw = load_telemetry(f)
    all_dfs.append(raw)
    all_names.append(region_labels[i] if i < len(region_labels) else f"REGION {i+1}")

# Apply simulation if active
if st.session_state.get("simulation_active", False):
    st.markdown("""
    <div class="alert-critical">
    ⚡ STRESS TEST MODE ACTIVE — Synthetic DoS attack injected into final 12 data points
    </div>
    """, unsafe_allow_html=True)
    sim_results = []
    for df_raw in all_dfs:
        sim_df, logs_df = simulate_advanced_attack(df_raw, logs_df)
        sim_results.append(sim_df)
    all_dfs = sim_results

# Run full ML pipeline on each region
analyzed = {}
scalers  = {}
for name, df_raw in zip(all_names, all_dfs):
    df_out, model_out, scaler_out = full_analysis(df_raw, logs_df)
    analyzed[name] = df_out
    scalers[name]  = scaler_out

combined_df = pd.concat(analyzed.values(), ignore_index=True)

# ─────────────────────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────────────────────
total_pts    = len(combined_df)
total_attack = (combined_df["status"] == "CYBER ATTACK").sum()
total_jam    = (combined_df["status"] == "SIGNAL JAMMING").sum()
total_cong   = (combined_df["status"] == "NETWORK CONGESTION").sum()
threat_pct   = round((total_attack + total_jam) / total_pts * 100, 1) if total_pts else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("DATA POINTS",      f"{total_pts:,}")
c2.metric("CYBER ATTACKS",    total_attack,  delta="HIGH RISK" if total_attack > 5 else "LOW RISK", delta_color="inverse")
c3.metric("SIGNAL JAMMING",   total_jam)
c4.metric("CONGESTION EVENTS",total_cong)
c5.metric("THREAT INDEX",     f"{threat_pct}%", delta="↑ ELEVATED" if threat_pct > 5 else "↓ NORMAL", delta_color="inverse")

st.markdown("")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_global, tab_forensic, tab_raw = st.tabs([
    "GLOBAL ANALYTICS",
    "FORENSIC ANALYSIS",
    "RAW TELEMETRY"
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL ANALYTICS
# ═════════════════════════════════════════════════════════════
with tab_global:

    for region_name, df_region in analyzed.items():
        st.markdown(f'<div class="section-label">{region_name}</div>', unsafe_allow_html=True)

        col_chart, col_pie = st.columns([3, 1])

        with col_chart:
            fig_ping = ts_plot_with_anomalies(
                df_region, COL_PING, "Latency (ms)",
                f"{region_name} — POP Ping Latency with Anomaly Zones"
            )
            st.plotly_chart(fig_ping, use_container_width=True)

        with col_pie:
            st.plotly_chart(status_distribution_pie(df_region), use_container_width=True)
            # Incident summary
            inc = df_region["status"].value_counts()
            for s, cnt in inc.items():
                badge_cls = {"STABLE":"badge-stable","CYBER ATTACK":"badge-attack",
                             "SIGNAL JAMMING":"badge-jamming","NETWORK CONGESTION":"badge-congestion"}.get(s,"badge-info")
                st.markdown(f'<span class="{badge_cls}">{s}: {cnt}</span><br>', unsafe_allow_html=True)

        # Second row — drop rate + throughput
        col_drop, col_thr = st.columns(2)
        with col_drop:
            fig_drop = ts_plot_with_anomalies(
                df_region, COL_DROP, "Drop Rate",
                "Packet Drop Rate"
            )
            st.plotly_chart(fig_drop, use_container_width=True)
        with col_thr:
            fig_thr = ts_plot_with_anomalies(
                df_region, COL_DL, "Throughput (Mbps)",
                "Downlink Throughput"
            )
            st.plotly_chart(fig_thr, use_container_width=True)

    # Correlation Score heatmap (if 2 regions)
    if len(analyzed) == 2:
        st.markdown('<div class="section-label" style="margin-top:16px">Multi-Region Comparative View</div>',
                    unsafe_allow_html=True)
        combined_df["hour"] = combined_df[COL_TS].dt.floor("h")
        grp = combined_df.groupby(["hour","REGION_TAG"] if "REGION_TAG" in combined_df.columns else ["hour"])[COL_PING].mean().reset_index()
        fig_comp = px.line(combined_df, x=COL_TS, y="correlation_score",
                           color="REGION_TAG" if "REGION_TAG" in combined_df.columns else None,
                           title="SNR-Throughput Correlation Score (higher = more suspicious)",
                           color_discrete_sequence=["#58A6FF", "#E3B341"],
                           template="plotly_dark")
        fig_comp.add_hline(y=0.65, line_dash="dot", line_color="#F85149",
                           annotation_text="Attack threshold (0.65)", annotation_font_color="#F85149")
        fig_comp.update_layout(height=280, **PLOTLY_TEMPLATE)
        st.plotly_chart(fig_comp, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# TAB 2 — FORENSIC ANALYSIS (XAI)
# ═════════════════════════════════════════════════════════════
with tab_forensic:
    st.markdown('<div class="section-label">Incident Selection</div>', unsafe_allow_html=True)

    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        region_sel = st.selectbox("Region", list(analyzed.keys()), key="forensic_region")
    with col_sel2:
        df_sel = analyzed[region_sel]
        anomaly_df = df_sel[df_sel["status"] != "STABLE"].copy()
        if anomaly_df.empty:
            st.info("No anomalies detected in selected region.")
            st.stop()

        incident_idx = st.selectbox(
            "Select Incident (timestamp · status)",
            anomaly_df.index,
            format_func=lambda i: f"{df_sel.loc[i, COL_TS].strftime('%Y-%m-%d %H:%M')}  ·  {df_sel.loc[i, 'status']}"
        )

    row = df_sel.loc[incident_idx]
    scaler = scalers[region_sel]

    # ── Incident header ───────────────────────────────────────
    s = row["status"]
    badge_map = {"CYBER ATTACK":"badge-attack","SIGNAL JAMMING":"badge-jamming",
                 "NETWORK CONGESTION":"badge-congestion"}
    badge = badge_map.get(s, "badge-stable")
    st.markdown(f"""
    <div style="background:#0D1117;border:1px solid #21262D;padding:16px;border-radius:8px;margin:12px 0;">
      <span class="{badge}">{s}</span>
      <span style="font-family:'JetBrains Mono',monospace;color:#8B949E;font-size:0.8rem;margin-left:12px">
        {row[COL_TS].strftime('%Y-%m-%d %H:%M UTC')}
      </span>
      <br><br>
      <span style="color:#CDD9E5;font-size:0.85rem">
        IsolationForest anomaly score: <b style="color:#F85149">{row['if_raw']:.4f}</b> &nbsp;|&nbsp;
        Packet drop: <b>{row[COL_DROP]*100:.2f}%</b> &nbsp;|&nbsp;
        Latency: <b>{row[COL_PING]:.1f} ms</b> &nbsp;|&nbsp;
        SNR: <b>{row[COL_SNR]:.3f}</b> &nbsp;|&nbsp;
        NMS Illegal RIP: <b style="color:{'#F85149' if row.get('nms_illegal_rip', False) else '#3FB950'}">{'YES' if row.get('nms_illegal_rip', False) else 'NO'}</b>
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── XAI feature attribution ───────────────────────────────
    contributions = explain_incident(row, scaler)
    col_xai, col_rules = st.columns([3, 2])

    with col_xai:
        st.plotly_chart(xai_bar_chart(contributions, s), use_container_width=True)

    with col_rules:
        st.markdown("**Classification Logic**")
        rules = {
            "CYBER ATTACK": [
                f"✓ IsolationForest flagged as anomaly (score < 0)",
                f"✓ Packet drop > 5%: {row[COL_DROP]*100:.2f}%",
                f"✓ NMS Illegal RIP event: {'YES' if row.get('nms_illegal_rip', False) else 'NO'}",
                f"✓ Correlation Score: {row['correlation_score']:.3f} (threshold 0.65)",
            ],
            "SIGNAL JAMMING": [
                f"✓ IsolationForest flagged as anomaly",
                f"✓ SNR below 0.5: {row[COL_SNR]:.3f}",
                f"✓ Z-score SNR: {row['z_snr']:.2f} (threshold {z_threshold})",
            ],
            "NETWORK CONGESTION": [
                f"✓ IsolationForest flagged as anomaly",
                f"✓ Ping Z-score: {row['z_ping']:.2f} (threshold {z_threshold})",
                f"✓ No NMS log correlation",
            ]
        }
        for rule in rules.get(s, []):
            st.markdown(f"<span style='font-size:0.82rem;color:#CDD9E5'>{rule}</span>", unsafe_allow_html=True)

    # ── Context window: 2-hour window around incident ─────────
    st.markdown('<div class="section-label" style="margin-top:16px">Incident Context Window (±2h)</div>',
                unsafe_allow_html=True)
    t0 = row[COL_TS]
    window = df_sel[(df_sel[COL_TS] >= t0 - pd.Timedelta("2h")) &
                    (df_sel[COL_TS] <= t0 + pd.Timedelta("2h"))]

    fig_ctx = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["Packet Drop Rate", "Ping Latency (ms)"],
                             vertical_spacing=0.12)

    fig_ctx.add_trace(go.Scatter(x=window[COL_TS], y=window[COL_DROP],
                                  mode="lines+markers", name="Drop Rate",
                                  line=dict(color="#58A6FF", width=1.5)), row=1, col=1)
    fig_ctx.add_trace(go.Scatter(x=window[COL_TS], y=window[COL_PING],
                                  mode="lines+markers", name="Latency",
                                  line=dict(color="#E3B341", width=1.5)), row=2, col=1)
    # Mark incident
    fig_ctx.add_vline(x=t0, line_color="#F85149", line_dash="dash",
                      annotation_text="INCIDENT", annotation_font_color="#F85149")
    fig_ctx.update_layout(height=320, showlegend=False, **PLOTLY_TEMPLATE)
    st.plotly_chart(fig_ctx, use_container_width=True)

    # ── Z-score table for context window ─────────────────────
    st.markdown('<div class="section-label" style="margin-top:8px">Statistical Profile (Context Window)</div>',
                unsafe_allow_html=True)
    z_summary = pd.DataFrame({
        "Metric": ["Ping Latency Z-score", "Packet Drop Z-score", "SNR Z-score", "Correlation Score"],
        "Incident Value": [
            f"{row['z_ping']:.3f}", f"{row['z_drop']:.3f}",
            f"{row['z_snr']:.3f}",  f"{row['correlation_score']:.3f}"
        ],
        "Window Mean": [
            f"{window['z_ping'].mean():.3f}",  f"{window['z_drop'].mean():.3f}",
            f"{window['z_snr'].mean():.3f}",   f"{window['correlation_score'].mean():.3f}"
        ],
        "Alert Threshold": [f">{z_threshold}", f">{z_threshold}", f">{z_threshold}", ">0.65"],
        "Triggered": [
            "⚠️" if row["z_ping"] > z_threshold else "✓",
            "⚠️" if row["z_drop"] > z_threshold else "✓",
            "⚠️" if row["z_snr"]  > z_threshold else "✓",
            "⚠️" if row["correlation_score"] > 0.65 else "✓"
        ]
    })
    st.dataframe(z_summary, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# TAB 3 — RAW TELEMETRY
# ═════════════════════════════════════════════════════════════
with tab_raw:
    st.markdown('<div class="section-label">Full Processed Dataset</div>', unsafe_allow_html=True)

    region_filter = st.selectbox("Region", ["ALL"] + list(analyzed.keys()), key="raw_region")
    status_filter = st.multiselect("Status Filter", list(STATUS_COLORS.keys()), default=list(STATUS_COLORS.keys()))

    view_df = combined_df if region_filter == "ALL" else analyzed[region_filter]
    view_df = view_df[view_df["status"].isin(status_filter)]

    display_cols = [COL_TS, COL_PING, COL_DROP, COL_DL, COL_SNR, COL_OBSTR,
                    "z_ping", "z_drop", "correlation_score", "if_raw", "nms_illegal_rip", "status"]
    display_cols = [c for c in display_cols if c in view_df.columns]

    st.dataframe(
        view_df[display_cols].rename(columns=FEATURE_LABELS).reset_index(drop=True),
        use_container_width=True
    )
    st.caption(f"Showing {len(view_df):,} rows · {(view_df['status']!='STABLE').sum()} anomalies")
