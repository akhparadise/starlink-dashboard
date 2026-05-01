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
    snr_norm  = (df[COL_SNR] - df[COL_SNR].min()) / (df[COL_SNR].max() - df[COL_SNR].min() + 1e-9)
    dl_norm   = (df[COL_DL]  - df[COL_DL].min())  / (df[COL_DL].max()  - df[COL_DL].min()  + 1e-9)
    df["correlation_score"] = snr_norm * (1 - dl_norm)
 
    # ── Obstruction indicator ────────────────────────────────────
    df["obstr_flag"] = (df[COL_OBSTR] > 0).astype(int)
 
    return df
 
# ─────────────────────────────────────────────────────────────
# DATA FUSION
# ─────────────────────────────────────────────────────────────
def fuse_with_logs(df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nms_illegal_rip"] = False
    df["nms_severity"]    = "NONE"
 
    if logs.empty or "_ts" not in logs.columns:
        return df
 
    desc_col = [c for c in logs.columns if "description" in c.lower() or "event" in c.lower()]
    if not desc_col:
        return df
    desc_col = desc_col[0]
 
    rip_logs = logs[logs[desc_col].str.contains("Illegal unicast RIP|Illegal RIP", case=False, na=False)].copy()
    if rip_logs.empty:
        return df
 
    rip_logs = rip_logs.dropna(subset=["_ts"])
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
    df["if_score"]  = model.fit_predict(X_scaled)
    df["if_raw"]    = model.decision_function(X_scaled)
    return df, model, scaler
 
def classify_incidents(row) -> str:
    is_anomaly = row["if_score"] == -1
    if not is_anomaly: return "STABLE"
    if row[COL_DROP] > 0.05 and row.get("nms_illegal_rip", False): return "CYBER ATTACK"
    if row["correlation_score"] > 0.65 and row["z_drop"] > 2.5: return "CYBER ATTACK"
    if row[COL_SNR] < 0.5 or row["z_snr"] > 2.5: return "SIGNAL JAMMING"
    if row["z_ping"] > 2.0: return "NETWORK CONGESTION"
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
    df_sim = df.copy()
    n = len(df_sim)
    attack_idx = range(max(0, n - 12), n)
    rng = np.random.default_rng(seed=42)
    df_sim.loc[df_sim.index[attack_idx], COL_DROP] = rng.uniform(0.12, 0.20, len(attack_idx))
    df_sim.loc[df_sim.index[attack_idx], COL_PING] = df_sim[COL_PING].median() * rng.uniform(2.1, 3.0, len(attack_idx))
    df_sim.loc[df_sim.index[attack_idx], COL_DL]   = df_sim[COL_DL].median()   * rng.uniform(0.05, 0.15, len(attack_idx))
 
    attack_times = df_sim[COL_TS].iloc[list(attack_idx)].values
    syn_logs = pd.DataFrame({
        "_ts": attack_times,
        "Event Description": ["Illegal unicast RIP message from 10.0.0.1" for _ in attack_times],
        "Severity": ["CRITICAL" for _ in attack_times],
        "Element Name": ["SIM-INJECT" for _ in attack_times],
    })
    merged_logs = syn_logs if logs.empty else pd.concat([logs, syn_logs], ignore_index=True)
    return df_sim, merged_logs
 
# ─────────────────────────────────────────────────────────────
# XAI — FIX: MULTIPLE VALUES FOR KEYWORD ARGUMENT
# ─────────────────────────────────────────────────────────────
def xai_bar_chart(contributions: dict, incident_status: str) -> go.Figure:
    feats = list(contributions.keys())
    vals  = list(contributions.values())
    color = STATUS_COLORS.get(incident_status, "#58A6FF")
 
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=[color] * len(vals), opacity=[0.4 + 0.6 * v for v in vals]),
        text=[f"{v*100:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#CDD9E5", size=10)
    ))
    
    # Сначала применяем общий академический шаблон
    fig.update_layout(**PLOTLY_TEMPLATE)
    # Затем переопределяем специфичные поля, чтобы избежать TypeError
    fig.update_layout(
        title=dict(text="Feature Attribution (XAI)", font=dict(size=12, color="#E6EDF3")),
        height=320,
        margin=dict(l=160, r=60, t=40, b=30),
    )
    # Настраиваем ось отдельно
    fig.update_xaxes(title="Relative Contribution", tickformat=".0%")
    return fig
 
# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────
def ts_plot_with_anomalies(df: pd.DataFrame, y_col: str, y_label: str, title: str) -> go.Figure:
    fig = go.Figure()
    normal = df[df["status"] == "STABLE"]
    fig.add_trace(go.Scatter(x=normal[COL_TS], y=normal[y_col], mode="lines", name="STABLE", line=dict(color="#3FB950", width=1.2)))
 
    for status, color in STATUS_COLORS.items():
        if status == "STABLE": continue
        sub = df[df["status"] == status]
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub[COL_TS], y=sub[y_col], mode="markers", name=status, marker=dict(color=color, size=7, symbol="circle-open", line=dict(width=2))))
 
    fig.update_layout(title=dict(text=title, font=dict(size=13, color="#E6EDF3")), yaxis_title=y_label, height=280, margin=dict(l=40, r=20, t=40, b=30), **PLOTLY_TEMPLATE)
    return fig
 
def status_distribution_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    fig = px.pie(counts, names="status", values="count", color="status", color_discrete_map=STATUS_COLORS, hole=0.55)
    fig.update_traces(textinfo="label+percent", textfont_color="#E6EDF3")
    fig.update_layout(showlegend=False, height=240, margin=dict(l=10, r=10, t=20, b=10), **PLOTLY_TEMPLATE)
    return fig

def explain_incident(row: pd.Series, scaler: StandardScaler) -> dict:
    vals = row[FEATURE_COLS].fillna(0).values.reshape(1, -1)
    scaled = scaler.transform(vals)[0]
    raw_contrib = np.abs(scaled)
    total = raw_contrib.sum() + 1e-9
    normed = raw_contrib / total
    return {FEATURE_LABELS[f]: float(normed[i]) for i, f in enumerate(FEATURE_COLS)}
 
# ─────────────────────────────────────────────────────────────
# HEADER & SIDEBAR
# ─────────────────────────────────────────────────────────────
st.markdown("""<div class="header-band"><span class="system-title">⬡ NADS &nbsp;|&nbsp; Network Attack Detection System</span><br><span class="system-subtitle">KBTU Master's Thesis &nbsp;·&nbsp; Starlink Telemetry Analytics &nbsp;·&nbsp; ML-Powered Threat Classification</span></div>""", unsafe_allow_html=True)
 
with st.sidebar:
    st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)
    csv_files  = st.file_uploader("Telemetry CSV", type="csv",  accept_multiple_files=True)
    xlsx_file  = st.file_uploader("Security Log", type=["xlsx", "xls"])
    run_sim = st.button("⚡ INJECT SYNTHETIC ATTACK")
    if run_sim: st.session_state["simulation_active"] = True
    if st.button("↺ RESET"): st.session_state["simulation_active"] = False
    contamination = st.slider("Contamination", 0.02, 0.15, 0.06)
    z_threshold   = st.slider("Z-Threshold", 1.5, 4.0, 2.5)
 
# ─────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────
if not csv_files:
    st.info("Upload CSV files to begin.")
    st.stop()
 
region_labels = ["REGION A", "REGION B"]
all_dfs, all_names = [], []
logs_df = load_security_logs(xlsx_file) if xlsx_file else pd.DataFrame()
 
for i, f in enumerate(csv_files[:2]):
    raw = load_telemetry(f)
    all_dfs.append(raw)
    all_names.append(region_labels[i] if i < len(region_labels) else f"REGION {i+1}")
 
if st.session_state.get("simulation_active", False):
    st.error("STRESS TEST MODE ACTIVE")
    all_dfs = [simulate_advanced_attack(df, logs_df)[0] for df in all_dfs]
 
analyzed, scalers = {}, {}
for name, df_raw in zip(all_names, all_dfs):
    df_out, model_out, scaler_out = full_analysis(df_raw, logs_df)
    analyzed[name] = df_out
    scalers[name]  = scaler_out
 
combined_df = pd.concat(analyzed.values(), ignore_index=True)
 
# ─────────────────────────────────────────────────────────────
# TABS & VISUALS
# ─────────────────────────────────────────────────────────────
tab_global, tab_forensic, tab_raw = st.tabs(["GLOBAL ANALYTICS", "FORENSIC ANALYSIS", "RAW TELEMETRY"])
 
with tab_global:
    for region_name, df_region in analyzed.items():
        st.markdown(f'<div class="section-label">{region_name}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1: st.plotly_chart(ts_plot_with_anomalies(df_region, COL_PING, "Latency", f"{region_name} Latency"), use_container_width=True)
        with c2: st.plotly_chart(status_distribution_pie(df_region), use_container_width=True)
 
with tab_forensic:
    region_sel = st.selectbox("Region", list(analyzed.keys()), key="f_reg")
    df_sel = analyzed[region_sel]
    anomaly_df = df_sel[df_sel["status"] != "STABLE"].copy()
    
    if anomaly_df.empty:
        st.info("No anomalies.")
    else:
        incident_idx = st.selectbox("Select Incident", anomaly_df.index, format_func=lambda i: f"{df_sel.loc[i, COL_TS]} · {df_sel.loc[i, 'status']}")
        row = df_sel.loc[incident_idx]
        s = row["status"]
        
        # FIX: ПЕРЕВОДИМ t0 В СТРОКУ ДЛЯ ПРЕДОТВРАЩЕНИЯ ОШИБКИ СЛОЖЕНИЯ TIMESTAMP
        t0 = row[COL_TS]
        t0_str = t0.strftime('%Y-%m-%d %H:%M:%S')
        
        st.markdown(f"Status: **{s}** | Score: {row['if_raw']:.4f}")
        
        # XAI
        st.plotly_chart(xai_bar_chart(explain_incident(row, scalers[region_sel]), s), use_container_width=True)
        
        # Context Window
        window = df_sel[(df_sel[COL_TS] >= t0 - pd.Timedelta("2h")) & (df_sel[COL_TS] <= t0 + pd.Timedelta("2h"))]
        fig_ctx = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_ctx.add_trace(go.Scatter(x=window[COL_TS], y=window[COL_DROP], name="Drop"), row=1, col=1)
        fig_ctx.add_trace(go.Scatter(x=window[COL_TS], y=window[COL_PING], name="Ping"), row=2, col=1)
        
        # FIX: ИСПОЛЬЗУЕМ СТРОКОВОЕ ЗНАЧЕНИЕ ДЛЯ ВЕРТИКАЛЬНОЙ ЛИНИИ
        fig_ctx.add_vline(x=t0_str, line_color="#F85149", line_dash="dash", annotation_text="INCIDENT")
        fig_ctx.update_layout(height=400, **PLOTLY_TEMPLATE)
        st.plotly_chart(fig_ctx, use_container_width=True)
 
with tab_raw:
    st.dataframe(combined_df)
