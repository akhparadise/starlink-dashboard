import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="STARLINK | MISSION CONTROL", layout="wide")

# --- PROFESSIONAL DARK THEME (STARLINK STYLE) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3, p { color: #FFFFFF !important; }
    [data-testid="stMetric"] {
        background-color: #0A0A0A !important;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 4px;
    }
    [data-testid="stMetricLabel"] { color: #AAAAAA !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    .attack-text {
        color: #FF0000;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: #000000; border-bottom: 1px solid #333333; }
    .stTabs [data-baseweb="tab"] { color: #888888; }
    .stTabs [data-baseweb="tab--active"] { color: #FFFFFF !important; border-bottom-color: #FFFFFF !important; }
    
    /* Улучшение читаемости таблиц */
    .stTable { color: #FFFFFF !important; }
    thead tr th { background-color: #111111 !important; color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC ---
def analyze_data(df):
    df.columns = df.columns.str.strip()
    # Базовые признаки для ИИ
    features = ['health.ut.pop_ping_latency_ms_avg', 'health.ut.rx_avg_snr']
    
    model = IsolationForest(contamination=0.07, random_state=42)
    df['anomaly_code'] = model.fit_predict(df[features].fillna(0))
    
    avg_ping = df['health.ut.pop_ping_latency_ms_avg'].median()

    def get_root_cause(row):
        if row['anomaly_code'] == 1: return "STABLE"
        # Логика детекции DoS-атак
        if row['health.ut.pop_ping_latency_ms_avg'] > avg_ping * 1.5 and row['health.ut.rx_avg_snr'] > 0.90:
            return "CYBER ATTACK (DoS)"
        elif row['health.ut.rx_avg_snr'] < 0.75:
            return "SIGNAL OBSTRUCTION"
        return "NETWORK CONGESTION"

    df['status'] = df.apply(get_root_cause, axis=1)
    return df

# --- HEADER ---
col_logo, col_nav = st.columns([1, 4])
with col_logo:
    st.title("STARLINK")
with col_nav:
    st.write("\n")
    st.markdown("ACCOUNT &nbsp;&nbsp; | &nbsp;&nbsp; **DASHBOARD** &nbsp;&nbsp; | &nbsp;&nbsp; SHOP &nbsp;&nbsp; | &nbsp;&nbsp; HELP CENTER", unsafe_allow_html=True)

st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("FILE UPLOAD")
uploaded_files = st.sidebar.file_uploader("Upload Telemetry CSV (Atyrau, Akmola, Almaty, Pavlodar)", type="csv", accept_multiple_files=True)

REGIONS = ["ATYRAU REGION", "AKMOLA REGION", "ALMATY REGION", "PAVLODAR REGION"]

if uploaded_files:
    processed_data = {}
    full_combined_df = pd.DataFrame()

    for i, file in enumerate(uploaded_files):
        if i < len(REGIONS):
            region_name = REGIONS[i]
            df = pd.read_csv(file)
            analyzed_df = analyze_data(df)
            analyzed_df['REGION_TAG'] = region_name
            processed_data[region_name] = analyzed_df
            full_combined_df = pd.concat([full_combined_df, analyzed_df], ignore_index=True)

    # --- MAIN TABS ---
    t1, t2 = st.tabs(["GLOBAL MONITOR", "REGIONAL ANALYSIS"])

    with t1:
        st.subheader("SERVICE LINES STATUS")
        
        # РАСШИРЕННАЯ ТАБЛИЦА СО ВСЕМИ ПАРАМЕТРАМИ
        status_list = []
        for name, data in processed_data.items():
            attacks = len(data[data['status'] == "CYBER ATTACK (DoS)"])
            health = "INACTIVE" if attacks > 5 else "ONLINE"
            color = "🔴" if health == "INACTIVE" else "🟢"
            
            # Динамический расчет параметров из датасета
            downlink = f"{round(data['health.ut.rx_throughput_bps'].mean() / 1e6, 2)} Mbps" if 'health.ut.rx_throughput_bps' in data.columns else "0.01Mbps"
            uplink = f"{round(data['health.ut.tx_throughput_bps'].mean() / 1e6, 2)} Mbps" if 'health.ut.tx_throughput_bps' in data.columns else "0.01Mbps"
            ping_drop = f"{round(data['health.ut.pop_ping_drop_rate_avg'].mean() * 100, 2)}%" if 'health.ut.pop_ping_drop_rate_avg' in data.columns else "0.00%"
            latency = f"{int(data['health.ut.pop_ping_latency_ms_avg'].mean())}ms"
            sw_version = data['health.ut.software_version'].iloc[-1] if 'health.ut.software_version' in data.columns else "2025.stable"
            
            status_list.append({
                "STATUS": color + " " + health,
                "SERVICE LINE": name,
                "ALERTS": attacks,
                "DOWNLINK": downlink,
                "UPLINK": uplink,
                "PING DROP": ping_drop,
                "LATENCY": latency,
                "SIGNAL": f"{round(data['health.ut.rx_avg_snr'].mean(), 2)} SNR",
                "SOFTWARE": sw_version
            })
        
        st.table(pd.DataFrame(status_list))
        
        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        total_attacks = sum(len(d[d['status'] == "CYBER ATTACK (DoS)"]) for d in processed_data.values())
        m1.metric("TOTAL SYSTEMS", len(processed_data))
        m2.metric("ONLINE STATUS", "ACTIVE")
        m3.metric("THREAT LEVEL", f"{total_attacks}", delta="CRITICAL" if total_attacks > 10 else "LOW", delta_color="inverse")
        m4.metric("GLOBAL AVG LATENCY", f"{int(full_combined_df['health.ut.pop_ping_latency_ms_avg'].mean())} ms")

        st.markdown("---")
        st.subheader("📋 RAW TELEMETRY ARCHIVE (FULL DATASET)")
        st.dataframe(full_combined_df, use_container_width=True)

    with t2:
        selected_region = st.selectbox("SELECT REGION", list(processed_data.keys()))
        data = processed_data[selected_region]
        
        col_graph, col_info = st.columns([3, 1])
        
        with col_graph:
            fig = px.scatter(data, x=data.index, y='health.ut.pop_ping_latency_ms_avg', 
                          color='status', title=f"REAL-TIME DATA STREAM: {selected_region}",
                          color_discrete_map={
                              "STABLE": "#00FF00", 
                              "CYBER ATTACK (DoS)": "#FF0000",
                              "SIGNAL OBSTRUCTION": "#FFA500",
                              "NETWORK CONGESTION": "#555555"
                          }, template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_info:
            st.markdown("### INCIDENT LOG")
            attack_logs = data[data['status'] == "CYBER ATTACK (DoS)"]
            
            if not attack_logs.empty:
                st.markdown(f'<p class="attack-text">⚠️ CRITICAL: {len(attack_logs)} ATTACKS DETECTED</p>', unsafe_allow_html=True)
                st.dataframe(attack_logs[['status']].head(10))
            else:
                # КРАСНЫЙ БЛОК СТАТУСА БЕЗОПАСНОСТИ
                st.markdown("""
                    <div style="border: 2px solid #FF0000; padding: 30px 10px; border-radius: 10px; text-align: center; margin-top: 20px; background-color: rgba(255, 0, 0, 0.05);">
                        <h1 style="color: #FF0000; margin: 0; font-size: 3rem;">🛡️</h1>
                        <h2 style="color: #FF0000; margin: 10px 0; font-weight: bold; text-transform: uppercase; font-size: 1.5rem;">NO THREATS DETECTED</h2>
                        <p style="color: #FFFFFF; font-size: 0.9rem; opacity: 0.8;">SYSTEM INTEGRITY: 100%</p>
                    </div>
                """, unsafe_allow_html=True)

else:
    st.info("WAITING FOR TELEMETRY DATA UPLOAD...")
