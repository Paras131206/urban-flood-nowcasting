import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import sqlite3
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# --- 1. SYSTEM SETUP ---
# Auto-refresh every 5 minutes to check IMD updates
st_autorefresh(interval=300000, key="risk_timer") 
st.set_page_config(page_title="Predictive Drainage Monitor", layout="wide")

# --- 2. VOICE NOTIFICATION ENGINE (Fixed for Browser Security) ---
def trigger_voice_alert(message):
    # This JS function only works if the user has clicked something on the page first
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{message}');
    msg.rate = 0.9; // Slightly slower for clarity
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0)

# --- 3. RISK PREDICTION LOGIC ---
def predict_risk_level(row, intensity):
    # Effective Capacity = Capacity reduced by blockage
    eff_capacity = row['Max_Flow_Capacity_m3s'] * (1 - (row['Blockage_Pct'] / 100))
    
    # Runoff Load = Area * Intensity * Coefficient
    load = (0.9 * (intensity/1000) * row['Catchment_Area_sqm']) / 3600
    
    # Hydraulic Ratio
    ratio = load / eff_capacity if eff_capacity > 0 else 10
    
    # Terrain Factor: Low elevation (basins) adds risk
    terrain_penalty = 0.3 if row['Elevation_m'] < 3.0 else 0.0
    
    final_score = ratio + terrain_penalty
    
    if final_score > 0.9: return "HIGH", "red", final_score
    if final_score > 0.5: return "MEDIUM", "orange", final_score
    return "LOW", "green", final_score

# --- 4. DASHBOARD UI ---
st.title("🛡️ Bandra Predictive Risk Control Center")

# Sidebar for Activation and IMD Simulation
with st.sidebar:
    st.header("Control Panel")
    # THE FIX: User must click this to enable audio in browser
    voice_enabled = st.checkbox("🔈 Enable Voice Notifications", value=False)
    st.info("Note: Browser blocks voice until you interact with the page.")
    
    st.divider()
    st.header("IMD Simulation")
    intensity = st.slider("Rain Intensity (mm/hr)", 0, 150, 45)

# --- 5. DATA PROCESSING ---
try:
    df = pd.read_csv("bandra_capacity.csv")
    
    # Run Prediction Engine
    df[['Risk_Level', 'Color', 'Score']] = df.apply(
        lambda x: pd.Series(predict_risk_level(x, intensity)), axis=1
    )

    # UI Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Intensity", f"{intensity} mm/hr")
    m2.metric("Critical Points", len(df[df['Risk_Level'] == 'HIGH']))
    m3.metric("System Risk", df['Risk_Level'].mode()[0])

    # --- 6. MAP & VOICE ALERTS ---
    st.subheader("📍 Predicted Risk Hotspots")
    m = folium.Map(location=[19.0544, 72.8402], zoom_start=15, tiles="OpenStreetMap")

    highest_risk_area = ""
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=15,
            color=row['Color'],
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Area: {row['Segment_Name']} | Risk: {row['Risk_Level']}"
        ).add_to(m)

        if row['Risk_Level'] == "HIGH":
            highest_risk_area = row['Segment_Name']

    st_folium(m, width=1200, height=500)

    # TRIGGER VOICE: Only if checkbox is ON and area is HIGH risk
    if voice_enabled and highest_risk_area != "":
        trigger_voice_alert(f"Alert. High flood risk predicted at {highest_risk_area}. Drainage capacity exceeded.")
        st.error(f"🚨 CRITICAL ALERT: {highest_risk_area} is at High Risk!")

    # --- 7. DETAILED RISK TABLE ---
    st.subheader("📋 Terrain & Condition Based Assessment")
    st.table(df[['Segment_Name', 'Elevation_m', 'Blockage_Pct', 'Risk_Level']])

except Exception as e:
    st.error(f"CSV Configuration Error: {e}")