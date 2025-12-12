import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.express as px
import plotly.graph_objects as go
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Data Center Network Optimizer")
DEFAULT_FILE_PATH = '240_Project_Master_File.csv'

# --- HELPER FUNCTIONS ---
def calculate_latency_matrix(df_demand, df_candidates, latency_factor):
    """Calculates Latency (ms) between all demand cities and candidate sites."""
    R = 3958.8 
    
    lat1 = np.radians(df_demand['Latitude'].values).reshape(-1, 1)
    lon1 = np.radians(df_demand['Longitude'].values).reshape(-1, 1)
    lat2 = np.radians(df_candidates['Latitude'].values).reshape(1, -1)
    lon2 = np.radians(df_candidates['Longitude'].values).reshape(1, -1)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    dist_miles = R * c
    latency = dist_miles * latency_factor
    latency[latency < 0.5] = 0.5
    return latency

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    col_mapping = {
        'City / Metropolitan Area': 'City',
        'Realistic GW per City': 'GW',
        'Metro Pop (In-State)': 'Population',
        'SD Cents/KWH': 'Cost_SD',
        'SD C02 (lb/MWh)': 'CO2_SD',
        'Cents/KWH': 'Cost_Raw',
        'C02 (lb/MWh)': 'CO2_Raw',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'State': 'State'
    }
    
    for col in df.columns:
        if 'SD Cents/KWH' in col:
            col_mapping[col] = 'Cost_SD'
    
    existing_cols = [c for c in col_mapping.keys() if c in df.columns]
    df = df[existing_cols].copy()
    df = df.rename(columns=col_mapping)
    
    if 'Population' in df.columns and df['Population'].dtype == object:
        df['Population'] = df['Population'].astype(str).str.replace(',', '').astype(float)
        
    numeric_cols = ['GW', 'Latitude', 'Longitude', 'Cost_SD', 'CO2_SD', 'Cost_Raw', 'CO2_Raw']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    crit_cols = ['City', 'GW', 'Population', 'Latitude', 'Longitude', 'Cost_SD', 'CO2_SD']
    df = df.dropna(subset=[c for c in crit_cols if c in df.columns])
    
    return df.reset_index(drop=True)

def run_optimization(df, num_centers, gw_limit, max_latency, co2_penalty, target_mode, target_val):
    
    df = df.reset_index(drop=True)
    df_demand = df.copy()
    
    df_candidates = df.loc[df['GW'] >= gw_limit].reset_index(drop=True)
    
    num_demand = len(df_demand)
    num_candidates = len(df_candidates)
    
    if num_candidates < 1:
        return "Error", f"No cities meet the {gw_limit} GW requirement.", None, None, 0

    populations = df_demand['Population'].values
    costs_sd = df_candidates['Cost_SD'].values
    emissions_sd = df_candidates['CO2_SD'].values
    
    latency_factor = 1.0/50.0
    lat_matrix = calculate_latency_matrix(df_demand, df_candidates, latency_factor)
    valid_connection = (lat_matrix <= max_latency).astype(int)

    # ------------------------------------------------
    # STAGE 1: Maximize Population (Feasibility)
    # ------------------------------------------------
    m1 = pulp.LpProblem("Stage1", pulp.LpMaximize)
    y1 = pulp.LpVariable.dicts("Build1", range(num_candidates), cat='Binary')
    z1 = pulp.LpVariable.dicts("Served1", range(num_demand), cat='Binary')
    
    m1 += pulp.lpSum(populations[i] * z1[i] for i in range(num_demand))
    m1 += pulp.lpSum(y1[j] for j in range(num_candidates)) <= num_centers
    
    for i in range(num_demand):
        reachable = [j for j in range(num_candidates) if valid_connection[i, j] == 1]
        if not reachable:
            m1 += z1[i] == 0
        else:
            m1 += z1[i] <= pulp.lpSum(y1[j] for j in reachable)
            
    m1.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[m1.status] != 'Optimal':
        return "Error", "Stage 1 Failed. Constraints too strict.", None, None, 0
        
    max_possible_pop = pulp.value(m1.objective)
    
    # ------------------------------------------------
    # STAGE 2: Cost & CO2 (With Target Logic)
    # ------------------------------------------------
    
    # Calculate constraint based on Mode
    status_msg = "Optimization Successful."
    
    if target_mode == 'Percent':
        # val is 0.0 to 1.0
        final_target_pop = max_possible_pop * target_val
        status_msg += f" Target set to {target_val*100:.0f}% of max ({final_target_pop:,.0f})."
    else:
        # val is absolute number
        final_target_pop = target_val
        
        # SAFETY CHECK: If manual target is impossible, cap it
        if final_target_pop > max_possible_pop:
            status_msg = f"Warning: Manual Target ({final_target_pop:,.0f}) exceeds Max Reachable ({max_possible_pop:,.0f}). Constraint capped at Max."
            final_target_pop = max_possible_pop
        else:
            status_msg += f" Target set to manual value: {final_target_pop:,.0f}."

    m2 = pulp.LpProblem("Stage2", pulp.LpMinimize)
    y2 = pulp.LpVariable.dicts("Build2", range(num_candidates), cat='Binary')
    x2 = pulp.LpVariable.dicts("Assign2", (range(num_demand), range(num_candidates)), cat='Binary')
    z2 = pulp.LpVariable.dicts("Served2", range(num_demand), cat='Binary')
    
    m2 += pulp.lpSum(y2[j] * (costs_sd[j] + co2_penalty * emissions_sd[j]) for j in range(num_candidates))
    
    m2 += pulp.lpSum(y2[j] for j in range(num_candidates)) <= num_centers
    m2 += pulp.lpSum(populations[i] * z2[i] for i in range(num_demand)) >= final_target_pop
    
    for i in range(num_demand):
        m2 += pulp.lpSum(x2[i][j] for j in range(num_candidates)) == z2[i]
        for j in range(num_candidates):
            m2 += x2[i][j] <= y2[j]
            if valid_connection[i, j] == 0:
                m2 += x2[i][j] == 0
                
    m2.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[m2.status] == 'Optimal':
        # --- PROCESS RESULTS ---
        selected_indices = [j for j in range(num_candidates) if pulp.value(y2[j]) > 0.5]
        
        center_results = []
        map_data = []
        
        for dc_idx in selected_indices:
            dc_name = df_candidates.iloc[dc_idx]['City']
            assigned = [i for i in range(num_demand) if pulp.value(x2[i][dc_idx]) > 0.5]
            cluster_pop = sum(populations[i] for i in assigned)
            
            if cluster_pop > 0:
                center_results.append({
                    'Data Center': dc_name,
                    'State': df_candidates.iloc[dc_idx]['State'],
                    'GW Capacity': df_candidates.iloc[dc_idx]['GW'],
                    'Cost ($/kWh)': df_candidates.iloc[dc_idx]['Cost_Raw'], 
                    'CO2 (lb/MWh)': df_candidates.iloc[dc_idx]['CO2_Raw'],
                    'Served Population': cluster_pop,
                    'Cities Served': len(assigned),
                    'Lat': df_candidates.iloc[dc_idx]['Latitude'],
                    'Lon': df_candidates.iloc[dc_idx]['Longitude']
                })
                
                for i in assigned:
                    map_data.append({
                        'City': df_demand.iloc[i]['City'],
                        'Lat': df_demand.iloc[i]['Latitude'],
                        'Lon': df_demand.iloc[i]['Longitude'],
                        'Population': df_demand.iloc[i]['Population'],
                        'Assigned To': dc_name,
                        'Latency': lat_matrix[i, dc_idx]
                    })
                
        df_centers = pd.DataFrame(center_results)
        df_map = pd.DataFrame(map_data)
        
        return "Success", status_msg, df_centers, df_map, max_possible_pop
        
    else:
        return "Error", "Stage 2 Failed (Infeasible).", None, None, max_possible_pop

# --- UI LAYOUT ---

st.title("Data Center Network Optimizer")

if not os.path.exists(DEFAULT_FILE_PATH):
    st.error(f"File not found: `{DEFAULT_FILE_PATH}`")
    st.stop()

df = load_data(DEFAULT_FILE_PATH)

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("1. Physical Constraints")
    num_centers = st.slider("Max Number of Centers (k)", 1, 10, 5)
    gw_limit = st.number_input("Min Energy Capacity (GW)", 0.0, 10.0, 3.0, 0.5)
    max_latency = st.slider("Max Latency Limit (ms)", 5, 100, 25)
    
    st.header("2. Optimization Goals")
    co2_penalty = st.slider("CO2 Penalty Weight", 0.0, 10.0, 1.0, 0.1, help="Higher = Green is more important than Cheap")
    
    st.header("3. Population Target Strategy")
    # NEW: Toggle between modes
    target_mode = st.radio("Choose Target Method:", ["Percentage (Slack)", "Manual (Absolute)"])
    
    target_val = 0
    if target_mode == "Percentage (Slack)":
        target_val = st.slider(
            "Target % of Max Reachable", 
            min_value=0.1, max_value=1.0, value=0.95, step=0.05,
            help="Stage 1 finds the max theoretical population. Stage 2 will serve this percentage of it."
        )
        mode_code = 'Percent'
    else:
        total_avail_pop = int(df['Population'].sum())
        target_val = st.number_input(
            "Target Population Count", 
            min_value=1000000, max_value=total_avail_pop, value=100000000, step=1000000,
            help="Set a specific number of people to serve."
        )
        mode_code = 'Manual'
    
    run_btn = st.button("Run Optimization", type="primary")

# MAIN DISPLAY
if run_btn:
    with st.spinner("Running 2-Stage Optimization..."):
        status, msg, df_centers, df_map, max_pop = run_optimization(
            df, num_centers, gw_limit, max_latency, co2_penalty, mode_code, target_val
        )
    
    if status == "Error":
        st.error(msg)
    elif "Warning" in msg:
        st.warning(msg)
        st.caption(f"Stage 1 determined Max Reachable Population was: {max_pop:,.0f}")
    else:
        st.success(msg)
        st.caption(f"Stage 1 determined Max Reachable Population was: {max_pop:,.0f}")
        
    if df_centers is not None and not df_centers.empty:
        # --- METRICS ROW ---
        total_pop_served = df_centers['Served Population'].sum()
        
        # Weighted Averages for RAW metrics
        avg_cost_raw = np.average(df_centers['Cost ($/kWh)'], weights=df_centers['Served Population'])
        avg_co2_raw = np.average(df_centers['CO2 (lb/MWh)'], weights=df_centers['Served Population'])
        
        if not df_map.empty:
            avg_lat = np.average(df_map['Latency'], weights=df_map['Population'])
        else:
            avg_lat = 0
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pop Served", f"{total_pop_served:,.0f}")
        c2.metric("Avg Latency", f"{avg_lat:.2f} ms")
        c3.metric("Avg Cost", f"{avg_cost_raw:.2f} ¢/kWh")
        c4.metric("Avg CO2", f"{avg_co2_raw:.0f} lb/MWh")
        
        # --- MAP VISUALIZATION ---
        st.subheader("Network Map")
        
        if not df_map.empty:
            fig = px.scatter_geo(
                df_map, 
                lat='Lat', lon='Lon', 
                color='Assigned To',
                hover_name='City',
                hover_data={'Population': ':,.0f', 'Latency': ':.1f', 'Lat':False, 'Lon':False},
                scope='usa',
                title='Served Population Clusters'
            )
            
            fig.add_trace(go.Scattergeo(
                lon=df_centers['Lon'],
                lat=df_centers['Lat'],
                text=df_centers['Data Center'],
                mode='markers+text',
                marker=dict(size=25, color='black', symbol='star'),
                textposition="top center",
                name='Data Centers'
            ))
            
            fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # --- DATA TABLES ---
        st.subheader("Selected Data Centers (Detailed)")
        
        display_cols = ['Data Center', 'State', 'GW Capacity', 'Served Population', 'Cost ($/kWh)', 'CO2 (lb/MWh)']
        st.dataframe(df_centers[display_cols].style.format({
            'Served Population': '{:,.0f}',
            'Cost ($/kWh)': '{:.2f}',
            'CO2 (lb/MWh)': '{:.0f}',
            'GW Capacity': '{:.1f}'
        }), use_container_width=True)

else:
    st.info("Adjust settings in the sidebar and click 'Run Optimization'")