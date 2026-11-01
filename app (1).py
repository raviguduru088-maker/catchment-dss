"""
Smart Catchment Optimization: Rainwater Harvesting (RWH) & Runoff Management
Decision Support System (DSS) for Mylavaram Mandal, NTR District, Andhra Pradesh
Author: Senior Geospatial & Machine Learning Engineer
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Catchment DSS | Mylavaram Mandal",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive engineering appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DEFAULT HYDROLOGICAL & GEOSPATIAL DATASET
# Focused specifically on Mylavaram Mandal, NTR District, AP (~16.76° N, 80.64° E)
# ---------------------------------------------------------
@st.cache_data
def get_default_catchment_data() -> pd.DataFrame:
    """
    Returns baseline micro-catchment dataset for Mylavaram Mandal, NTR District, AP.
    Incorporates Hydrological Soil Groups (HSG A-D), Land Use, Slope, and Baseline CN.
    """
    data = [
        {"catchment_id": "MYL-C01", "name": "Mylavaram Town North", "lat": 16.7720, "lon": 80.6385, "area_ha": 340.5, "slope_pct": 2.1, "soil_group": "B", "soil_type": "Sandy Clay Loam", "land_use": "Built-up / Settlement"},
        {"catchment_id": "MYL-C02", "name": "Pulluru Agricultural Basin", "lat": 16.7915, "lon": 80.6120, "area_ha": 520.0, "slope_pct": 2.8, "soil_group": "B", "soil_type": "Alluvial Loam", "land_use": "Agricultural Land"},
        {"catchment_id": "MYL-C03", "name": "Chandragudem Foothills", "lat": 16.7450, "lon": 80.6110, "area_ha": 285.2, "slope_pct": 8.5, "soil_group": "C", "soil_type": "Red Gravelly Soil", "land_use": "Scrubland / Wasteland"},
        {"catchment_id": "MYL-C04", "name": "Velvadam Micro-Watershed", "lat": 16.7320, "lon": 80.6720, "area_ha": 410.8, "slope_pct": 3.4, "soil_group": "B", "soil_type": "Loamy Sand", "land_use": "Agricultural Land"},
        {"catchment_id": "MYL-C05", "name": "Morusumilli Drainage Basin", "lat": 16.8040, "lon": 80.6550, "area_ha": 375.0, "slope_pct": 1.9, "soil_group": "A", "soil_type": "Coarse Sand & Silt", "land_use": "Agricultural Land"},
        {"catchment_id": "MYL-C06", "name": "Kuntamukkala Valley", "lat": 16.7180, "lon": 80.6250, "area_ha": 490.4, "slope_pct": 6.2, "soil_group": "C", "soil_type": "Clay Loam", "land_use": "Scrubland / Wasteland"},
        {"catchment_id": "MYL-C07", "name": "Ponnavaram Ridge", "lat": 16.7590, "lon": 80.5890, "area_ha": 310.0, "slope_pct": 11.4, "soil_group": "D", "soil_type": "Dense Hardpan Clay", "land_use": "Barren / Rocky"},
        {"catchment_id": "MYL-C08", "name": "Dasullapalem Agro-Zone", "lat": 16.7860, "lon": 80.6740, "area_ha": 260.6, "slope_pct": 2.4, "soil_group": "B", "soil_type": "Medium Black Soil", "land_use": "Agricultural Land"},
        {"catchment_id": "MYL-C09", "name": "Sabjapadu Forest Periphery", "lat": 16.8220, "lon": 80.6180, "area_ha": 445.0, "slope_pct": 7.8, "soil_group": "B", "soil_type": "Sandy Loam", "land_use": "Forest / Plantation"},
        {"catchment_id": "MYL-C10", "name": "Tolukodu Stream Inflow", "lat": 16.7050, "lon": 80.6620, "area_ha": 395.2, "slope_pct": 4.1, "soil_group": "C", "soil_type": "Clay Loam", "land_use": "Agricultural Land"},
        {"catchment_id": "MYL-C11", "name": "Jangalapalle Hill Slope", "lat": 16.7410, "lon": 80.5650, "area_ha": 230.5, "slope_pct": 13.5, "soil_group": "D", "soil_type": "Rocky Shallow Soil", "land_use": "Barren / Rocky"},
        {"catchment_id": "MYL-C12", "name": "Thalluru Micro-Catchment", "lat": 16.7780, "lon": 80.5730, "area_ha": 315.0, "slope_pct": 5.0, "soil_group": "C", "soil_type": "Red Sandy Clay", "land_use": "Scrubland / Wasteland"},
        {"catchment_id": "MYL-C13", "name": "Chevuturu Lowland Plain", "lat": 16.8120, "lon": 80.6900, "area_ha": 580.0, "slope_pct": 1.5, "soil_group": "A", "soil_type": "Alluvial Silt Loam", "land_use": "Agricultural Land"},
        {"catchment_id": "MYL-C14", "name": "Badava Nala Corridor", "lat": 16.7630, "lon": 80.6480, "area_ha": 195.4, "slope_pct": 3.8, "soil_group": "C", "soil_type": "Silty Clay Loam", "land_use": "Scrubland / Wasteland"},
        {"catchment_id": "MYL-C15", "name": "Kondapalli RF Foothills Catchment", "lat": 16.6980, "lon": 80.5890, "area_ha": 620.0, "slope_pct": 9.2, "soil_group": "B", "soil_type": "Gravelly Loam", "land_use": "Forest / Plantation"},
        {"catchment_id": "MYL-C16", "name": "Mylavaram Market & Highway Hub", "lat": 16.7645, "lon": 80.6320, "area_ha": 180.0, "slope_pct": 1.8, "soil_group": "B", "soil_type": "Urban Compacted", "land_use": "Built-up / Settlement"},
        {"catchment_id": "MYL-C17", "name": "Keethanakonda Basin", "lat": 16.7290, "lon": 80.5980, "area_ha": 340.0, "slope_pct": 4.6, "soil_group": "B", "soil_type": "Loam", "land_use": "Agricultural Land"},
        {"catchment_id": "MYL-C18", "name": "Konda Parva Hill catchment", "lat": 16.7890, "lon": 80.5480, "area_ha": 290.0, "slope_pct": 14.2, "soil_group": "D", "soil_type": "Weathered Granite & Clay", "land_use": "Barren / Rocky"}
    ]
    return pd.DataFrame(data)


# ---------------------------------------------------------
# SCS-CN & MCDA HYDROLOGICAL ENGINE
# ---------------------------------------------------------
def get_curve_number(land_use: str, soil_group: str) -> int:
    """
    Standard USDA NRCS / Indian Watershed Curve Number (CN) matrix for AMC-II.
    """
    cn_matrix = {
        "Agricultural Land": {"A": 64, "B": 75, "C": 83, "D": 87},
        "Scrubland / Wasteland": {"A": 48, "B": 67, "C": 77, "D": 83},
        "Forest / Plantation": {"A": 36, "B": 60, "C": 73, "D": 79},
        "Built-up / Settlement": {"A": 77, "B": 85, "C": 90, "D": 92},
        "Barren / Rocky": {"A": 71, "B": 80, "C": 85, "D": 88}
    }
    land_entry = cn_matrix.get(land_use, {"A": 70, "B": 80, "C": 85, "D": 90})
    return land_entry.get(soil_group, 75)


def calculate_hydrology(df: pd.DataFrame, rainfall_mm: float) -> pd.DataFrame:
    """
    Calculates SCS-CN Direct Runoff and MCDA Rainwater Harvesting Suitability.
    S = (25400 / CN) - 254
    Ia = 0.2 * S
    Q = ((P - Ia)^2) / (P + 0.8 * S) when P > Ia, else 0.0
    Runoff Volume (m^3) = Q(mm)/1000 * Area(ha)*10000 = Q * Area * 10
    """
    df = df.copy()
    
    # Calculate CN for each catchment
    df['CN'] = df.apply(lambda row: get_curve_number(row['land_use'], row['soil_group']), axis=1)
    
    # Potential Maximum Retention S (mm)
    df['S_retention_mm'] = (25400.0 / df['CN']) - 254.0
    
    # Initial Abstraction Ia (mm)
    df['Ia_mm'] = 0.2 * df['S_retention_mm']
    
    # Runoff Depth Q (mm)
    def compute_q(row):
        p = rainfall_mm
        ia = row['Ia_mm']
        s = row['S_retention_mm']
        if p > ia:
            return ((p - ia) ** 2) / (p + (0.8 * s))
        return 0.0

    df['runoff_depth_mm'] = df.apply(compute_q, axis=1)
    
    # Runoff discharge volume in m3 (for given rainfall storm event/hour)
    df['runoff_volume_m3'] = df['runoff_depth_mm'] * df['area_ha'] * 10.0
    
    # ---------------------------------------------------------
    # MULTI-CRITERIA DECISION ANALYSIS (MCDA)
    # Suitability = 0.30*S_slope + 0.30*S_runoff + 0.20*S_soil + 0.20*S_landuse
    # ---------------------------------------------------------
    
    # Slope Suitability Score (0-100): gentle slopes (<3%) ideal for infiltration/ponds, moderate good for check dams
    def score_slope(slope):
        if slope <= 2.0:
            return 95.0
        elif slope <= 3.5:
            return 88.0
        elif slope <= 6.0:
            return 75.0
        elif slope <= 10.0:
            return 60.0
        else:
            return 35.0
            
    df['S_slope'] = df['slope_pct'].apply(score_slope)
    
    # Runoff Suitability Score (0-100): higher harvestable runoff depth gives higher potential yield
    max_q = max(df['runoff_depth_mm'].max(), 1.0)
    df['S_runoff'] = df['runoff_depth_mm'].apply(lambda q: min(100.0, max(15.0, (q / max_q) * 95.0)))
    
    # Soil Suitability Score (0-100): Infiltration capacity of HSG
    soil_scores = {"A": 95.0, "B": 82.0, "C": 60.0, "D": 40.0}
    df['S_soil'] = df['soil_group'].apply(lambda s: soil_scores.get(s, 65.0))
    
    # Land Use Suitability Score (0-100)
    lu_scores = {
        "Agricultural Land": 92.0,
        "Scrubland / Wasteland": 85.0,
        "Built-up / Settlement": 78.0,
        "Forest / Plantation": 60.0,
        "Barren / Rocky": 45.0
    }
    df['S_landuse'] = df['land_use'].apply(lambda l: lu_scores.get(l, 70.0))
    
    # Weighted Linear Combination MCDA
    df['suitability_score'] = (
        0.30 * df['S_slope'] +
        0.30 * df['S_runoff'] +
        0.20 * df['S_soil'] +
        0.20 * df['S_landuse']
    ).round(2)
    
    # Suitability Classification
    def classify_zone(score):
        if score >= 70.0:
            return "High"
        elif score >= 50.0:
            return "Medium"
        else:
            return "Low"
            
    df['suitability_class'] = df['suitability_score'].apply(classify_zone)
    
    # ---------------------------------------------------------
    # INTERVENTION RECOMMENDATION RULES
    # ---------------------------------------------------------
    def assign_intervention(row):
        lu = row['land_use']
        slope = row['slope_pct']
        soil = row['soil_group']
        
        if lu == "Built-up / Settlement":
            return "Rooftop RWH & Infiltration Recharge Trenches"
        elif slope <= 3.0:
            if soil in ["A", "B"]:
                return "Percolation Tank / Subsurface Recharge Shaft"
            else:
                return "Farm Pond / Surface Storage Reservoir"
        elif slope > 3.0 and slope <= 8.0:
            return "Masonry Check Dam / Earthen Nala Bund"
        else: # Steeper slope (> 8%)
            if lu == "Barren / Rocky":
                return "Continuous Contour Trenches (CCT) & Boulder Gully Plugs"
            return "Gabion Check Dam & Loose Stone Benches"
            
    df['recommended_intervention'] = df.apply(assign_intervention, axis=1)
    return df


# ---------------------------------------------------------
# MACHINE LEARNING ENGINE: SYNTHETIC FLOOD RISK MODEL
# ---------------------------------------------------------
@st.cache_resource
def train_flood_risk_model():
    """
    Trains a robust Random Forest Classifier on synthetic catchment flood risk dynamics
    based on hydrologic factors (Slope, Area, Soil Group, Land Use, Rainfall, Runoff Depth).
    """
    np.random.seed(42)
    n_samples = 1200
    
    slopes = np.random.uniform(0.8, 16.0, n_samples)
    areas = np.random.uniform(80.0, 750.0, n_samples)
    soils = np.random.choice(["A", "B", "C", "D"], size=n_samples, p=[0.15, 0.40, 0.30, 0.15])
    landuses = np.random.choice(
        ["Agricultural Land", "Scrubland / Wasteland", "Forest / Plantation", "Built-up / Settlement", "Barren / Rocky"],
        size=n_samples,
        p=[0.45, 0.20, 0.15, 0.12, 0.08]
    )
    rainfalls = np.random.uniform(20.0, 120.0, n_samples)
    
    # Calculate hydrological variables for training set
    records = []
    for sl, ar, so, lu, rf in zip(slopes, areas, soils, landuses, rainfalls):
        cn = get_curve_number(lu, so)
        s_ret = (25400.0 / cn) - 254.0
        ia = 0.2 * s_ret
        q = ((rf - ia) ** 2) / (rf + 0.8 * s_ret) if rf > ia else 0.0
        
        # Synthetic risk calculation logic
        # High rainfall + high CN/compact soil + high slope or extreme flat bowl -> higher flood risk
        risk_metric = (q * 1.5) + (ar / 100.0) - (sl * 1.2 if sl > 8 else -sl)
        if so == "D":
            risk_metric += 15.0
        elif so == "C":
            risk_metric += 8.0
        if lu == "Built-up / Settlement":
            risk_metric += 22.0
            
        if risk_metric > 75:
            risk = "High"
        elif risk_metric > 42:
            risk = "Medium"
        else:
            risk = "Low"
            
        records.append({
            "slope_pct": sl,
            "area_ha": ar,
            "soil_group": so,
            "land_use": lu,
            "rainfall_mm": rf,
            "runoff_depth_mm": q,
            "flood_risk": risk
        })
        
    synth_df = pd.DataFrame(records)
    
    le_soil = LabelEncoder()
    le_landuse = LabelEncoder()
    
    synth_df['soil_encoded'] = le_soil.fit_transform(synth_df['soil_group'])
    synth_df['landuse_encoded'] = le_landuse.fit_transform(synth_df['land_use'])
    
    features = ['slope_pct', 'area_ha', 'soil_encoded', 'landuse_encoded', 'rainfall_mm', 'runoff_depth_mm']
    X = synth_df[features]
    y = synth_df['flood_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_clf.fit(X_train, y_train)
    acc = rf_clf.score(X_test, y_test)
    
    return rf_clf, le_soil, le_landuse, acc, features


# ---------------------------------------------------------
# SIDEBAR CONTROLS & DATA INGESTION
# ---------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1534274988757-a28bf1a57c17?auto=format&fit=crop&w=400&q=80", caption="NTR District Hydrology Monitoring")
st.sidebar.header("🕹️ Simulation Controls")

rainfall_input = st.sidebar.slider(
    "🌧️ Storm Rainfall Intensity (P in mm/hr)",
    min_value=10.0,
    max_value=150.0,
    value=55.0,
    step=2.5,
    help="Design hourly rainfall intensity used for SCS-CN runoff calculation."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Catchment Data Source")
data_source = st.sidebar.radio(
    "Select Source",
    ["Default Mylavaram Mandal Baseline (18 Sites)", "Upload Custom Catchment CSV"],
    index=0
)

custom_file = None
if data_source == "Upload Custom Catchment CSV":
    custom_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Required columns: catchment_id, name, lat, lon, area_ha, slope_pct, soil_group (A/B/C/D), land_use"
    )

# Load data based on selection
if custom_file is not None:
    try:
        raw_df = pd.read_csv(custom_file)
        st.sidebar.success(f"Loaded {len(raw_df)} custom catchment records.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        raw_df = get_default_catchment_data()
else:
    raw_df = get_default_catchment_data()

# Execute Hydrology & MCDA engine
df_processed = calculate_hydrology(raw_df, rainfall_input)

# Train or retrieve Random Forest model
rf_model, le_soil, le_landuse, model_accuracy, ml_feature_names = train_flood_risk_model()


# ---------------------------------------------------------
# MAIN DASHBOARD HEADER & TOP SUMMARY KPI CARDS
# ---------------------------------------------------------
st.markdown('<div class="main-title">🌊 Smart Catchment Optimization DSS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Rainwater Harvesting (RWH) & Runoff Management Decision Support System | '
    '<strong>Mylavaram Mandal, NTR District, Andhra Pradesh (16.76° N, 80.64° E)</strong></div>',
    unsafe_allow_html=True
)

total_area = df_processed['area_ha'].sum()
avg_rainfall = rainfall_input
total_runoff_vol = df_processed['runoff_volume_m3'].sum()
high_suit_count = (df_processed['suitability_class'] == "High").sum()
med_suit_count = (df_processed['suitability_class'] == "Medium").sum()
total_recommended = high_suit_count + med_suit_count

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Total Study Area",
        value=f"{total_area:,.1f} ha",
        help="Total catchment watershed area analyzed across Mylavaram Mandal"
    )

with kpi2:
    st.metric(
        label="Average Rainfall (P)",
        value=f"{avg_rainfall:.1f} mm/hr",
        help="Simulated storm event precipitation depth"
    )

with kpi3:
    st.metric(
        label="Estimated Runoff Discharge",
        value=f"{total_runoff_vol:,.0f} m³/hr",
        delta=f"Avg Q: {df_processed['runoff_depth_mm'].mean():.1f} mm",
        help="Aggregated surface runoff volume generated across catchments"
    )

with kpi4:
    st.metric(
        label="Priority RWH Sites",
        value=f"{total_recommended} Sites",
        delta=f"{high_suit_count} High Suitability",
        delta_color="normal",
        help="High and Medium priority sites eligible for engineered RWH interventions"
    )

st.markdown("---")

# ---------------------------------------------------------
# TABBED PRESENTATION SYSTEM (Tabs 1 to 6)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 Interactive Geospatial Map",
    "📐 SCS-CN & MCDA Analysis",
    "📊 Slope & Soil Distribution",
    "🎯 Priority Interventions Table",
    "📥 Downloadable Reports",
    "🤖 ML Real-Time Prediction"
])

# =========================================================
# TAB 1: INTERACTIVE GEOSPATIAL MAP (FOLIUM)
# =========================================================
with tab1:
    st.subheader("Geospatial Suitability & Engineering Intervention Map")
    st.write(
        "Interactive OpenStreetMap centered on Mylavaram Mandal (16.76° N, 80.64° E). "
        "Markers are color-coded by MCDA Suitability: 🟢 **High (≥70)**, 🟠 **Medium (50–69)**, 🔴 **Low (<50)**."
    )
    
    # Map controls
    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        selected_classes = st.multiselect(
            "Filter by Suitability Class:",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    
    filtered_map_df = df_processed[df_processed['suitability_class'].isin(selected_classes)]
    
    # Initialize Folium Map centered on Mylavaram Mandal
    m = folium.Map(
        location=[16.7600, 80.6350],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True
    )
    
    # Color mapping
    color_map = {
        "High": "#10B981",    # Emerald Green
        "Medium": "#F59E0B",  # Amber Orange
        "Low": "#EF4444"      # Crimson Red
    }
    
    # Plot markers
    for _, row in filtered_map_df.iterrows():
        color = color_map.get(row['suitability_class'], "#3B82F6")
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; width: 260px;">
            <h4 style="margin: 0 0 6px 0; color: #1E3A8A;">{row['name']} ({row['catchment_id']})</h4>
            <hr style="margin: 4px 0; border: 0.5px solid #CBD5E1;"/>
            <b>Suitability Score:</b> <span style="color: {color}; font-weight: bold;">{row['suitability_score']}/100 ({row['suitability_class']})</span><br/>
            <b>Land Use:</b> {row['land_use']}<br/>
            <b>Hydrologic Soil Group:</b> {row['soil_group']} ({row['soil_type']})<br/>
            <b>Slope:</b> {row['slope_pct']}%<br/>
            <b>SCS Curve Number:</b> {row['CN']}<br/>
            <b>Runoff Depth:</b> {row['runoff_depth_mm']:.1f} mm ({row['runoff_volume_m3']:,.0f} m³)<br/>
            <div style="margin-top: 8px; padding: 6px; background: #F1F5F9; border-radius: 4px; border-left: 3px solid {color};">
                <b>Recommended Intervention:</b><br/>
                <span style="color: #0F172A; font-weight: 600;">{row['recommended_intervention']}</span>
            </div>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['name']} | Suitability: {row['suitability_score']} ({row['suitability_class']})",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2
        ).add_to(m)
        
    st_folium(m, width="100%", height=520)


# =========================================================
# TAB 2: SCS-CN & MCDA HYDROLOGICAL ANALYSIS
# =========================================================
with tab2:
    st.subheader("Hydrological Formulation & Mathematical Framework")
    
    st.markdown(r"""
    The **USDA Soil Conservation Service Curve Number (SCS-CN)** method is implemented for surface runoff estimation, 
    coupled with **Multi-Criteria Decision Analysis (MCDA)** for optimal rainwater harvesting site selection:
    """)
    
    eq_col1, eq_col2 = st.columns(2)
    with eq_col1:
        st.markdown(r"""
        ##### 1. SCS-CN Runoff Equations
        * **Potential Maximum Retention ($S$):**
          $$S = \frac{25400}{CN} - 254 \quad (\text{mm})$$
        * **Initial Abstraction ($I_a$):**
          $$I_a = 0.2 \cdot S \quad (\text{mm})$$
        * **Direct Runoff Depth ($Q$):**
          $$Q = \begin{cases} \frac{(P - I_a)^2}{P + 0.8S} & \text{if } P > I_a \\ 0 & \text{otherwise} \end{cases}$$
        * **Runoff Volume ($V$):**
          $$V = Q \times \text{Area (ha)} \times 10 \quad (\text{m}^3)$$
        """)
        
    with eq_col2:
        st.markdown(r"""
        ##### 2. MCDA Weighted Linear Combination
        $$\text{Suitability} = 0.30 \cdot S_{slope} + 0.30 \cdot S_{runoff} + 0.20 \cdot S_{soil} + 0.20 \cdot S_{landuse}$$
        * **High Suitability Zone:** $\ge 70$ (Optimal for major RWH investment)
        * **Medium Suitability Zone:** $50 \text{ to } 69$ (Feasible with site conditioning)
        * **Low Suitability Zone:** $< 50$ (Steep runoff loss or poor infiltration)
        """)
        
    st.markdown("---")
    st.subheader("Rainfall vs Runoff Discharge Scatter Analysis")
    
    # Plotly Scatter Plot: Rainfall vs Runoff
    fig_scatter = px.scatter(
        df_processed,
        x="runoff_depth_mm",
        y="runoff_volume_m3",
        size="area_ha",
        color="suitability_class",
        hover_name="name",
        hover_data=["catchment_id", "CN", "slope_pct", "suitability_score", "recommended_intervention"],
        color_discrete_map={"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"},
        labels={
            "runoff_depth_mm": "Direct Runoff Depth Q (mm)",
            "runoff_volume_m3": "Discharge Volume (m³/hr)",
            "area_ha": "Catchment Area (ha)",
            "suitability_class": "Suitability Class"
        },
        title=f"Runoff Depth vs Runoff Volume at P = {rainfall_input} mm/hr (Bubble Size = Catchment Area)"
    )
    fig_scatter.update_layout(height=480, template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================
# TAB 3: SLOPE & SOIL DISTRIBUTION ANALYSIS
# =========================================================
with tab3:
    st.subheader("Catchment Geomorphology: Slope & Soil Distributions")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Box plot of Slope % across Land Use
        fig_box = px.box(
            df_processed,
            x="land_use",
            y="slope_pct",
            color="land_use",
            points="all",
            labels={"land_use": "Land Use Class", "slope_pct": "Slope Percentage (%)"},
            title="Slope (%) Distribution across Land Use Classes"
        )
        fig_box.update_layout(showlegend=False, height=450, template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_chart2:
        # Bar chart comparing Soil Group counts & Mean Runoff Depth
        soil_agg = df_processed.groupby("soil_group").agg(
            count=("catchment_id", "count"),
            avg_runoff=("runoff_depth_mm", "mean"),
            avg_suitability=("suitability_score", "mean")
        ).reset_index()
        
        fig_bar = px.bar(
            soil_agg,
            x="soil_group",
            y="avg_runoff",
            color="soil_group",
            text="avg_runoff",
            labels={
                "soil_group": "Hydrologic Soil Group (HSG)",
                "avg_runoff": "Mean Runoff Depth (mm)"
            },
            title="Mean Runoff Depth by Hydrologic Soil Group (HSG)"
        )
        fig_bar.update_traces(texttemplate='%{text:.1f} mm', textposition='outside')
        fig_bar.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)


# =========================================================
# TAB 4: PRIORITY INTERVENTIONS TABLE
# =========================================================
with tab4:
    st.subheader("Priority Engineering Interventions Matrix")
    st.write("Ranked dataframe of high and moderate suitability sites detailing catchment ID, land use, runoff, slope, score, and recommended structures.")
    
    # Priority sorting: High and Medium first, ordered by score descending
    priority_df = df_processed[df_processed['suitability_class'].isin(["High", "Medium"])].sort_values(
        by="suitability_score", ascending=False
    )
    
    display_cols = [
        "catchment_id", "name", "land_use", "soil_group", "slope_pct", 
        "CN", "runoff_depth_mm", "runoff_volume_m3", "suitability_score", 
        "suitability_class", "recommended_intervention"
    ]
    
    st.dataframe(
        priority_df[display_cols].rename(columns={
            "catchment_id": "Catchment ID",
            "name": "Micro-Catchment Name",
            "land_use": "Land Use",
            "soil_group": "HSG",
            "slope_pct": "Slope (%)",
            "runoff_depth_mm": "Runoff Q (mm)",
            "runoff_volume_m3": "Volume (m³)",
            "suitability_score": "Suitability Score",
            "suitability_class": "Class",
            "recommended_intervention": "Recommended Structure"
        }),
        use_container_width=True,
        height=400
    )


# =========================================================
# TAB 5: DOWNLOADABLE REPORTS
# =========================================================
with tab5:
    st.subheader("Executive Reports & Export Center")
    st.write("Download certified DSS planning reports for the NTR District Ground Water & Irrigation authorities.")
    
    export_df = df_processed[[
        "catchment_id", "name", "lat", "lon", "area_ha", "slope_pct", 
        "soil_group", "soil_type", "land_use", "CN", "runoff_depth_mm", 
        "runoff_volume_m3", "suitability_score", "suitability_class", 
        "recommended_intervention"
    ]].copy()
    
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    st.download_button(
        label="📥 Download Recommended Sites Report (CSV)",
        data=csv_bytes,
        file_name=f"Mylavaram_RWH_Catchment_Report_P{int(rainfall_input)}mm.csv",
        mime="text/csv",
        help="One-click CSV download containing complete geospatial and hydrological findings."
    )
    
    st.markdown("#### Sample Executive Summary Preview")
    st.text_area(
        label="Report Summary",
        value=f"""GOVERNMENT OF ANDHRA PRADESH | WATER RESOURCES & PR DEPARTMENT
DISTRICT: NTR DISTRICT | MANDAL: MYLAVARAM
DECISION SUPPORT SYSTEM (DSS) - SMART CATCHMENT OPTIMIZATION
================================================================================
Design Rainfall Intensity: {rainfall_input:.1f} mm/hr
Total Micro-Catchments Evaluated: {len(df_processed)}
Total Geographical Area: {total_area:,.1f} Hectares
Total Peak Runoff Discharge: {total_runoff_vol:,.0f} m³/hr
High Suitability Priority Interventions Identified: {high_suit_count}
Medium Suitability Priority Interventions: {med_suit_count}

Top 3 High Priority Sites:
{priority_df[['catchment_id', 'name', 'suitability_score', 'recommended_intervention']].head(3).to_string(index=False)}
================================================================================
Generated via Smart Catchment Geospatial & ML Decision Support System
""",
        height=260
    )


# =========================================================
# TAB 6: ML REAL-TIME FLOOD RISK PREDICTION
# =========================================================
with tab6:
    st.subheader("Machine Learning: Real-Time Catchment Flood Risk Prediction")
    st.write(
        f"Embedded **Random Forest Classifier** trained on basin hydrological parameters "
        f"(Model Test Accuracy: **{model_accuracy*100:.1f}%**). Enter custom catchment characteristics below:"
    )
    
    with st.form("ml_prediction_form"):
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            inp_slope = st.slider("Catchment Slope (%)", min_value=0.5, max_value=25.0, value=3.5, step=0.5)
            inp_area = st.number_input("Catchment Area (ha)", min_value=10.0, max_value=2000.0, value=250.0, step=10.0)
            
        with pred_col2:
            inp_soil = st.selectbox("Hydrologic Soil Group (HSG)", ["A", "B", "C", "D"], index=1)
            inp_landuse = st.selectbox(
                "Land Use Type",
                ["Agricultural Land", "Scrubland / Wasteland", "Forest / Plantation", "Built-up / Settlement", "Barren / Rocky"],
                index=0
            )
            
        with pred_col3:
            inp_rain = st.slider("Simulated Storm Rainfall (mm/hr)", min_value=10.0, max_value=150.0, value=65.0, step=5.0)
            submit_btn = st.form_submit_button("⚡ Predict Flood Risk & Mitigation Strategy")
            
    if submit_btn:
        # Calculate instant hydrological parameters
        cn_val = get_curve_number(inp_landuse, inp_soil)
        s_val = (25400.0 / cn_val) - 254.0
        ia_val = 0.2 * s_val
        q_val = ((inp_rain - ia_val) ** 2) / (inp_rain + 0.8 * s_val) if inp_rain > ia_val else 0.0
        
        soil_code = le_soil.transform([inp_soil])[0]
        lu_code = le_landuse.transform([inp_landuse])[0]
        
        features_input = np.array([[inp_slope, inp_area, soil_code, lu_code, inp_rain, q_val]])
        predicted_risk = rf_model.predict(features_input)[0]
        prob_distribution = rf_model.predict_proba(features_input)[0]
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            risk_badge_colors = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
            st.markdown(f"### Predicted Risk: {risk_badge_colors.get(predicted_risk, '')} **{predicted_risk} Risk**")
            
            # Probability bar chart
            prob_df = pd.DataFrame({
                "Risk Class": rf_model.classes_,
                "Confidence Probability": prob_distribution
            })
            fig_prob = px.bar(
                prob_df,
                x="Risk Class",
                y="Confidence Probability",
                color="Risk Class",
                color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
                range_y=[0, 1.0]
            )
            fig_prob.update_layout(height=260, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with res_col2:
            st.markdown("#### Hydrological Diagnostics & Engineering Action Plan")
            st.info(f"""
            - **Estimated SCS Curve Number (CN):** {cn_val}
            - **Initial Abstraction ($I_a$):** {ia_val:.2f} mm
            - **Potential Direct Runoff Depth ($Q$):** {q_val:.2f} mm
            - **Total Runoff Yield:** {(q_val * inp_area * 10):,.0f} m³
            """)
            
            if predicted_risk == "High":
                st.error("""
                ⚠️ **Mitigation Strategy: High Runoff Velocity & Flood Hazard**
                - Construct cascading series of Masonry Check Dams with energy dissipators.
                - Implement deep Continuous Contour Trenches (CCT) upstream to retard overland velocity.
                - Evacuation and diversion bunds towards regional percolation reservoirs.
                """)
            elif predicted_risk == "Medium":
                st.warning("""
                🟠 **Mitigation Strategy: Moderate Inflow & Harvesting Potential**
                - Construct Earthen Nala Bunds and Recharge Shafts.
                - Establish Farm Ponds with surplus weir spillways for agricultural water security.
                """)
            else:
                st.success("""
                🟢 **Mitigation Strategy: Stable Low-Risk Infiltration Zone**
                - Ideal for Subsurface Infiltration Trenches and Dug-well Recharge Systems.
                - Minimal flood risk under current rainfall intensity.
                """)

st.markdown("---")
st.caption("Mylavaram Mandal Catchment Decision Support System | Developed for NTR District Water Resources Management")
