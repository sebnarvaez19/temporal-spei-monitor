import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import datetime

# Add virtual environment site-packages if not active
venv_site_packages = os.path.abspath(r"venv\Lib\site-packages")
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

from data_pipeline import run_pipeline, SIMPLIFIED_GEOJSON_PATH
from forecaster import forecast_pet_for_basins, CACHED_TC_PATH, COMPLETED_PET_PATH
from spei_calculator import calculate_spei_all_basins, CACHED_CHIRPS_PATH, SPEI_RESULTS_PATH

# Set page configuration
st.set_page_config(
    page_title="Monitor de Sequía - Atlántico",
    page_icon="public/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Styling headers */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1e293b;
        font-weight: 700;
    }
    
    /* Card Container */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #3b82f6;
        margin-bottom: 15px;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .card-title {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }
    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .card-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 5px;
    }
    
    /* Colors for classifications */
    .wet-extreme { border-left-color: #0f172a; }
    .wet-very { border-left-color: #1e3a8a; }
    .wet-mod { border-left-color: #2563eb; }
    .normal { border-left-color: #10b981; }
    .dry-mod { border-left-color: #f59e0b; }
    .dry-severe { border-left-color: #ea580c; }
    .dry-extreme { border-left-color: #dc2626; }
</style>
""", unsafe_allow_html=True)

# Helper to classify SPEI values
def get_spei_class(val):
    if pd.isna(val):
        return "Sin Datos", "normal", "#cbd5e1"
    if val >= 2.0:
        return "Extremadamente Húmedo", "wet-extreme", "#0f172a"
    elif val >= 1.5:
        return "Muy Húmedo", "wet-very", "#1e3a8a"
    elif val >= 1.0:
        return "Moderadamente Húmedo", "wet-mod", "#2563eb"
    elif val > -1.0:
        return "Normal", "normal", "#10b981"
    elif val > -1.5:
        return "Moderadamente Seco", "dry-mod", "#f59e0b"
    elif val > -2.0:
        return "Severamente Seco", "dry-severe", "#ea580c"
    else:
        return "Extremadamente Seco", "dry-extreme", "#dc2626"

# Title Section
col_logo, col_title = st.columns([1, 12])
with col_logo:
    st.image("public/logo.png", width=110)
with col_title:
    st.markdown("""
        <div style="padding-top: 12px;">
            <h1 style="margin:0; font-family:'Outfit', sans-serif; font-size:32px; line-height: 1.1;">Monitor de Sequía (SPEI) - Departamento del Atlántico</h1>
            <p style="margin:5px 0 0 0; font-size:16px; color:#475569; line-height: 1.3;">Visualización interactiva y balance hidrológico del Departamento del Atlántico, Colombia. Proyecto desarrollado para la Corporación Autónoma Regional del Atlántico (CRA).</p>
        </div>
    """, unsafe_allow_html=True)
st.write("") # Spacer

# Load datasets
@st.cache_data
def load_all_data():
    if not os.path.exists(SPEI_RESULTS_PATH):
        # Run pipelines to generate files if missing
        run_pipeline()
        forecast_pet_for_basins(CACHED_TC_PATH, COMPLETED_PET_PATH)
        calculate_spei_all_basins(CACHED_CHIRPS_PATH, COMPLETED_PET_PATH, SPEI_RESULTS_PATH)
        
    df_results = pd.read_csv(SPEI_RESULTS_PATH, parse_dates=["month"])
    gdf_basins = gpd.read_file(SIMPLIFIED_GEOJSON_PATH)
    return df_results, gdf_basins

try:
    df_results, gdf_basins = load_all_data()
except Exception as e:
    st.error(f"Error cargando los datos del sistema: {e}")
    st.info("Por favor presione 'Sincronizar Datos' en la barra lateral para volver a ejecutar el pipeline completo.")
    df_results, gdf_basins = pd.DataFrame(), None

# Initialize session state for selected basin
if "selected_basin" not in st.session_state:
    st.session_state["selected_basin"] = "Rio Magdalena"

# Sidebar Configuration
st.sidebar.header("⚙️ Controles e Instrumentos")

st.sidebar.write("### Capa Base del Mapa")
map_style = st.sidebar.radio(
    "Seleccionar Capa Base:",
    options=["Esri World Imagery", "Jawg Lagoon"],
    index=0
)

if not df_results.empty:
    # 1. Scale selection
    scale = st.sidebar.selectbox(
        "Escala Temporal del SPEI",
        options=[1, 3, 6, 12],
        index=1,
        format_func=lambda x: f"SPEI-{x} ({x} {'mes' if x==1 else 'meses'})"
    )
    
    # 2. Year/Month Selection
    available_months = sorted(df_results["month"].unique(), reverse=True)
    available_months_str = [pd.to_datetime(m).strftime('%Y-%m') for m in available_months]
    
    selected_month_str = st.sidebar.selectbox(
        "Mes de Visualización",
        options=available_months_str,
        index=0 # Default to latest month
    )
    selected_month = pd.to_datetime(selected_month_str)
    
    # 3. Sync Data Button
    st.sidebar.markdown("---")
    st.sidebar.subheader("Actualizar Monitoreo")
    st.sidebar.write("Descarga los datos más recientes de CHIRPS, proyecta el Evapotranspiración (PET) actual con ARIMAX y calcula el SPEI.")
    
    if st.sidebar.button("🔄 Sincronizar Datos"):
        with st.spinner("Ejecutando pipeline completo... Esto tomará un par de minutos"):
            try:
                # Force redownload in pipeline
                run_pipeline(force_redownload=True)
                forecast_pet_for_basins(CACHED_TC_PATH, COMPLETED_PET_PATH)
                calculate_spei_all_basins(CACHED_CHIRPS_PATH, COMPLETED_PET_PATH, SPEI_RESULTS_PATH)
                st.cache_data.clear()
                st.success("¡Datos actualizados exitosamente!")
                st.rerun()
            except Exception as ex:
                st.error(f"Error durante la sincronización: {ex}")

    # Layout Setup
    col_map, col_details = st.columns([1.1, 0.9])
    
    # Filtering data for the selected month and scale
    df_month = df_results[df_results["month"] == selected_month].copy()
    spei_col = f"spei_{scale}"
    
    # 1. Map Section
    with col_map:
        st.subheader(f"🗺️ Clasificación de Sequía - {selected_month.strftime('%B %Y')}")
        
        # Merge GeoJSON with results
        gdf_month = gdf_basins.merge(df_month, on="basin", how="left")
        
        # Convert Timestamp column to string so it is JSON serializable for Folium
        if "month" in gdf_month.columns:
            gdf_month["month"] = gdf_month["month"].dt.strftime("%Y-%m")
        
        # Center coordinates for Atlántico
        map_center = [10.65, -74.95]
        zoom_start = 9.5
        
        # Configure map based on layer selection
        if map_style == "Jawg Lagoon":
            jawg_token = "XnKSEzQMZxWKbrdxePWWk36HSJhZQyFVDuY1tMX5oDQ9LSpRUnfmW41sSLVJoC5h"
            tile_url = f"https://tile.jawg.io/jawg-lagoon/{{z}}/{{x}}/{{y}}{{r}}.png?access-token={jawg_token}"
            attribution = '&copy; <a href="https://www.jawg.io" target="_blank">Jawg</a> &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
            m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=tile_url, attr=attribution)
        else:
            esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            esri_attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=esri_url, attr=esri_attr)
        
        # Define style function
        def style_fn(feature):
            basin_name = feature['properties']['basin']
            is_selected = (basin_name == st.session_state["selected_basin"])
            
            val = feature['properties'].get(spei_col, np.nan)
            _, _, fill_color = get_spei_class(val)
            
            return {
                'fillColor': fill_color,
                'color': '#000000' if is_selected else '#64748b',
                'weight': 3.0 if is_selected else 1.2,
                'fillOpacity': 0.8 if is_selected else 0.55
            }
            
        def highlight_fn(feature):
            return {
                'weight': 3.5,
                'color': '#0f172a',
                'fillOpacity': 0.9
            }
            
        # Add GeoJSON layer to map
        folium.GeoJson(
            gdf_month,
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=['basin', spei_col],
                aliases=['Cuenca:', 'SPEI:'],
                localize=True
            )
        ).add_to(m)
        
        # Display Folium map
        map_data = st_folium(m, height=450, width=None, use_container_width=True, key="folium_map")
        
        # Handle click selection
        if map_data and map_data.get("last_active_drawing"):
            clicked_basin = map_data["last_active_drawing"]["properties"]["basin"]
            if clicked_basin != st.session_state["selected_basin"]:
                st.session_state["selected_basin"] = clicked_basin
                st.rerun()
                
    # 2. Details Section
    with col_details:
        st.subheader(f"📊 Detalle: {st.session_state['selected_basin']}")
        
        # Calculate metric parameters (area in sq m, perimeter in m) by reprojecting to EPSG:9377 (Colombia Origen Nacional)
        gdf_proj = gdf_basins.to_crs(epsg=9377)
        basin_row_proj = gdf_proj[gdf_proj["basin"] == st.session_state["selected_basin"]].iloc[0]
        area_sq_m = basin_row_proj.geometry.area
        perimeter_m = basin_row_proj.geometry.length
        
        # Show parameters above the card
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <div style="font-size: 13px; color: #64748b; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Parámetros Geométricos</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                <span style="color: #475569; font-weight: 500;">Área Total:</span>
                <span style="color: #0f172a; font-weight: 700;">{area_sq_m:,.1f} m²</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 14px;">
                <span style="color: #475569; font-weight: 500;">Perímetro:</span>
                <span style="color: #0f172a; font-weight: 700;">{perimeter_m:,.1f} m</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Metric card for the selected basin
        row = df_month[df_month["basin"] == st.session_state["selected_basin"]]
        if not row.empty:
            row = row.iloc[0]
            spei_val = row[spei_col]
            precip_val = row["precip"]
            pet_val = row["pet"]
            
            label, css_class, _ = get_spei_class(spei_val)
            val_str = f"{spei_val:.2f}" if not pd.isna(spei_val) else "N/A"
            
            st.markdown(f"""
            <div class="metric-card {css_class}" style="padding: 25px;">
                <div class="card-title">Valor del {spei_col.upper()}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 15px;">
                    <div class="card-value" style="font-size: 36px;">{val_str}</div>
                    <div style="font-weight: 600; font-size: 16px; padding: 6px 14px; border-radius: 20px; background: #f1f5f9; color: #0f172a;">{label}</div>
                </div>
                <div style="border-top: 1px solid #e2e8f0; padding-top: 12px; margin-top: 10px; font-size: 14px; color: #475569;">
                    <span style="font-weight: 600; color: #2563eb;">🌧️ Lluvia:</span> {precip_val:.1f} mm <br>
                    <span style="font-weight: 600; color: #ef4444;">🌡️ Evapotranspiración (PET):</span> {pet_val:.1f} mm
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No hay datos para la cuenca seleccionada en este mes.")
            
    # Basin Detail Plot
    st.markdown("---")
    st.header("📈 Gráficos Históricos y Balance Hidrológico")
    
    # Define tabs
    tab1, tab2 = st.tabs(["📉 Evolución Histórica del SPEI", "📊 Balance Lluvia vs Evapotranspiración (PET)"])
    
    # Filter historical dataset for the selected basin
    df_basin = df_results[df_results["basin"] == st.session_state["selected_basin"]].sort_values("month").copy()
    
    with tab1:
        st.subheader(f"Evolución Histórica de {scale} meses - {st.session_state['selected_basin']}")
        
        # Plotly figure for SPEI
        fig_spei = go.Figure()
        
        # Add historical SPEI line
        fig_spei.add_trace(go.Scatter(
            x=df_basin["month"],
            y=df_basin[spei_col],
            mode="lines",
            line=dict(color="#2563eb", width=2.2),
            name=f"SPEI-{scale}",
            hovertemplate="Fecha: %{x|%B %Y}<br>SPEI: %{y:.2f}<extra></extra>"
        ))
        
        # Add reference lines
        fig_spei.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="solid"))
        fig_spei.add_hline(y=-1.0, line=dict(color="#f59e0b", width=1, dash="dash"), 
                           annotation_text="Seco Moderado (-1.0)", annotation_position="bottom left")
        fig_spei.add_hline(y=-1.5, line=dict(color="#ea580c", width=1, dash="dash"), 
                           annotation_text="Seco Severo (-1.5)", annotation_position="bottom left")
        fig_spei.add_hline(y=-2.0, line=dict(color="#dc2626", width=1, dash="dash"), 
                           annotation_text="Seco Extremo (-2.0)", annotation_position="bottom left")
        
        # Highlight selected month
        if selected_month in df_basin["month"].values:
            sel_row = df_basin[df_basin["month"] == selected_month]
            sel_val = sel_row[spei_col].values[0]
            fig_spei.add_trace(go.Scatter(
                x=[selected_month],
                y=[sel_val],
                mode="markers",
                marker=dict(color="#dc2626", size=10, line=dict(color="white", width=1.5)),
                name="Mes Seleccionado",
                hovertemplate="Fecha: %{x|%B %Y}<br>SPEI: %{y:.2f} (Seleccionado)<extra></extra>"
            ))
            
        fig_spei.update_layout(
            yaxis=dict(range=[-3.2, 3.2], gridcolor="#f1f5f9", title="Valor del SPEI"),
            xaxis=dict(gridcolor="#f1f5f9", title="Año"),
            template="plotly_white",
            margin=dict(l=40, r=40, t=20, b=40),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_spei, use_container_width=True)
        
    with tab2:
        st.subheader(f"Balance de Precipitación vs PET - {st.session_state['selected_basin']}")
        
        # Show last 5 years for clearer visualization of the balance
        df_basin_recent = df_basin[df_basin["month"] >= (selected_month - pd.DateOffset(years=5))].copy()
        
        # Plotly figure for balance
        fig_balance = go.Figure()
        
        # Add Precipitation line
        fig_balance.add_trace(go.Scatter(
            x=df_basin_recent["month"],
            y=df_basin_recent["precip"],
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="Precipitación (Lluvia)",
            hovertemplate="Fecha: %{x|%B %Y}<br>Precipitación: %{y:.1f} mm<extra></extra>"
        ))
        
        # Add PET line
        fig_balance.add_trace(go.Scatter(
            x=df_basin_recent["month"],
            y=df_basin_recent["pet"],
            mode="lines",
            line=dict(color="#ef4444", width=2),
            name="Evapotranspiración (PET)",
            hovertemplate="Fecha: %{x|%B %Y}<br>PET: %{y:.1f} mm<extra></extra>"
        ))
        
        fig_balance.update_layout(
            yaxis=dict(gridcolor="#f1f5f9", title="Milímetros (mm/mes)"),
            xaxis=dict(gridcolor="#f1f5f9", title="Mes"),
            template="plotly_white",
            margin=dict(l=40, r=40, t=20, b=40),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_balance, use_container_width=True)

else:
    st.info("No hay datos disponibles en este momento. Por favor, haga clic en el botón 'Sincronizar Datos' en la barra lateral para iniciar la descarga y el cálculo del SPEI.")
