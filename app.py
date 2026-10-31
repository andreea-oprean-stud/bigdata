"""
Main Streamlit Application: Big Data Platform for Satellite Pass Prediction & Orbital Analytics.
Integrates Data Ingestion, SGP4 Orbital Mechanics, PySpark Parquet Lake, and Interactive Visualizations.
"""

import time
from datetime import datetime, timezone, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

from config import (
    POPULAR_SATELLITES,
    OBSERVER_PRESETS,
    CELESTRAK_GROUPS,
    N2YO_API_KEY,
)
from src.data_ingestion import (
    CelesTrakClient,
    N2YOClient,
    DataQualityValidator,
    SAMPLE_TLES,
    get_sample_satcat_df,
)
from src.orbital_engine import SGP4Engine, PassPredictor, SolarCalculator
from src.big_data_pipeline import (
    SparkSessionManager,
    ParquetLakeManager,
    HistoricalOrbitalGenerator,
    SparkOrbitalAnalytics,
    ScalabilityBenchmark,
)
from src.validation import OrbitalValidator
from src.utils import (
    create_3d_globe_visualization,
    create_polar_sky_chart,
    create_pass_elevation_profile,
    create_folium_ground_track_map,
    format_lat_lon,
    haversine_distance_km,
)

# Page configuration
st.set_page_config(
    page_title="Satellite Pass Prediction & Big Data Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark-themed aerospace UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f3ff;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1a1f2c;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00f3ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }
    .badge-visual {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-radio {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Caching helpers
@st.cache_resource
def get_clients():
    celestrak = CelesTrakClient()
    n2yo = N2YOClient()
    spark_analytics = SparkOrbitalAnalytics()
    lake_mgr = ParquetLakeManager()
    benchmark_runner = ScalabilityBenchmark()
    validator = OrbitalValidator(n2yo_client=n2yo, celestrak_client=celestrak)
    generator = HistoricalOrbitalGenerator()
    return celestrak, n2yo, spark_analytics, lake_mgr, benchmark_runner, validator, generator


celestrak_client, n2yo_client, spark_analytics, lake_mgr, benchmark_runner, validator, generator = get_clients()


# ---------------- SIDEBAR: OBSERVER & SATELLITE SELECTION ---------------- #
st.sidebar.title("🛰️ Mission Control")

# 1. Observer Location Configuration
st.sidebar.subheader("📍 Observer Location")
observer_mode = st.sidebar.radio("Location Mode", ["Preset City", "Custom Coordinates"], index=0)

if observer_mode == "Preset City":
    selected_city = st.sidebar.selectbox("Select Ground Station", list(OBSERVER_PRESETS.keys()), index=0)
    obs_info = OBSERVER_PRESETS[selected_city]
    obs_lat = obs_info["lat"]
    obs_lon = obs_info["lon"]
    obs_alt = obs_info["alt"]
    obs_name = selected_city
else:
    obs_name = st.sidebar.text_input("Station Name", "My Ground Station")
    obs_lat = st.sidebar.number_input("Latitude (°)", value=44.4268, min_value=-90.0, max_value=90.0, format="%.4f")
    obs_lon = st.sidebar.number_input("Longitude (°)", value=26.1025, min_value=-180.0, max_value=180.0, format="%.4f")
    obs_alt = st.sidebar.number_input("Altitude (m)", value=75.0, min_value=0.0, max_value=9000.0)

st.sidebar.caption(f"Coordinates: {format_lat_lon(obs_lat, obs_lon)} | Alt: {obs_alt:.0f}m")

# 2. Target Satellite Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Target Satellite")
sat_selection_mode = st.sidebar.radio("Satellite Selection", ["Popular Satellites", "NORAD ID / Custom"], index=0)

if sat_selection_mode == "Popular Satellites":
    sat_options = {f"{name} (NORAD {nid})": nid for nid, name in POPULAR_SATELLITES.items()}
    chosen_label = st.sidebar.selectbox("Choose Satellite", list(sat_options.keys()), index=0)
    current_norad_id = sat_options[chosen_label]
else:
    current_norad_id = st.sidebar.number_input("Enter NORAD ID", min_value=1, max_value=99999, value=25544, step=1)

# Fetch TLE for target satellite
with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ System Status")
    n2yo_key_input = st.text_input("N2YO API Key (Optional)", value=n2yo_client.api_key, type="password")
    if n2yo_key_input != n2yo_client.api_key:
        n2yo_client.api_key = n2yo_key_input

    api_badge = "🟢 Live Connected" if n2yo_client.is_configured() else "🟡 SGP4 Offline Engine Mode"
    st.caption(f"API Mode: **{api_badge}**")

    # Spark Data Lake quick stats
    lake_stats = lake_mgr.get_lake_stats()
    st.caption(f"Parquet Lake: **{lake_stats['total_records']:,}** records ({lake_stats['total_size_mb']} MB)")


# Load TLE data
tle_data = celestrak_client.get_tle_by_catnr(int(current_norad_id))
if not tle_data:
    st.error(f"Could not load TLE for NORAD ID {current_norad_id}. Falling back to ISS default.")
    tle_data = SAMPLE_TLES[25544]
    tle_data["norad_id"] = 25544

sat_engine = SGP4Engine(tle_data["line1"], tle_data["line2"], name=tle_data.get("name", f"SAT-{current_norad_id}"))
sat_name = tle_data.get("name", f"SAT-{current_norad_id}")


# ---------------- MAIN HEADER ---------------- #
st.markdown('<div class="main-header">🛰️ Big Data Platform for Satellite Pass Prediction & Orbital Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Scalable SGP4 Orbital Propagation, Distributed Apache Spark Analytics & Ground Station Pass Tracking</div>', unsafe_allow_html=True)


# ---------------- NAVIGATION TABS ---------------- #
tabs = st.tabs([
    "🛰️ Real-Time Tracking & 3D/2D Orbit",
    "🔭 Pass Predictor & Sky Radar",
    "📚 NORAD SATCAT Explorer",
    "⚡ Spark Big Data Lake & Analytics",
    "📊 Scalability & Storage Benchmarks",
    "🧪 Validation & Reference Engine",
    "ℹ️ Architecture & Proposal",
])


# =========================================================================
# TAB 1: REAL-TIME TRACKING & 3D/2D ORBIT
# =========================================================================
with tabs[0]:
    st.subheader(f"Live Telemetry & Orbital Ephemeris: **{sat_name}** (NORAD {current_norad_id})")

    now_utc = datetime.now(timezone.utc)
    current_pos = sat_engine.get_geodetic_position(now_utc)
    current_topo = sat_engine.get_topocentric_coordinates(now_utc, obs_lat, obs_lon, obs_alt)
    sol_vis = SolarCalculator.evaluate_pass_visibility(
        obs_lat, obs_lon, current_pos["latitude"], current_pos["longitude"], current_pos["altitude_km"], now_utc
    )

    # Telemetry Metric Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Latitude", f"{current_pos['latitude']:.2f}°")
    c2.metric("Longitude", f"{current_pos['longitude']:.2f}°")
    c3.metric("Altitude", f"{current_pos['altitude_km']:.1f} km")
    c4.metric("Orbital Velocity", f"{current_pos['velocity_km_s']:.2f} km/s")
    c5.metric("Observer Elevation", f"{current_topo['elevation_deg']:.1f}°")
    c6.metric("Slant Range", f"{current_topo['range_km']:.0f} km")

    st.markdown("---")

    # Compute Ground Track for +/- 45 minutes
    track_start = now_utc - timedelta(minutes=30)
    ground_track = sat_engine.compute_ground_track(track_start, duration_minutes=120, step_seconds=60)

    col_3d, col_2d = st.columns([1, 1])

    with col_3d:
        st.markdown("#### 🌐 3D Space Orbit View")
        globe_fig = create_3d_globe_visualization(
            sat_lat=current_pos["latitude"],
            sat_lon=current_pos["longitude"],
            sat_alt_km=current_pos["altitude_km"],
            sat_name=sat_name,
            ground_track=ground_track,
            obs_lat=obs_lat,
            obs_lon=obs_lon,
            obs_name=obs_name,
        )
        st.plotly_chart(globe_fig, use_container_width=True)

    with col_2d:
        st.markdown("#### 🗺️ 2D Ground Track & Coverage Footprint")
        folium_map = create_folium_ground_track_map(
            sat_lat=current_pos["latitude"],
            sat_lon=current_pos["longitude"],
            sat_alt_km=current_pos["altitude_km"],
            sat_name=sat_name,
            footprint_km=current_pos["footprint_radius_km"],
            ground_track=ground_track,
            obs_lat=obs_lat,
            obs_lon=obs_lon,
            obs_name=obs_name,
        )
        st_folium(folium_map, height=450, width="100%")

    # Orbit summary parameters
    st.markdown("#### 📋 Two-Line Element (TLE) Details")
    tle_col1, tle_col2 = st.columns([2, 1])
    with tle_col1:
        st.code(f"{tle_data['line1']}\n{tle_data['line2']}", language="text")
    with tle_col2:
        st.info(
            f"**Footprint Radius:** {current_pos['footprint_radius_km']:.0f} km\n\n"
            f"**Sunlit Status:** {'☀️ Sunlit' if sol_vis['satellite_sunlit'] else '🌑 Eclipse / Shadow'}\n\n"
            f"**Observer Sky:** {sol_vis['twilight_category']} (Solar El: {sol_vis['solar_elevation_deg']:.1f}°)"
        )


# =========================================================================
# TAB 2: PASS PREDICTOR & SKY RADAR
# =========================================================================
with tabs[1]:
    st.subheader(f"Upcoming Satellite Pass Predictions for **{obs_name}**")
    st.markdown("Calculates geometric horizon crossing, peak elevation, duration, and optical visibility.")

    pred_c1, pred_c2, pred_c3 = st.columns([1, 1, 1])
    with pred_c1:
        lookahead_days = st.slider("Prediction Horizon (Days)", min_value=1, max_value=7, value=3)
    with pred_c2:
        min_el_threshold = st.slider("Minimum Elevation Threshold (°)", min_value=5, max_value=45, value=10)
    with pred_c3:
        st.write("")
        st.write("")
        calc_button = st.button("🚀 Calculate Upcoming Passes", type="primary", use_container_width=True)

    # Predict passes
    predictor = PassPredictor(sat_engine)
    with st.spinner("Propagating orbital steps and predicting passes..."):
        pass_list = predictor.predict_passes(
            obs_lat=obs_lat,
            obs_lon=obs_lon,
            obs_alt_m=obs_alt,
            duration_days=lookahead_days,
            min_elevation_deg=min_el_threshold,
        )

    if not pass_list:
        st.warning(f"No passes above {min_el_threshold}° elevation found for {sat_name} in the next {lookahead_days} days.")
    else:
        st.success(f"Found **{len(pass_list)}** upcoming passes for **{sat_name}** over **{obs_name}**!")

        # Summary Table
        table_rows = []
        for p in pass_list:
            table_rows.append({
                "Pass #": p["pass_id"],
                "Start Time (UTC)": p["aos_time_utc"],
                "Max Elevation Time (UTC)": p["tca_time_utc"],
                "End Time (UTC)": p["los_time_utc"],
                "Max El (°)": f"{p['max_elevation_deg']:.1f}°",
                "Duration": f"{p['duration_minutes']} min",
                "Trajectory (Rise → Set)": f"{p['start_compass']} ({p['start_azimuth_deg']}°) → {p['end_compass']} ({p['end_azimuth_deg']}°)",
                "Type": "✨ Visual (Visible)" if p["is_visual"] else "📻 Radio / Day",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Detailed Pass Sky Radar & Elevation Profile")

        pass_options = {f"Pass #{p['pass_id']}: {p['aos_time_utc']} (Max: {p['max_elevation_deg']}° - {p['visibility_type']})": idx for idx, p in enumerate(pass_list)}
        selected_pass_label = st.selectbox("Select Pass to Inspect", list(pass_options.keys()))
        selected_pass = pass_list[pass_options[selected_pass_label]]

        col_radar, col_elev = st.columns([1, 1])

        with col_radar:
            radar_fig = create_polar_sky_chart(
                trajectory_points=selected_pass["trajectory"],
                sat_name=sat_name,
                max_el=selected_pass["max_elevation_deg"],
                start_time=selected_pass["aos_time_utc"],
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        with col_elev:
            elev_fig = create_pass_elevation_profile(
                trajectory_points=selected_pass["trajectory"],
                sat_name=sat_name,
            )
            st.plotly_chart(elev_fig, use_container_width=True)

            # Details card
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: left;">
                    <b>Pass Details:</b><br>
                    • <b>Acquisition of Signal (Rise):</b> {selected_pass['aos_time_utc']} UTC at {selected_pass['start_compass']} ({selected_pass['start_azimuth_deg']}° Azimuth)<br>
                    • <b>Maximum Culmination (TCA):</b> {selected_pass['tca_time_utc']} UTC at {selected_pass['max_elevation_deg']}° Elevation ({selected_pass['max_compass']})<br>
                    • <b>Loss of Signal (Set):</b> {selected_pass['los_time_utc']} UTC at {selected_pass['end_compass']} ({selected_pass['end_azimuth_deg']}° Azimuth)<br>
                    • <b>Duration:</b> {selected_pass['duration_seconds']} seconds ({selected_pass['duration_minutes']} minutes)<br>
                    • <b>Lighting / Observer Condition:</b> {selected_pass['twilight_condition']}<br>
                    • <b>Classification:</b> {selected_pass['visibility_type']}
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================================
# TAB 3: NORAD SATCAT EXPLORER
# =========================================================================
with tabs[2]:
    st.subheader("📚 CelesTrak SATCAT & TLE Catalog Explorer")
    st.markdown("Explore catalog metadata, filter active satellites vs space debris, and validate TLE checksums.")

    satcat_df = celestrak_client.get_satcat_catalog()

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        type_filter = st.multiselect(
            "Object Type",
            options=sorted(list(satcat_df["object_type"].unique())),
            default=list(satcat_df["object_type"].unique()),
        )
    with f_col2:
        regime_filter = st.multiselect(
            "Orbit Regime",
            options=sorted(list(satcat_df["orbit_regime"].unique())),
            default=list(satcat_df["orbit_regime"].unique()),
        )
    with f_col3:
        owner_filter = st.multiselect(
            "Owner Nation/Agency",
            options=sorted(list(satcat_df["owner"].unique())),
            default=list(satcat_df["owner"].unique())[:5],
        )
    with f_col4:
        search_query = st.text_input("Search Name / NORAD ID", "")

    # Apply filters
    filtered_df = satcat_df[
        satcat_df["object_type"].isin(type_filter)
        & satcat_df["orbit_regime"].isin(regime_filter)
        & satcat_df["owner"].isin(owner_filter)
    ]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["object_name"].str.contains(search_query, case=False, na=False)
            | filtered_df["norad_id"].astype(str).str.contains(search_query)
        ]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(satcat_df):,} catalog records**")
    st.dataframe(filtered_df, use_container_width=True)

    # TLE Validator Section
    st.markdown("---")
    st.subheader("🧪 TLE Format & Checksum Validator")
    val_c1, val_c2 = st.columns([2, 1])
    with val_c1:
        custom_tle_l1 = st.text_input("TLE Line 1", value=tle_data["line1"])
        custom_tle_l2 = st.text_input("TLE Line 2", value=tle_data["line2"])
    with val_c2:
        st.write("")
        st.write("")
        is_valid, msg = DataQualityValidator.validate_tle_pair(custom_tle_l1, custom_tle_l2)
        if is_valid:
            st.success("✅ **TLE Checksum & Schema Valid!**\n\nCompliant with NORAD standard.")
        else:
            st.error(f"❌ **TLE Validation Error:**\n\n{msg}")


# =========================================================================
# TAB 4: SPARK BIG DATA LAKE & ANALYTICS
# =========================================================================
with tabs[3]:
    st.subheader("⚡ Apache Spark Distributed Orbital Analytics & Data Lake")
    st.markdown("Processes massive satellite position records stored in partitioned Parquet files (`data/parquet_lake/`).")

    lake_info = lake_mgr.get_lake_stats()

    # Top Metric Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records in Lake", f"{lake_info['total_records']:,}")
    m2.metric("Parquet Storage Size", f"{lake_info['total_size_mb']} MB")
    m3.metric("Partitioned Files", f"{lake_info['file_count']}")
    m4.metric("Spark Cluster Mode", "Local Multi-Core")

    st.markdown("---")

    # Mass Generator Action
    with st.expander("🛠️ Generate Large-Scale Orbital Records into Data Lake (PySpark)", expanded=(lake_info["total_records"] == 0)):
        st.write("Generate synthetic and propagated orbital observations across constellations and debris fields:")
        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            records_to_generate = st.select_slider(
                "Select Volume to Generate",
                options=[10000, 50000, 100000, 250000, 500000, 1000000],
                value=50000,
            )
        with gen_col2:
            st.write("")
            st.write("")
            if st.button("🚀 Run PySpark Generator", type="primary", use_container_width=True):
                with st.spinner(f"Generating {records_to_generate:,} records and writing partitioned Parquet..."):
                    gen_result = generator.generate_and_save_dataset(total_records=records_to_generate)
                    st.success(f"Generated {gen_result['generated_records']:,} records in {gen_result['total_time_sec']}s (Throughput: {gen_result['throughput_records_per_sec']:,} rec/s)!")
                    st.rerun()

    if lake_info["total_records"] > 0:
        spark_analytics.refresh_view()

        # Analytics Visualizations Row 1
        an_col1, an_col2 = st.columns(2)

        with an_col1:
            st.markdown("#### 🛰️ Population Distribution by Regime & Object Type")
            dist_df = spark_analytics.get_regime_and_type_distribution()
            if not dist_df.empty:
                fig_dist = px.bar(
                    dist_df,
                    x="orbit_regime",
                    y="count",
                    color="object_type",
                    barmode="group",
                    title="Object Count by Orbital Regime (LEO, MEO, GEO, HEO)",
                    color_discrete_sequence=["#00f3ff", "#ff0055", "#ffaa00"],
                )
                fig_dist.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#131722", font=dict(color="white"))
                st.plotly_chart(fig_dist, use_container_width=True)

        with an_col2:
            st.markdown("#### 💥 Space Debris Density by Altitude (50km Shells)")
            debris_df = spark_analytics.get_debris_altitude_density(bin_size_km=50)
            if not debris_df.empty:
                fig_debris = px.area(
                    debris_df,
                    x="altitude_bin_km",
                    y=["debris_count", "payload_count"],
                    title="Debris vs Active Payload Density in LEO (<2000 km)",
                    color_discrete_sequence=["#ff0055", "#00f3ff"],
                )
                fig_debris.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#131722", font=dict(color="white"))
                st.plotly_chart(fig_debris, use_container_width=True)

        # Analytics Visualizations Row 2
        an_col3, an_col4 = st.columns(2)

        with an_col3:
            st.markdown("#### 🌍 Fleet Breakdown by Nation / Operator")
            fleet_df = spark_analytics.get_fleet_by_owner()
            if not fleet_df.empty:
                fig_fleet = px.pie(
                    fleet_df.head(7),
                    names="owner",
                    values="total_records",
                    title="Catalog Records by Country / Agency Share",
                    hole=0.4,
                )
                fig_fleet.update_layout(paper_bgcolor="#0e1117", font=dict(color="white"))
                st.plotly_chart(fig_fleet, use_container_width=True)

        with an_col4:
            st.markdown("#### 🔬 Keplerian Velocity vs Altitude Profile")
            sample_vel_df = spark_analytics.get_velocity_vs_altitude_profile(sample_size=2000)
            if not sample_vel_df.empty:
                fig_vel = px.scatter(
                    sample_vel_df,
                    x="altitude_km",
                    y="velocity_km_s",
                    color="orbit_regime",
                    title="Orbital Mechanics Validation: Velocity = √(GM / r)",
                    opacity=0.6,
                )
                fig_vel.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#131722", font=dict(color="white"))
                st.plotly_chart(fig_vel, use_container_width=True)

        # Interactive Spark SQL Console
        st.markdown("---")
        st.subheader("💻 Interactive Spark SQL Query Console")
        default_query = "SELECT orbit_regime, COUNT(*) as objects, ROUND(AVG(altitude_km), 1) as avg_alt_km, ROUND(AVG(velocity_km_s), 2) as avg_vel_km_s FROM orbital_records GROUP BY orbit_regime ORDER BY objects DESC"
        custom_sql = st.text_area("Enter Spark SQL Query (Table name: orbital_records)", value=default_query, height=80)
        if st.button("▶️ Execute Spark SQL", type="primary"):
            sql_res = spark_analytics.execute_custom_sql(custom_sql)
            if sql_res["success"]:
                st.success(f"Query executed in **{sql_res['execution_time_sec']} seconds** ({sql_res['rows_returned']} rows returned):")
                st.dataframe(sql_res["data"], use_container_width=True)
            else:
                st.error(f"Spark SQL Error: {sql_res['error']}")
    else:
        st.info("The Parquet Lake is currently empty. Click 'Generate Large-Scale Orbital Records' above to populate it.")


# =========================================================================
# TAB 5: SCALABILITY & STORAGE BENCHMARKS
# =========================================================================
with tabs[4]:
    st.subheader("📊 Big Data Scalability & Storage Efficiency Benchmarking")
    st.markdown("Answers the proposal's primary research question: *'Evaluate data-processing speed, storage efficiency, and scalability.'*")

    bench_c1, bench_c2 = st.columns([1, 1])

    with bench_c1:
        st.markdown("#### 💾 Storage Compression Test (CSV vs Parquet)")
        comp_records = st.selectbox("Sample Size for Compression Benchmark", [20000, 50000, 100000], index=1)
        if st.button("🧪 Run Storage Compression Benchmark", type="primary"):
            with st.spinner("Writing CSV, Snappy Parquet, and Gzip Parquet..."):
                storage_res = benchmark_runner.run_storage_efficiency_test(record_count=comp_records)
                st.success(f"Completed! Parquet achieved **{storage_res['snappy_compression_ratio']}x** compression ratio with **{storage_res['storage_savings_pct']}%** storage savings.")

    with bench_c2:
        st.markdown("#### ⚡ Multi-Scale Ingestion & Query Benchmark")
        if st.button("🚀 Run Multi-Scale Spark Benchmark (20k → 250k)", type="primary"):
            with st.spinner("Running scalability tests across scales..."):
                scale_df = benchmark_runner.run_scaling_benchmark([20000, 50000, 100000, 250000])
                st.success("Scalability benchmark suite completed!")

    # Display benchmark visualizations if available
    latest_bench = benchmark_runner.get_latest_benchmark_results()

    if latest_bench["storage_efficiency"]:
        st.markdown("---")
        st.markdown("### 🗜️ Storage Efficiency Comparison Results")
        se = latest_bench["storage_efficiency"]

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Raw CSV Size", f"{se['csv_size_mb']} MB")
        col_s2.metric("Parquet (Snappy)", f"{se['parquet_snappy_size_mb']} MB", f"-{se['storage_savings_pct']}%")
        col_s3.metric("Parquet (Gzip)", f"{se['parquet_gzip_size_mb']} MB")
        col_s4.metric("Snappy Ratio", f"{se['snappy_compression_ratio']}x")

        fig_comp = go.Figure(data=[
            go.Bar(name="Disk Footprint (MB)", x=["Raw CSV", "Parquet (Snappy)", "Parquet (Gzip)"], y=[se["csv_size_mb"], se["parquet_snappy_size_mb"], se["parquet_gzip_size_mb"]], marker_color=["#ff0055", "#00f3ff", "#00ff66"])
        ])
        fig_comp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#131722", font=dict(color="white"), title="Storage Size Footprint (MB) for Identical Orbital Observations")
        st.plotly_chart(fig_comp, use_container_width=True)

    if latest_bench["scaling_benchmark"] is not None and not latest_bench["scaling_benchmark"].empty:
        st.markdown("---")
        st.markdown("### 📈 Spark Throughput & Query Latency Scaling")
        sc_df = latest_bench["scaling_benchmark"]
        st.dataframe(sc_df, use_container_width=True)

        sc_c1, sc_c2 = st.columns(2)
        with sc_c1:
            fig_tp = px.line(
                sc_df,
                x="record_count",
                y="throughput_records_s",
                markers=True,
                title="Pipeline Throughput (Records / Second) vs Data Volume",
                color_discrete_sequence=["#00f3ff"],
            )
            fig_tp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#131722", font=dict(color="white"))
            st.plotly_chart(fig_tp, use_container_width=True)

        with sc_c2:
            fig_lat = px.line(
                sc_df,
                x="record_count",
                y=["aggregation_query_s", "filtered_query_s"],
                markers=True,
                title="Spark SQL Aggregation Latency (Seconds) vs Data Volume",
                color_discrete_sequence=["#ff0055", "#00ff66"],
            )
            fig_lat.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#131722", font=dict(color="white"))
            st.plotly_chart(fig_lat, use_container_width=True)


# =========================================================================
# TAB 6: VALIDATION & REFERENCE ENGINE
# =========================================================================
with tabs[5]:
    st.subheader(f"🧪 Orbital Propagation Validation: **{sat_name}**")
    st.markdown("Validates local SGP4 orbital mechanics against N2YO REST API & reference ephemeris to quantify positional and elevation error residuals.")

    val_col1, val_col2 = st.columns([1, 1])
    with val_col1:
        val_sec = st.slider("Validation Time Window (Seconds)", min_value=5, max_value=30, value=10)
    with val_col2:
        st.write("")
        st.write("")
        run_val_btn = st.button("🔬 Execute Validation Suite", type="primary")

    val_res = validator.validate_satellite_positions(
        norad_id=int(current_norad_id),
        obs_lat=obs_lat,
        obs_lon=obs_lon,
        obs_alt_m=obs_alt,
        seconds=val_sec,
    )

    if "error" in val_res:
        st.error(val_res["error"])
    else:
        m = val_res["metrics"]
        st.success(f"Validation Reference Source: **{m['source']}** | Validation Check: **{'PASSED (Residuals < 0.1°)' if m['validation_passed'] else 'ATTENTION'}**")

        # Metric summary
        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("MAE Latitude Residual", f"{m['mae_latitude_deg']}°")
        vm2.metric("MAE Longitude Residual", f"{m['mae_longitude_deg']}°")
        vm3.metric("MAE Altitude Residual", f"{m['mae_altitude_km']} km")
        vm4.metric("MAE Elevation Residual", f"{m['mae_elevation_deg']}°")

        st.markdown("#### 📋 Position Residuals Comparison Table")
        st.dataframe(val_res["comparison_df"], use_container_width=True)


# =========================================================================
# TAB 7: ARCHITECTURE & PROPOSAL
# =========================================================================
with tabs[6]:
    st.subheader("🏛️ Big Data System Architecture & Proposal Overview")
    st.markdown(
        """
        ### 📖 Project Title
        **Big Data Platform for Satellite Pass Prediction Using NORAD Catalog Data and REST APIs**

        ### 🎯 Problem Statement & Research Question
        > **How can a scalable big data pipeline use satellite orbital data to predict passes over a selected location while efficiently processing large volumes of historical records?**

        ### 🧱 Architecture Pipeline
        1. **Data Ingestion (`data_ingestion`)**:
           - Live REST ingestion from **N2YO.com** & **CelesTrak GP/SATCAT**.
           - Checksum and TLE schema validation with automated cleaning and local raw cache.
        2. **Orbital Calculation Engine (`orbital_engine`)**:
           - High-precision SGP4 propagation (TEME to ECEF/Geodetic conversion via Bowring & GMST).
           - Topocentric observer coordinate transformation (Azimuth, Elevation, Slant Range).
           - Precise AOS/TCA/LOS pass prediction, solar illumination, and twilight optical visibility classification.
        3. **Big Data Processing & Partitioned Parquet Lake (`big_data_pipeline`)**:
           - **Apache Spark (PySpark)** for distributed historical record processing.
           - Dual partitioned Parquet data lake (`orbit_regime`, `object_type`).
           - Distributed analytics for space debris altitude density, orbital fleet distributions, and Keplerian profiles.
           - Scalability and storage efficiency benchmarking (CSV vs Snappy Parquet vs Gzip Parquet).
        4. **Visualization & Integration Dashboard (`dashboard`)**:
           - Interactive multi-tab Streamlit interface.
           - 3D Plotly Space Orbit Globe, 2D Folium Map, and Astronomical Polar Sky Radar charts.
        """
    )
