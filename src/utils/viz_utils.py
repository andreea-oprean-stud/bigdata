"""
Visualization Utilities.
Generates interactive Plotly 3D globes, Polar Sky Charts, 2D Ground Tracks,
Folium Leaflet maps, and Big Data Spark analytical visual charts.
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium import plugins


def create_3d_globe_visualization(
    sat_lat: float,
    sat_lon: float,
    sat_alt_km: float,
    sat_name: str,
    ground_track: Optional[List[Dict[str, Any]]] = None,
    obs_lat: Optional[float] = None,
    obs_lon: Optional[float] = None,
    obs_name: Optional[str] = None,
) -> go.Figure:
    """Creates a 3D Earth globe with satellite orbit trajectory and real-time position."""
    # Earth radius normalized to 1.0 unit
    r_earth = 1.0
    r_sat = 1.0 + (sat_alt_km / 6378.137) * 0.25  # Scale altitude for visual clarity

    # Generate wireframe sphere for Earth
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    x_earth = r_earth * np.sin(phi) * np.cos(theta)
    y_earth = r_earth * np.sin(phi) * np.sin(theta)
    z_earth = r_earth * np.cos(phi)

    fig = go.Figure()

    # Earth Sphere Surface
    fig.add_trace(go.Surface(
        x=x_earth,
        y=y_earth,
        z=z_earth,
        colorscale=[[0, "#0b1b3d"], [0.5, "#17386d"], [1.0, "#1f5399"]],
        showscale=False,
        opacity=0.75,
        hoverinfo="none",
    ))

    # Satellite Position vector
    lat_r = math.radians(sat_lat)
    lon_r = math.radians(sat_lon)
    xs = r_sat * math.cos(lat_r) * math.cos(lon_r)
    ys = r_sat * math.cos(lat_r) * math.sin(lon_r)
    zs = r_sat * math.sin(lat_r)

    # Sub-satellite point on Earth surface
    xg = r_earth * math.cos(lat_r) * math.cos(lon_r)
    yg = r_earth * math.cos(lat_r) * math.sin(lon_r)
    zg = r_earth * math.sin(lat_r)

    # Line connecting Earth center to Satellite
    fig.add_trace(go.Scatter3d(
        x=[xg, xs],
        y=[yg, ys],
        z=[zg, zs],
        mode="lines",
        line=dict(color="#00f3ff", width=4, dash="dot"),
        name="Sub-satellite Zenith",
    ))

    # Satellite 3D Marker
    fig.add_trace(go.Scatter3d(
        x=[xs],
        y=[ys],
        z=[zs],
        mode="markers+text",
        marker=dict(size=10, color="#ff0055", symbol="diamond"),
        text=[f"🛰️ {sat_name}<br>Alt: {sat_alt_km:.0f} km"],
        textposition="top center",
        name=sat_name,
    ))

    # Ground Track Path in 3D
    if ground_track:
        tx, ty, tz = [], [], []
        for pt in ground_track:
            p_lat = math.radians(pt["lat"])
            p_lon = math.radians(pt["lon"])
            p_alt = 1.0 + (pt.get("alt_km", sat_alt_km) / 6378.137) * 0.25
            tx.append(p_alt * math.cos(p_lat) * math.cos(p_lon))
            ty.append(p_alt * math.cos(p_lat) * math.sin(p_lon))
            tz.append(p_alt * math.sin(p_lat))

        fig.add_trace(go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="lines",
            line=dict(color="#00f3ff", width=4),
            name="Orbit Path (±1 Orbit)",
        ))

    # Observer location marker
    if obs_lat is not None and obs_lon is not None:
        o_lat_r = math.radians(obs_lat)
        o_lon_r = math.radians(obs_lon)
        xo = r_earth * 1.01 * math.cos(o_lat_r) * math.cos(o_lon_r)
        yo = r_earth * 1.01 * math.cos(o_lat_r) * math.sin(o_lon_r)
        zo = r_earth * 1.01 * math.sin(o_lat_r)

        fig.add_trace(go.Scatter3d(
            x=[xo],
            y=[yo],
            z=[zo],
            mode="markers+text",
            marker=dict(size=8, color="#00ff66", symbol="circle"),
            text=[f"📍 {obs_name or 'Observer'}"],
            textposition="top center",
            name="Observer",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            bgcolor="#0e1117",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        paper_bgcolor="#0e1117",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)"),
        title=dict(text=f"3D Orbit Space View: {sat_name}", font=dict(color="white", size=15)),
        height=480,
    )
    return fig


def create_polar_sky_chart(
    trajectory_points: List[Dict[str, Any]],
    sat_name: str,
    max_el: float,
    start_time: str,
) -> go.Figure:
    """
    Creates an astronomical polar sky plot (Radar chart) representing the satellite pass.
    Azimuth is theta (0-360 deg, North=0, East=90, South=180, West=270).
    Radial distance represents Zenith Angle (90 - Elevation), so Zenith (90° elevation) is at the center (r=0),
    and Horizon (0° elevation) is at the outer perimeter (r=90).
    """
    if not trajectory_points:
        fig = go.Figure()
        fig.update_layout(title="No pass trajectory points available")
        return fig

    # Compute zenith distance for each point: r = 90 - elevation
    thetas = [pt["azimuth"] for pt in trajectory_points]
    zenith_radii = [max(0.0, 90.0 - pt["elevation"]) for pt in trajectory_points]
    elevations = [pt["elevation"] for pt in trajectory_points]
    times = [pt.get("time_utc", "") for pt in trajectory_points]
    ranges = [pt.get("range_km", 0.0) for pt in trajectory_points]
    is_visual = [pt.get("is_visual", False) for pt in trajectory_points]

    hover_texts = [
        f"Time: {t} UTC<br>El: {el:.1f}°<br>Az: {az:.1f}°<br>Range: {r:.0f} km<br>{'✨ Visual' if v else '📻 Radio'}"
        for t, el, az, r, v in zip(times, elevations, thetas, ranges, is_visual)
    ]

    fig = go.Figure()

    # Pass trajectory curve
    fig.add_trace(go.Scatterpolar(
        r=zenith_radii,
        theta=thetas,
        mode="lines+markers",
        line=dict(color="#00e5ff", width=3),
        marker=dict(
            size=6,
            color=["#ffdd00" if v else "#00e5ff" for v in is_visual],
            showscale=False,
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Pass Trajectory",
    ))

    # Mark Start (AOS)
    fig.add_trace(go.Scatterpolar(
        r=[zenith_radii[0]],
        theta=[thetas[0]],
        mode="markers+text",
        marker=dict(size=12, color="#00ff66", symbol="square"),
        text=["AOS (Rise)"],
        textposition="top right",
        hoverinfo="text",
        name="AOS (Rise)",
    ))

    # Mark Peak (TCA)
    max_idx = int(np.argmin(zenith_radii))
    fig.add_trace(go.Scatterpolar(
        r=[zenith_radii[max_idx]],
        theta=[thetas[max_idx]],
        mode="markers+text",
        marker=dict(size=14, color="#ff0055", symbol="star"),
        text=[f"Peak ({max_el:.1f}° El)"],
        textposition="top center",
        hoverinfo="text",
        name="Max Elevation (TCA)",
    ))

    # Mark End (LOS)
    fig.add_trace(go.Scatterpolar(
        r=[zenith_radii[-1]],
        theta=[thetas[-1]],
        mode="markers+text",
        marker=dict(size=12, color="#ff3b30", symbol="triangle-down"),
        text=["LOS (Set)"],
        textposition="bottom right",
        hoverinfo="text",
        name="LOS (Set)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 90],
                tickmode="array",
                tickvals=[0, 30, 60, 80, 90],
                ticktext=["90° (Zenith)", "60°", "30°", "10°", "0° (Horizon)"],
                angle=45,
                gridcolor="#2b313e",
                linecolor="#4e5d6c",
                tickfont=dict(color="#a0aec0", size=10),
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,  # 0 deg (North) at top
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N (0°)", "NE", "E (90°)", "SE", "S (180°)", "SW", "W (270°)", "NW"],
                gridcolor="#2b313e",
                linecolor="#4e5d6c",
                tickfont=dict(color="white", size=11, family="sans-serif"),
            ),
            bgcolor="#131722",
        ),
        paper_bgcolor="#0e1117",
        title=dict(
            text=f"🔭 Sky Radar Plot: {sat_name} (Max El: {max_el:.1f}°) — {start_time}",
            font=dict(color="white", size=14),
        ),
        legend=dict(font=dict(color="white"), orientation="h", y=-0.1),
        margin=dict(l=30, r=30, b=40, t=40),
        height=450,
    )
    return fig


def create_pass_elevation_profile(trajectory_points: List[Dict[str, Any]], sat_name: str) -> go.Figure:
    """Renders Elevation angle and Range vs Time curve for a satellite pass."""
    if not trajectory_points:
        return go.Figure()

    df = pd.DataFrame(trajectory_points)
    fig = go.Figure()

    # Elevation Area Curve
    fig.add_trace(go.Scatter(
        x=df["time_utc"],
        y=df["elevation"],
        mode="lines+markers",
        name="Elevation (°)",
        line=dict(color="#00f3ff", width=3),
        fill="tozeroy",
        fillcolor="rgba(0, 243, 255, 0.15)",
        yaxis="y1",
    ))

    # Range secondary curve
    if "range_km" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time_utc"],
            y=df["range_km"],
            mode="lines",
            name="Slant Range (km)",
            line=dict(color="#ff9900", width=2, dash="dash"),
            yaxis="y2",
        ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#131722",
        title=dict(text=f"Pass Elevation & Slant Range Profile: {sat_name}", font=dict(color="white")),
        xaxis=dict(title="UTC Time", gridcolor="#2b313e", color="white"),
        yaxis=dict(title="Elevation Angle (°)", range=[0, 95], gridcolor="#2b313e", color="#00f3ff"),
        yaxis2=dict(
            title="Slant Range (km)",
            overlaying="y",
            side="right",
            gridcolor="transparent",
            color="#ff9900",
        ),
        legend=dict(font=dict(color="white"), orientation="h", y=1.1),
        margin=dict(l=40, r=40, b=40, t=50),
        height=320,
    )
    return fig


def create_folium_ground_track_map(
    sat_lat: float,
    sat_lon: float,
    sat_alt_km: float,
    sat_name: str,
    footprint_km: float,
    ground_track: Optional[List[Dict[str, Any]]] = None,
    obs_lat: Optional[float] = None,
    obs_lon: Optional[float] = None,
    obs_name: Optional[str] = None,
) -> folium.Map:
    """Generates interactive 2D Folium Map with footprint circle and ground track."""
    m = folium.Map(
        location=[sat_lat, sat_lon],
        zoom_start=3,
        tiles="CartoDB dark_matter",
    )

    # Add Ground Track Polyline
    if ground_track and len(ground_track) > 1:
        # Split polylines across antimeridian to avoid horizontal wrap lines
        segments = []
        current_segment = []
        for i in range(len(ground_track)):
            pt = ground_track[i]
            if current_segment and abs(pt["lon"] - current_segment[-1][1]) > 180:
                segments.append(current_segment)
                current_segment = []
            current_segment.append([pt["lat"], pt["lon"]])
        if current_segment:
            segments.append(current_segment)

        for seg in segments:
            folium.PolyLine(
                seg,
                color="#00f3ff",
                weight=3,
                opacity=0.8,
                tooltip=f"Ground Track for {sat_name}",
            ).add_to(m)

    # Add Footprint Circle
    folium.Circle(
        location=[sat_lat, sat_lon],
        radius=footprint_km * 1000.0,
        color="#00f3ff",
        weight=1,
        fill=True,
        fill_color="#00f3ff",
        fill_opacity=0.15,
        tooltip=f"Visibility Footprint: {footprint_km:.0f} km radius",
    ).add_to(m)

    # Satellite Icon Marker
    folium.Marker(
        location=[sat_lat, sat_lon],
        tooltip=f"🛰️ <b>{sat_name}</b><br>Lat: {sat_lat:.2f}°, Lon: {sat_lon:.2f}°<br>Alt: {sat_alt_km:.0f} km",
        icon=folium.Icon(color="red", icon="plane", prefix="fa"),
    ).add_to(m)

    # Observer Location Marker
    if obs_lat is not None and obs_lon is not None:
        folium.Marker(
            location=[obs_lat, obs_lon],
            tooltip=f"📍 <b>{obs_name or 'Observer'}</b><br>Lat: {obs_lat:.2f}°, Lon: {obs_lon:.2f}°",
            icon=folium.Icon(color="green", icon="eye", prefix="fa"),
        ).add_to(m)

    return m
