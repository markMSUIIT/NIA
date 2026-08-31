"""Interactive local water-distance viewer for Midsayap.

This app never contacts Google Sheets or any external API. Export the read-only
sheet as CSV/XLSX and load it locally through the sidebar.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio
import streamlit as st
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.warp import transform as reproject_coordinates
from scipy.interpolate import griddata
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent
BOUNDARY_FILE = ROOT / "NIA_Midsayap_Boundaries.shp"
DEM_FILE = ROOT / "Midsayap_Copernicus_GLO30_UTM51.tif"
DISTANCE_RASTERS = sorted(ROOT.glob("*water*distance*.tif")) + sorted(ROOT.glob("*distance*water*.tif"))
PUBLIC_SHEET_CSV = "https://docs.google.com/spreadsheets/d/1ptkH-wUQ_VTyNv-afoFM_YizJILHwYw_mUtQpeiZPEE/export?format=csv&gid=0"
DEVICE_LOCATIONS_FILE = ROOT / "device_locations_template.csv"
SOIL_SURFACE_READING_CM = 20.0
IOT_FACTORS = [
    "Moisture (%)", "Soil Temp (°C)", "EC (µS/cm)", "pH", "Nitrogen (ppm)",
    "Phosphorus (ppm)", "Potassium (ppm)", "Air Temp (°C)", "Humidity (%)", "Wind Speed (m/s)",
]

# Deep blue -> blue -> cyan -> yellow -> orange/red, matching the supplied reference.
WATER_COLORS = [
    [0.00, "#0700d9"],
    [0.18, "#004dff"],
    [0.42, "#00b6df"],
    [0.62, "#16dec4"],
    [0.78, "#e5ef28"],
    [0.90, "#ff9700"],
    [1.00, "#b30000"],
]
RICE_AWD_COLORS = [
    [0.00, "#1579b5"],  # Water at soil surface
    [0.02, "#2f9e44"],  # Small drawdown / watch
    [0.37, "#f0c419"],
    [0.38, "#e67e22"],  # Default 15 cm deficit irrigation threshold
    [1.00, "#c0392b"],
]


@st.cache_data(show_spinner=False)
def load_boundaries() -> gpd.GeoDataFrame:
    return gpd.read_file(BOUNDARY_FILE)


def feature_columns(frame: gpd.GeoDataFrame) -> list[str]:
    return [column for column in frame.columns if column != "geometry"]


def choose_place_column(frame: gpd.GeoDataFrame) -> str:
    preferred = ["NAME_OF_IA", "NAME_OF_SY", "MUNICIPALI", "NAME", "IMO"]
    return next((column for column in preferred if column in frame.columns), feature_columns(frame)[0])


def draw_boundary(fig: go.Figure, boundaries: gpd.GeoDataFrame, crs) -> None:
    projected = boundaries.to_crs(crs)
    outline = projected.geometry.union_all()
    polygons = [outline] if outline.geom_type == "Polygon" else list(outline.geoms)
    x_values, y_values = [], []
    for polygon in polygons:
        x, y = polygon.exterior.xy
        x_values.extend([*x, None])
        y_values.extend([*y, None])
    fig.add_trace(go.Scattergl(
        x=x_values, y=y_values, mode="lines", line={"color": "#f4f4f4", "width": 1.4},
        hoverinfo="skip", showlegend=False,
    ))


def raster_distance(path_or_upload, selected: gpd.GeoDataFrame) -> tuple[np.ndarray, rasterio.Affine, object]:
    """Crop a distance raster to the active place boundary and return its grid."""
    with rasterio.open(path_or_upload) as dataset:
        clip = selected.to_crs(dataset.crs)
        data, transform = mask(dataset, [mapping(item) for item in clip.geometry], crop=True, filled=True, nodata=np.nan)
        values = data[0].astype(float)
        if dataset.nodata is not None:
            values[np.isclose(values, dataset.nodata)] = np.nan
        return values, transform, dataset.crs


def dem_preview(selected: gpd.GeoDataFrame) -> tuple[np.ndarray, rasterio.Affine, object]:
    """Provide terrain only as a clearly-labelled fallback, not as water distance."""
    return raster_distance(DEM_FILE, selected)


def interpolate_sheet(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    selected: gpd.GeoDataFrame,
    raster_crs,
    grid_size: int,
) -> tuple[np.ndarray, rasterio.Affine]:
    clean = table[[x_column, y_column, value_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 3:
        raise ValueError("At least three rows with valid coordinates and distance values are required.")

    points = gpd.GeoDataFrame(
        clean, geometry=gpd.points_from_xy(clean[x_column], clean[y_column]), crs="EPSG:4326"
    ).to_crs(raster_crs)
    area = selected.to_crs(raster_crs).geometry.union_all()
    min_x, min_y, max_x, max_y = area.bounds
    width = grid_size
    height = max(2, round(grid_size * (max_y - min_y) / (max_x - min_x)))
    x = np.linspace(min_x, max_x, width)
    y = np.linspace(max_y, min_y, height)
    grid_x, grid_y = np.meshgrid(x, y)
    xy = np.column_stack([points.geometry.x, points.geometry.y])
    values = clean[value_column].to_numpy(float)

    # Linear interpolation gives smooth coverage; nearest fills edges outside its convex hull.
    grid = griddata(xy, values, (grid_x, grid_y), method="linear")
    nearest = griddata(xy, values, (grid_x, grid_y), method="nearest")
    grid = np.where(np.isnan(grid), nearest, grid)
    pixel_width = (max_x - min_x) / max(width - 1, 1)
    pixel_height = (max_y - min_y) / max(height - 1, 1)
    transform = rasterio.Affine(pixel_width, 0, min_x, 0, -pixel_height, max_y)
    inside = geometry_mask([mapping(area)], out_shape=grid.shape, transform=transform, invert=True)
    return np.where(inside, grid, np.nan), transform


def interpolate_idw(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    selected: gpd.GeoDataFrame,
    raster_crs,
    grid_size: int,
    power: float = 2.0,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Estimate a smooth surface with inverse-distance weighting (IDW)."""
    clean = table[[x_column, y_column, value_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 3:
        raise ValueError("At least three rows with valid coordinates and distance values are required.")
    points = gpd.GeoDataFrame(clean, geometry=gpd.points_from_xy(clean[x_column], clean[y_column]), crs="EPSG:4326").to_crs(raster_crs)
    area = selected.to_crs(raster_crs).geometry.union_all()
    min_x, min_y, max_x, max_y = area.bounds
    width = grid_size
    height = max(2, round(grid_size * (max_y - min_y) / (max_x - min_x)))
    x = np.linspace(min_x, max_x, width)
    y = np.linspace(max_y, min_y, height)
    grid_x, grid_y = np.meshgrid(x, y)
    dx = grid_x[..., np.newaxis] - points.geometry.x.to_numpy()
    dy = grid_y[..., np.newaxis] - points.geometry.y.to_numpy()
    distance = np.hypot(dx, dy)
    weights = 1 / np.maximum(distance, 1.0) ** power
    grid = np.sum(weights * clean[value_column].to_numpy(float), axis=2) / np.sum(weights, axis=2)
    pixel_width = (max_x - min_x) / max(width - 1, 1)
    pixel_height = (max_y - min_y) / max(height - 1, 1)
    transform = rasterio.Affine(pixel_width, 0, min_x, 0, -pixel_height, max_y)
    inside = geometry_mask([mapping(area)], out_shape=grid.shape, transform=transform, invert=True)
    return np.where(inside, grid, np.nan), transform


def nearest_sensor_distance(
    points_table: pd.DataFrame, selected: gpd.GeoDataFrame, raster_crs, grid_size: int
) -> tuple[np.ndarray, rasterio.Affine]:
    """Return distance in metres to the nearest sensor, a transparent confidence proxy."""
    points = gpd.GeoDataFrame(points_table, geometry=gpd.points_from_xy(points_table.longitude, points_table.latitude), crs="EPSG:4326").to_crs(raster_crs)
    area = selected.to_crs(raster_crs).geometry.union_all()
    min_x, min_y, max_x, max_y = area.bounds
    width, height = grid_size, max(2, round(grid_size * (max_y - min_y) / (max_x - min_x)))
    x, y = np.linspace(min_x, max_x, width), np.linspace(max_y, min_y, height)
    grid_x, grid_y = np.meshgrid(x, y)
    distance = np.min(np.hypot(grid_x[..., np.newaxis] - points.geometry.x.to_numpy(), grid_y[..., np.newaxis] - points.geometry.y.to_numpy()), axis=2)
    transform = rasterio.Affine((max_x - min_x) / max(width - 1, 1), 0, min_x, 0, -(max_y - min_y) / max(height - 1, 1), max_y)
    inside = geometry_mask([mapping(area)], out_shape=distance.shape, transform=transform, invert=True)
    return np.where(inside, distance, np.nan), transform


def validate_idw(points_table: pd.DataFrame, raster_crs) -> tuple[float, float]:
    """Leave one station out, predict it from the others, and return MAE/RMSE in cm."""
    points = gpd.GeoDataFrame(points_table, geometry=gpd.points_from_xy(points_table.longitude, points_table.latitude), crs="EPSG:4326").to_crs(raster_crs)
    xy = np.column_stack([points.geometry.x, points.geometry.y])
    observed = points_table["Water Distance (cm)"].to_numpy(float)
    predicted = []
    for index, point in enumerate(xy):
        keep = np.arange(len(xy)) != index
        distance = np.hypot(xy[keep, 0] - point[0], xy[keep, 1] - point[1])
        weights = 1 / np.maximum(distance, 1.0) ** 2
        predicted.append(np.sum(weights * observed[keep]) / np.sum(weights))
    errors = np.asarray(predicted) - observed
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def make_satellite_figure(values: np.ndarray, transform, crs, selected: gpd.GeoDataFrame, title: str, unit: str, sensors: pd.DataFrame | None = None, colorscale=None, zmax: float | None = None) -> go.Figure:
    """Draw the interpolated surface over Esri World Imagery without a Mapbox token."""
    height, width = values.shape
    stride = max(1, int(np.ceil(max(height, width) / 220)))
    rows, columns = np.indices(values.shape)
    rows, columns, sampled_values = rows[::stride, ::stride], columns[::stride, ::stride], values[::stride, ::stride]
    valid = np.isfinite(sampled_values)
    x_coordinates = transform.c + transform.a * (columns[valid] + 0.5)
    y_coordinates = transform.f + transform.e * (rows[valid] + 0.5)
    longitudes, latitudes = reproject_coordinates(crs, "EPSG:4326", x_coordinates.tolist(), y_coordinates.tolist())
    longitudes, latitudes = list(longitudes), list(latitudes)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
    if zmax is not None:
        low, high = 0, zmax
    if low == high:
        high = low + 1
    fig = go.Figure(go.Scattermap(
        lon=longitudes, lat=latitudes, mode="markers",
        marker={"size": max(4, 9 * stride), "color": sampled_values[valid], "colorscale": colorscale or WATER_COLORS, "cmin": low, "cmax": high, "opacity": 0.62, "colorbar": {"title": unit, "thickness": 18}},
        hovertemplate="Distance: %{marker.color:.1f} " + unit + "<extra></extra>", showlegend=False,
    ))
    boundary = selected.to_crs("EPSG:4326").geometry.union_all()
    polygons = [boundary] if boundary.geom_type == "Polygon" else list(boundary.geoms)
    for polygon in polygons:
        longitude, latitude = polygon.exterior.xy
        fig.add_trace(go.Scattermap(lon=list(longitude), lat=list(latitude), mode="lines", line={"color": "#ffffff", "width": 2}, hoverinfo="skip", showlegend=False))
    if sensors is not None and not sensors.empty:
        colors = sensors["AWD status"].map({"Sufficient: at/above soil surface": "#1579b5", "Watch: below soil surface": "#f0c419", "Irrigation needed": "#c0392b"}).fillna("#ffffff")
        fig.add_trace(go.Scattermap(lon=sensors.longitude, lat=sensors.latitude, mode="markers+text", text=[f"D{int(device)}" for device in sensors["Device"]], textposition="top center", textfont={"color": "white", "size": 11}, marker={"size": 12, "color": colors, "opacity": 1}, name="Sensor station"))
    centroid = selected.to_crs("EPSG:4326").geometry.union_all().centroid
    fig.update_layout(title=title, map={"style": "white-bg", "center": {"lon": centroid.x, "lat": centroid.y}, "zoom": 12, "layers": [{"below": "traces", "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], "sourceattribution": "Esri, Maxar, Earthstar Geographics"}]}, margin={"l": 5, "r": 5, "t": 48, "b": 5}, height=760)
    return fig


def make_figure(values: np.ndarray, transform, crs, selected: gpd.GeoDataFrame, title: str, unit: str, sensors: pd.DataFrame | None = None, colorscale=None, zmax: float | None = None, satellite: bool = False) -> go.Figure:
    if satellite:
        return make_satellite_figure(values, transform, crs, selected, title, unit, sensors, colorscale, zmax)
    height, width = values.shape
    x = np.linspace(transform.c + transform.a / 2, transform.c + transform.a * (width - 0.5), width)
    y = np.linspace(transform.f + transform.e / 2, transform.f + transform.e * (height - 0.5), height)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
    if zmax is not None:
        low, high = 0, zmax
    if low == high:
        high = low + 1
    fig = go.Figure(go.Heatmap(
        z=values, x=x, y=y, colorscale=colorscale or WATER_COLORS, zmin=low, zmax=high,
        colorbar={"title": unit, "thickness": 18}, hovertemplate="Distance: %{z:.1f} " + unit + "<extra></extra>",
        connectgaps=False,
    ))
    draw_boundary(fig, selected, crs)
    if sensors is not None and not sensors.empty:
        sensor_points = gpd.GeoDataFrame(sensors, geometry=gpd.points_from_xy(sensors.longitude, sensors.latitude), crs="EPSG:4326").to_crs(crs)
        labels = [f"D{int(device)}" for device in sensor_points["Device"]]
        places = sensor_points["place"] if "place" in sensor_points else pd.Series("", index=sensor_points.index)
        timestamps = sensor_points["Timestamp"] if "Timestamp" in sensor_points else pd.Series("Median observation", index=sensor_points.index)
        statuses = sensor_points["AWD status"] if "AWD status" in sensor_points else pd.Series("", index=sensor_points.index)
        station_colors = sensor_points["AWD status"].map({
            "Sufficient: at/above soil surface": "#1579b5",
            "Watch: below soil surface": "#f0c419",
            "Irrigation needed": "#c0392b",
        }).fillna("#ffffff")
        fig.add_trace(go.Scattergl(
            x=sensor_points.geometry.x, y=sensor_points.geometry.y, mode="markers+text", text=labels,
            textposition="top center", textfont={"color": "white", "size": 11},
            marker={"size": 13, "color": station_colors, "line": {"color": "#ffffff", "width": 1.5}},
            customdata=np.column_stack([places, sensor_points["Water Distance (cm)"], sensor_points.get("AWD water deficit (cm)", pd.Series(np.nan, index=sensor_points.index)), statuses, timestamps]),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Raw sensor distance: %{customdata[1]:.2f} cm<br>AWD deficit below soil: %{customdata[2]:.2f} cm<br>Status: %{customdata[3]}<br>Observed: %{customdata[4]}<extra></extra>",
            name="Sensor station",
        ))
    fig.update_layout(
        title=title, template="plotly_dark", paper_bgcolor="#202020", plot_bgcolor="#202020",
        margin={"l": 5, "r": 5, "t": 48, "b": 5}, height=760,
        xaxis={"visible": False, "scaleanchor": "y"}, yaxis={"visible": False},
    )
    return fig


def read_sheet(upload) -> pd.DataFrame:
    return pd.read_excel(upload) if upload.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(upload)


@st.cache_data(ttl=900, show_spinner="Loading the public read-only spreadsheet…")
def load_public_sheet_raw() -> pd.DataFrame:
    """Fetch the sheet's published CSV snapshot; this is not a Sheets API call."""
    return pd.read_csv(PUBLIC_SHEET_CSV)


def clean_water_distance(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only physically valid water-distance readings in the requested range."""
    data = table.copy()
    data["Water Distance (cm)"] = pd.to_numeric(data["Water Distance (cm)"], errors="coerce")
    return data[data["Water Distance (cm)"].gt(0) & data["Water Distance (cm)"].le(60)].copy()


def load_public_sheet() -> pd.DataFrame:
    return clean_water_distance(load_public_sheet_raw())


@st.cache_data(ttl=1800, show_spinner=False)
def load_open_meteo_forecast(latitude: float, longitude: float, days: int) -> pd.DataFrame:
    """Load a short, local daily forecast from Open-Meteo without credentials."""
    query = urlencode({
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,et0_fao_evapotranspiration,wind_speed_10m_max,weather_code",
        "timezone": "Asia/Manila",
        "forecast_days": days,
    })
    with urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=15) as response:
        payload = json.load(response)
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        raise ValueError("Open-Meteo returned no daily forecast data.")
    forecast = pd.DataFrame(daily).rename(columns={
        "time": "Date",
        "temperature_2m_max": "Max temperature (C)",
        "temperature_2m_min": "Min temperature (C)",
        "precipitation_sum": "Rainfall (mm)",
        "precipitation_probability_max": "Rain probability (%)",
        "et0_fao_evapotranspiration": "Reference ET (mm)",
        "wind_speed_10m_max": "Max wind speed (km/h)",
        "weather_code": "Weather code",
    })
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    return forecast


def weather_condition(code: int) -> str:
    conditions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
        80: "Rain showers", 81: "Heavy showers", 82: "Violent showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm",
    }
    return conditions.get(int(code), "Unclassified conditions")


def show_sheet_timeseries(table: pd.DataFrame) -> None:
    timestamp_column = "Timestamp"
    distance_column = "Water Distance (cm)"
    device_column = "Device"
    required = {timestamp_column, distance_column}
    if not required.issubset(table.columns):
        st.error("The public sheet does not contain the expected Timestamp and Water Distance (cm) columns.")
        return
    data = table.copy()
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
    data[distance_column] = pd.to_numeric(data[distance_column], errors="coerce")
    data = data.dropna(subset=[timestamp_column, distance_column]).sort_values(timestamp_column)
    if data.empty:
        st.warning("The sheet has no valid water-distance measurements.")
        return
    devices = sorted(data[device_column].dropna().unique().tolist()) if device_column in data else []
    chosen_devices = st.sidebar.multiselect("Device", devices, default=devices) if devices else []
    if chosen_devices:
        data = data[data[device_column].isin(chosen_devices)]
    latest = data.iloc[-1]
    first, second, third = st.columns(3)
    first.metric("Latest water distance", f"{latest[distance_column]:.2f} cm")
    second.metric("Lowest recorded", f"{data[distance_column].min():.2f} cm")
    third.metric("Measurements", f"{len(data):,}")
    color = device_column if device_column in data else None
    fig = go.Figure()
    for device, group in data.groupby(device_column) if color else [("Measurements", data)]:
        fig.add_trace(go.Scattergl(
            x=group[timestamp_column], y=group[distance_column], mode="lines+markers", name=f"Device {device}",
            line={"color": "#00b6df"}, marker={"size": 4},
            hovertemplate="%{x}<br>Water distance: %{y:.2f} cm<extra></extra>",
        ))
    fig.update_layout(
        title="Water distance recorded by the public sheet", template="plotly_dark",
        paper_bgcolor="#202020", plot_bgcolor="#202020", height=560,
        xaxis={"title": "Timestamp", "rangeslider": {"visible": True}},
        yaxis={"title": "Water distance (cm)"}, margin={"l": 5, "r": 5, "t": 48, "b": 5},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download cleaned water-distance dataset", data=data.to_csv(index=False), file_name="cleaned_google_sheet_water_distance.csv", mime="text/csv")
    st.info("The shared sheet currently supplies timestamps, device readings, and water distance, but no longitude, latitude, or place field. It can be explored by time and device, but cannot truthfully be drawn as a place-adjusted map until location data is added.")


def show_iot_conditions(table: pd.DataFrame) -> None:
    """Explore the environmental and soil observations paired with clean AWD readings."""
    data = table.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    available = [column for column in IOT_FACTORS if column in data.columns]
    for column in ["Water Distance (cm)", *available]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Timestamp", "Water Distance (cm)"]).sort_values("Timestamp")
    metric = st.sidebar.selectbox("IoT factor to inspect", available, index=0)
    devices = sorted(data["Device"].dropna().unique().tolist())
    selected_devices = st.sidebar.multiselect("Devices", devices, default=devices, key="iot_devices")
    if selected_devices:
        data = data[data["Device"].isin(selected_devices)]
    st.subheader("Latest field readings by station")
    latest = data.sort_values("Timestamp").groupby("Device", as_index=False).tail(1)
    latest_columns = ["Device", "Timestamp", "Water Distance (cm)", *available]
    st.dataframe(latest[latest_columns].sort_values("Device").round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "Download latest station readings",
        data=latest[latest_columns].sort_values("Device").to_csv(index=False),
        file_name="latest_iot_station_readings.csv", mime="text/csv", key="download_latest_iot",
    )
    st.subheader("Descriptive environmental statistics")
    descriptive_columns = ["Water Distance (cm)", *available]
    descriptive = data[descriptive_columns].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    descriptive.index.name = "Sensor variable"
    selected_stats = descriptive.loc[metric]
    stat_one, stat_two, stat_three, stat_four = st.columns(4)
    stat_one.metric("Average", f"{selected_stats['mean']:.2f}")
    stat_two.metric("Typical value (median)", f"{selected_stats['50%']:.2f}")
    stat_three.metric("Lowest recorded", f"{selected_stats['min']:.2f}")
    stat_four.metric("Highest recorded", f"{selected_stats['max']:.2f}")
    histogram_column, box_column = st.columns(2)
    histogram = go.Figure(go.Histogram(x=data[metric].dropna(), nbinsx=30, marker={"color": "#4f8a3c"}))
    histogram.update_layout(title=f"How often each {metric} value occurs", template="plotly_white", height=360, xaxis_title=metric, yaxis_title="Number of readings", margin={"l": 5, "r": 5, "t": 48, "b": 5})
    with histogram_column:
        st.plotly_chart(histogram, use_container_width=True)
    station_box = go.Figure()
    for device, group in data.dropna(subset=[metric]).groupby("Device"):
        station_box.add_trace(go.Box(y=group[metric], name=f"D{int(device)}", boxmean=True, marker_color="#c7992d"))
    station_box.update_layout(title=f"{metric} range by station", template="plotly_white", height=360, yaxis_title=metric, xaxis_title="Sensor station", margin={"l": 5, "r": 5, "t": 48, "b": 5}, showlegend=False)
    with box_column:
        st.plotly_chart(station_box, use_container_width=True)
    with st.expander("Show numeric descriptive statistics"):
        st.dataframe(descriptive, use_container_width=True)
        st.caption("Statistics use cleaned valid water-distance records and the selected devices. `50%` is the median; `std` is standard deviation.")
    chart = go.Figure()
    for device, group in data.dropna(subset=[metric]).groupby("Device"):
        chart.add_trace(go.Scattergl(
            x=group["Timestamp"], y=group[metric], mode="lines", name=f"D{int(device)}",
            hovertemplate="%{x}<br>" + metric + ": %{y:.2f}<extra></extra>",
        ))
    chart.update_layout(
        title=f"{metric} by device", template="plotly_dark", paper_bgcolor="#202020", plot_bgcolor="#202020",
        height=540, xaxis={"title": "Timestamp", "rangeslider": {"visible": True}}, yaxis={"title": metric},
        margin={"l": 5, "r": 5, "t": 48, "b": 5},
    )
    st.plotly_chart(chart, use_container_width=True)
    correlations = data[["Water Distance (cm)", *available]].corr(numeric_only=True)["Water Distance (cm)"].drop("Water Distance (cm)").sort_values(key=np.abs, ascending=False)
    st.subheader("Association with water-distance observations")
    st.dataframe(correlations.rename("Pearson correlation").to_frame(), use_container_width=True)
    st.caption("These correlations describe co-occurrence in the recorded data; they do not prove that an IoT factor causes water deficit. AWD sufficiency status continues to use the confirmed 20 cm soil-surface calibration.")


def water_distance_by_device(table: pd.DataFrame, use_latest: bool, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Summarize valid sheet readings into one actual distance value per device."""
    data = table[["Timestamp", "Device", "Water Distance (cm)"]].copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data["Water Distance (cm)"] = pd.to_numeric(data["Water Distance (cm)"], errors="coerce")
    data = data.dropna().sort_values("Timestamp")
    if as_of is not None:
        data = data[data["Timestamp"] <= as_of]
    if use_latest:
        return data.groupby("Device", as_index=False).tail(1)
    return data.groupby("Device", as_index=False)["Water Distance (cm)"].median()


def prepare_water_distance_points(locations_upload, use_latest: bool, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    locations = pd.read_csv(locations_upload)
    required = {"Device", "longitude", "latitude"}
    if not required.issubset(locations.columns):
        raise ValueError("The device-location file must contain Device, longitude, and latitude columns.")
    locations["longitude"] = pd.to_numeric(locations["longitude"], errors="coerce")
    locations["latitude"] = pd.to_numeric(locations["latitude"], errors="coerce")
    locations = locations.dropna(subset=["Device", "longitude", "latitude"])
    merged = locations.merge(water_distance_by_device(load_public_sheet(), use_latest, as_of), on="Device", how="inner")
    if len(merged) < 3:
        raise ValueError("At least three located devices with water-distance readings are required for a surface map.")
    return merged


def apply_awd_status(points: pd.DataFrame, irrigation_trigger_cm: float) -> pd.DataFrame:
    """Apply AWD status with a non-negative deficit below the soil surface."""
    data = points.copy()
    raw = data["Water Distance (cm)"]
    data["AWD water deficit (cm)"] = np.maximum(raw - SOIL_SURFACE_READING_CM, 0)
    data["AWD status"] = np.select(
        [raw <= SOIL_SURFACE_READING_CM, raw <= irrigation_trigger_cm],
        ["Sufficient: at/above soil surface", "Watch: below soil surface"],
        default="Irrigation needed",
    )
    return data


def recent_drawdown_rate(table: pd.DataFrame, device: int | float) -> float:
    """Estimate the sensor's recent daily change in raw water distance."""
    history = table[table["Device"] == device][["Timestamp", "Water Distance (cm)"]].copy()
    history["Timestamp"] = pd.to_datetime(history["Timestamp"], errors="coerce")
    history["Water Distance (cm)"] = pd.to_numeric(history["Water Distance (cm)"], errors="coerce")
    history = history.dropna().sort_values("Timestamp").tail(30)
    if len(history) < 2:
        return 0.5
    elapsed_days = (history["Timestamp"] - history["Timestamp"].iloc[0]).dt.total_seconds() / 86400
    if elapsed_days.iloc[-1] <= 0:
        return 0.5
    rate = np.polyfit(elapsed_days, history["Water Distance (cm)"], 1)[0]
    return float(np.clip(rate, -2.0, 6.0))


def build_irrigation_plan(
    readings: pd.DataFrame,
    locations_source,
    irrigation_trigger_cm: float,
    forecast_days: int,
    season: str = "Wet Season",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine latest IoT data, its trend, and local weather into a station plan."""
    locations = pd.read_csv(locations_source)
    required = {"Device", "longitude", "latitude"}
    if not required.issubset(locations.columns):
        raise ValueError("The device-location file must contain Device, longitude, and latitude columns.")
    locations["longitude"] = pd.to_numeric(locations["longitude"], errors="coerce")
    locations["latitude"] = pd.to_numeric(locations["latitude"], errors="coerce")
    latest = water_distance_by_device(readings, use_latest=True)
    stations = locations.dropna(subset=["Device", "longitude", "latitude"]).merge(latest, on="Device", how="inner")
    if stations.empty:
        raise ValueError("No IoT readings could be matched to the device locations.")

    # Seasonal model parameters (PhilRice / IRRI AWD adaptation)
    is_dry = season == "Dry Season"
    et_factor = 0.20 if is_dry else 0.15
    rain_factor = 0.04 if is_dry else 0.06
    min_rain_deferral = 15.0 if is_dry else 8.0
    min_prob_deferral = 60.0 if is_dry else 50.0

    station_rows, forecast_rows = [], []
    for _, station in stations.iterrows():
        weather = load_open_meteo_forecast(float(station["latitude"]), float(station["longitude"]), forecast_days).copy()
        trend = recent_drawdown_rate(readings, station["Device"])
        # Rainfall and ET adjust the observed trend according to seasonal parameters.
        weather["Projected daily change (cm)"] = np.clip(
            trend + et_factor * weather["Reference ET (mm)"] - rain_factor * weather["Rainfall (mm)"], -3.0, 8.0
        )
        weather["Projected water distance (cm)"] = float(station["Water Distance (cm)"]) + weather["Projected daily change (cm)"].cumsum()
        weather["Device"] = station["Device"]
        weather["Place"] = station.get("place", "Unspecified")
        weather["Weather"] = weather["Weather code"].map(weather_condition)
        forecast_rows.append(weather)

        threshold_dates = weather.loc[weather["Projected water distance (cm)"] > irrigation_trigger_cm, "Date"]
        threshold_date = threshold_dates.iloc[0] if not threshold_dates.empty else pd.NaT
        first_day_rain = float(weather["Rainfall (mm)"].iloc[0])
        first_day_probability = float(weather["Rain probability (%)"].iloc[0])
        raw_distance = float(station["Water Distance (cm)"])
        rain_deferral = first_day_rain >= min_rain_deferral and first_day_probability >= min_prob_deferral
        if raw_distance > irrigation_trigger_cm and not rain_deferral:
            action = "Irrigate today"
        elif raw_distance > irrigation_trigger_cm:
            action = "Hold for forecast rain"
        elif pd.notna(threshold_date) and (threshold_date - weather["Date"].iloc[0]).days <= 2:
            action = "Schedule within 48 h"
        else:
            action = "Monitor"
        station_rows.append({
            "Device": station["Device"],
            "Place": station.get("place", "Unspecified"),
            "Longitude": station["longitude"],
            "Latitude": station["latitude"],
            "Last observation": station["Timestamp"],
            "Current water distance (cm)": raw_distance,
            "Recent drawdown (cm/day)": trend,
            "Rain next 24 h (mm)": first_day_rain,
            "Rain probability next 24 h (%)": first_day_probability,
            "Threshold date": threshold_date,
            "Recommended action": action,
        })
    return pd.DataFrame(station_rows), pd.concat(forecast_rows, ignore_index=True)


def make_priority_map(plan: pd.DataFrame) -> go.Figure:
    """Map irrigation priority so the dispatcher can scan locations first."""
    colors = {
        "Irrigate today": "#c0392b",
        "Schedule within 48 h": "#e67e22",
        "Hold for forecast rain": "#2874a6",
        "Monitor": "#2e8b57",
    }
    figure = go.Figure()
    for action, color in colors.items():
        stations = plan[plan["Recommended action"] == action]
        if stations.empty:
            continue
        figure.add_trace(go.Scattermap(
            lon=stations["Longitude"], lat=stations["Latitude"], mode="markers+text",
            text=[f"D{int(device)}" for device in stations["Device"]], textposition="top center",
            textfont={"color": "#ffffff", "size": 11}, name=action,
            marker={"size": 17, "color": color, "opacity": 0.95},
            customdata=np.column_stack([
                stations["Place"], stations["Current water distance (cm)"],
                stations["Recent drawdown (cm/day)"], stations["Rain next 24 h (mm)"],
            ]),
            hovertemplate=("<b>%{text}</b><br>%{customdata[0]}<br>Action: " + action +
                           "<br>Water distance: %{customdata[1]:.1f} cm<br>Drawdown: %{customdata[2]:.2f} cm/day" +
                           "<br>Rain next 24 h: %{customdata[3]:.1f} mm<extra></extra>"),
        ))
    figure.update_layout(
        title="Irrigation priority by sensor station", height=500,
        margin={"l": 5, "r": 5, "t": 48, "b": 5}, legend={"orientation": "h", "y": 1.02},
        map={
            "style": "white-bg",
            "center": {"lon": plan["Longitude"].mean(), "lat": plan["Latitude"].mean()},
            "zoom": 11,
            "layers": [{"below": "traces", "sourcetype": "raster", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], "sourceattribution": "Esri, Maxar, Earthstar Geographics"}],
        },
    )
    return figure


def make_district_weather_figure(forecast: pd.DataFrame) -> go.Figure:
    """Summarize weather pressure across all configured sensor locations."""
    daily = forecast.groupby("Date", as_index=False).agg({
        "Rainfall (mm)": "mean", "Rain probability (%)": "mean", "Reference ET (mm)": "mean",
        "Max temperature (C)": "mean", "Max wind speed (km/h)": "mean",
    })
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=daily["Date"], y=daily["Rainfall (mm)"], name="Rainfall",
        marker_color="#2d82b7", hovertemplate="%{x|%d %b}<br>Rainfall: %{y:.1f} mm<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Reference ET (mm)"], name="ET0",
        mode="lines+markers", line={"color": "#d06b23", "width": 3},
        hovertemplate="%{x|%d %b}<br>ET0: %{y:.1f} mm<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_white", height=360, showlegend=True,
        margin={"l": 56, "r": 12, "t": 12, "b": 38},
        legend={"orientation": "h", "x": 0, "y": 1, "xanchor": "left", "yanchor": "bottom"},
        xaxis={"title": None, "tickformat": "%d %b", "showgrid": False},
        yaxis={"title": "Daily water (mm)", "rangemode": "tozero", "gridcolor": "#e4e8e4"},
    )
    return figure


def show_forecast_advisory(
    locations_source,
    irrigation_trigger_cm: float,
    forecast_days: int,
    field_area_ha: float,
    delivery_depth_mm: float,
    season: str = "Wet Season",
) -> None:
    plan, forecast = build_irrigation_plan(
        load_public_sheet(), locations_source, irrigation_trigger_cm, forecast_days, season
    )
    urgent_actions = plan["Recommended action"].isin(["Irrigate today", "Schedule within 48 h"])
    rainfall_holds = plan["Recommended action"].eq("Hold for forecast rain")
    delivery_volume = urgent_actions.sum() * field_area_ha * 10_000 * delivery_depth_mm / 1000
    with st.container(horizontal=True):
        st.metric("Stations assessed", len(plan), border=True)
        st.metric("Urgent or due within 48 h", int(urgent_actions.sum()), border=True)
        st.metric("Held for forecast rain", int(rainfall_holds.sum()), border=True)
        st.metric("Indicative delivery volume", f"{delivery_volume:,.0f} m3", border=True)
    st.caption(
        f"Volume scenario assumes {field_area_ha:.1f} ha per urgent station and a {delivery_depth_mm:.0f} mm application. "
        "Use it for canal scheduling, not as a measured field-volume estimate."
    )
    map_column, weather_column = st.columns(2)
    with map_column:
        st.plotly_chart(make_priority_map(plan), width="stretch")
    with weather_column:
        st.subheader("District water-balance outlook")
        st.caption("Average forecast across sensor locations. Rainfall offsets field drawdown; ET0 indicates atmospheric water demand.")
        st.plotly_chart(make_district_weather_figure(forecast), width="stretch")
    action_order = {"Irrigate today": 0, "Schedule within 48 h": 1, "Hold for forecast rain": 2, "Monitor": 3}
    plan["_priority"] = plan["Recommended action"].map(action_order)
    st.subheader("Station action queue")
    st.dataframe(
        plan.sort_values(["_priority", "Current water distance (cm)"], ascending=[True, False]).drop(columns="_priority").round(2),
        hide_index=True,
        width="stretch",
        column_config={
            "Last observation": st.column_config.DatetimeColumn(format="D MMM YYYY, h:mm a"),
            "Threshold date": st.column_config.DateColumn(format="D MMM YYYY"),
            "Longitude": st.column_config.NumberColumn(format="%.5f"),
            "Latitude": st.column_config.NumberColumn(format="%.5f"),
        },
    )

    devices = sorted(plan["Device"].unique().tolist())
    selected_device = st.selectbox("Station forecast detail", devices, format_func=lambda device: f"Device {int(device)}")
    station_forecast = forecast[forecast["Device"] == selected_device].copy()
    station_plan = plan[plan["Device"] == selected_device].iloc[0]
    st.subheader(f"Device {int(selected_device)} water-distance forecast")

    # Current Status Summary Cards (Requirements 7, 8, 9)
    with st.container(horizontal=True):
        st.metric("Recommended action", station_plan["Recommended action"], border=True)
        st.metric("Current water distance", f"{station_plan['Current water distance (cm)']:.1f} cm", border=True)
        st.metric("Recent drawdown", f"{station_plan['Recent drawdown (cm/day)']:.2f} cm/day", border=True)
        crossing_str = station_plan["Threshold date"].strftime("%d %b %Y") if pd.notna(station_plan["Threshold date"]) else "No crossing in forecast"
        st.metric("Projected trigger crossing", crossing_str, border=True)

    # Calculate optimal Y-axis range for Water Distance (Requirement 5)
    min_wd = min(station_forecast["Projected water distance (cm)"].min(), irrigation_trigger_cm, station_plan["Current water distance (cm)"])
    max_wd = max(station_forecast["Projected water distance (cm)"].max(), irrigation_trigger_cm, station_plan["Current water distance (cm)"])
    padding = max((max_wd - min_wd) * 0.15, 2.0)
    y1_range = [min_wd - padding, max_wd + padding]

    # Redesigned Subplot: Main Water Distance Chart + Compact Bottom Rainfall Strip (Requirements 1, 2, 4, 5, 6, 9, 10, 11, 12)
    forecast_chart = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
    )

    # 1. Primary Visualization: Projected Water Distance
    forecast_chart.add_trace(
        go.Scatter(
            x=station_forecast["Date"],
            y=station_forecast["Projected water distance (cm)"],
            name="Projected water distance",
            mode="lines+markers",
            line={"color": "#1f77b4", "width": 3.5},
            marker={"size": 7, "color": "#1f77b4", "symbol": "circle"},
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Projected water distance: <b>%{y:.1f} cm</b><extra></extra>",
        ),
        row=1, col=1,
    )

    # 2. Irrigation Trigger Reference Line
    forecast_chart.add_hline(
        y=irrigation_trigger_cm,
        line_dash="dash",
        line_color="#e67e22",
        line_width=2,
        annotation_text=f"Irrigation trigger ({irrigation_trigger_cm:.0f} cm)",
        annotation_position="top right",
        annotation_font_color="#e67e22",
        annotation_yshift=6,
        row=1,
    )

    # 4. Secondary Visualization: Compact Bottom Rainfall Strip
    forecast_chart.add_trace(
        go.Bar(
            x=station_forecast["Date"],
            y=station_forecast["Rainfall (mm)"],
            name="Rainfall (mm)",
            marker_color="#5bc0de",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast rainfall: <b>%{y:.1f} mm</b><extra></extra>",
        ),
        row=2, col=1,
    )

    # Configure Axes, Margins, and Legend (Requirements 5, 6, 11, 12)
    forecast_chart.update_yaxes(
        title_text="Water distance (cm)",
        range=y1_range,
        gridcolor="#e9ecef",
        row=1, col=1,
    )
    forecast_chart.update_yaxes(
        title_text="Rainfall (mm)",
        rangemode="tozero",
        gridcolor="#e9ecef",
        row=2, col=1,
    )
    forecast_chart.update_xaxes(
        tickformat="%d %b",
        showgrid=True,
        gridcolor="#f0f0f0",
        row=2, col=1,
    )
    forecast_chart.update_xaxes(
        showgrid=True,
        gridcolor="#f0f0f0",
        row=1, col=1,
    )
    forecast_chart.update_layout(
        template="plotly_white",
        height=480,
        margin={"l": 60, "r": 30, "t": 40, "b": 35},
        legend={"orientation": "h", "x": 0, "y": 1.1, "xanchor": "left", "yanchor": "bottom"},
    )
    st.plotly_chart(forecast_chart, width="stretch")
    weather_table = station_forecast[["Date", "Weather", "Min temperature (C)", "Max temperature (C)", "Rainfall (mm)", "Rain probability (%)", "Reference ET (mm)", "Max wind speed (km/h)", "Projected water distance (cm)"]]
    st.subheader("Daily weather and water-distance outlook")
    st.dataframe(weather_table.round(2), hide_index=True, width="stretch")
    st.caption("Forecast source: Open-Meteo. Water-distance projection uses each station's recent trend, adjusted by forecast reference ET and rainfall. Validate recommendations against field conditions and canal availability.")


def main() -> None:
    st.set_page_config(page_title="Midsayap Water Distance", page_icon="💧", layout="wide")
    styles = """
        <style>
        .stApp { background: #f4f0e4; color: #213b26; }
        [data-testid='stSidebar'] { background: #e5eee4; }
        [data-testid='stSidebar'] * { color: #213b26 !important; }
        [data-testid='stSidebar'] .react-aria-ComboBox > div,
        [data-testid='stSidebar'] input,
        [data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'],
        [data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'] button { background: #fffdf6 !important; color: #213b26 !important; }
        h1, h2, h3 { color: #245b35; font-family: Georgia, serif; }
        [data-testid='stMetric'] { background: #fffdf6; border: 1px solid #d8d1b9; border-radius: 12px; padding: 12px; }
        [data-testid='stMetric'] * { color: #213b26 !important; }
        .rice-hero { background: linear-gradient(115deg, #1f5d37, #689f38); padding: 22px 28px; border-radius: 16px; color: white; margin-bottom: 14px; }
        .rice-hero h1 { color: white; margin: 0; font-family: Georgia, serif; }
        .rice-hero p { margin: 5px 0 0; color: #eef5db; }
        @media (prefers-color-scheme: dark) {
            .stApp { background: #101a13; color: #edf4df; }
            [data-testid='stSidebar'] { background: #172d21; }
            [data-testid='stSidebar'] * { color: #edf4df !important; }
            [data-testid='stSidebar'] .react-aria-ComboBox > div,
            [data-testid='stSidebar'] input,
            [data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'],
            [data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'] button { background: #1f3728 !important; border-color: #8cbf51 !important; color: #edf4df !important; }
            h1, h2, h3 { color: #b8d77b; }
            [data-testid='stMetric'] { background: #1c3023; border-color: #5f8e49; }
            [data-testid='stMetric'] * { color: #edf4df !important; }
            .rice-hero { background: linear-gradient(115deg, #245d36, #76ad3f); }
        }
        </style>
    """
    boundaries = load_boundaries()
    place_column = choose_place_column(boundaries)
    labels = sorted(boundaries[place_column].fillna("Unspecified").astype(str).unique())
    with st.sidebar:
        st.header("Controls")
        cropping_season = st.radio(
            "Cropping season",
            ["Wet Season (May – Oct)", "Dry Season (Nov – Apr)"],
            index=0,
            help="Adapts irrigation trigger defaults, ET0 evapotranspiration weights, and rainfall deferral thresholds for Philippines rice crop cycles.",
        )
        season = "Wet Season" if "Wet" in cropping_season else "Dry Season"
        default_trigger = 35.0 if season == "Wet Season" else 30.0

    st.markdown(styles + f"<div class='rice-hero'><h1>🌾 Midsayap Rice Water Advisory</h1><p>{season} AWD monitoring for field-level irrigation decisions</p></div>", unsafe_allow_html=True)
    st.caption("Decision support for rice growers and irrigation managers • live public sensor data")

    with st.sidebar:
        place = st.selectbox("Adjust view by place", ["All Midsayap"] + labels)
        source = st.radio("Workspace", ["Forecast-aware irrigation plan", "Water distance map (Google Sheet)", "IoT conditions and relationships", "Terrain surface (reference style)", "Public Google Sheet (read-only)", "Distance raster (.tif)", "Exported Google Sheet (CSV/XLSX)"], index=0)
        unit = st.text_input("Distance unit", "metres")
        upload = None
        forecast_locations_upload = None
        if source == "Forecast-aware irrigation plan":
            st.caption(f"Combines latest IoT water distance with Open-Meteo's daily forecast adapted for {season}. Forecasts refresh every 30 minutes.")
            forecast_locations_upload = st.file_uploader("Upload device locations CSV", type=["csv"], key="forecast_locations")
            forecast_days = st.slider("Forecast horizon (days)", 3, 14, 7)
            forecast_trigger = st.number_input(f"{season} irrigation trigger (raw sensor cm)", min_value=SOIL_SURFACE_READING_CM, max_value=60.0, value=default_trigger, step=1.0, key="forecast_trigger")
            field_area_ha = st.number_input("Representative area per station (ha)", min_value=0.1, max_value=500.0, value=5.0, step=0.5)
            delivery_depth_mm = st.number_input("Planned application depth (mm)", min_value=1.0, max_value=200.0, value=30.0, step=5.0)
            if forecast_locations_upload is None and DEVICE_LOCATIONS_FILE.exists():
                st.success("Using the verified local coordinates for Devices 1–9.")
        elif source == "Distance raster (.tif)":
            upload = st.file_uploader("Upload a local distance raster", type=["tif", "tiff"])
            if DISTANCE_RASTERS:
                st.caption("Local distance raster found: " + DISTANCE_RASTERS[0].name)
        elif source == "Exported Google Sheet (CSV/XLSX)":
            upload = st.file_uploader("Upload an exported sheet", type=["csv", "xlsx", "xls"])
            grid_size = st.slider("Interpolation detail", 180, 900, 500, 20)
        elif source == "Water distance map (Google Sheet)":
            st.caption("Uses the public sheet's Water Distance (cm) values; coordinates remain local.")
            locations_upload = st.file_uploader("Upload device locations CSV", type=["csv"])
            sheet_times = pd.to_datetime(load_public_sheet()["Timestamp"], errors="coerce").dropna()
            selected_time = st.slider("Observation cutoff time", min_value=sheet_times.min().to_pydatetime(), max_value=sheet_times.max().to_pydatetime(), value=sheet_times.max().to_pydatetime(), format="YYYY-MM-DD HH:mm")
            distance_statistic = st.radio("Measurement used", ["Latest reading per device", "Median reading per device"])
            spatial_model = st.selectbox("Spatial model", ["Inverse-distance weighting (recommended)", "Linear interpolation"])
            irrigation_trigger = st.number_input(f"{season} irrigation trigger (raw sensor cm)", min_value=SOIL_SURFACE_READING_CM, max_value=60.0, value=default_trigger, step=1.0, help="20 cm is the confirmed soil-surface reading. Values above this trigger are classified as irrigation needed.")
            map_layer = st.radio("Map layer", [f"AWD {season.lower()} water sufficiency", "Water-distance estimate", "Model confidence (nearest sensor)"])
            satellite_mode = st.checkbox("Satellite imagery basemap", value=False, help="Uses Esri World Imagery while keeping the water-distance layer and station markers visible. Requires an internet connection.")
            grid_size = st.slider("Interpolation detail", 180, 900, 500, 20)
            if locations_upload is None and not DEVICE_LOCATIONS_FILE.exists():
                st.download_button(
                    "Download device locations template",
                    data=(ROOT / "device_locations_template.csv").read_bytes(),
                    file_name="device_locations.csv",
                    mime="text/csv",
                )
            elif locations_upload is None:
                st.success("Using the verified local coordinates for Devices 1–9.")

    selected = boundaries if place == "All Midsayap" else boundaries[boundaries[place_column].fillna("Unspecified").astype(str) == place]
    if selected.empty:
        st.error("No boundary matches the selected place.")
        st.stop()

    try:
        if source == "Forecast-aware irrigation plan":
            location_source = forecast_locations_upload or DEVICE_LOCATIONS_FILE
            if location_source is None or (isinstance(location_source, Path) and not location_source.exists()):
                st.error("Upload a device-locations CSV with Device, longitude, and latitude columns to create the forecast plan.")
                st.stop()
            show_forecast_advisory(location_source, forecast_trigger, forecast_days, field_area_ha, delivery_depth_mm, season=season)
            return
        if source == "Public Google Sheet (read-only)":
            show_sheet_timeseries(load_public_sheet())
            return
        if source == "IoT conditions and relationships":
            show_iot_conditions(load_public_sheet())
            return
        if source == "Water distance map (Google Sheet)":
            location_source = locations_upload or DEVICE_LOCATIONS_FILE
            if location_source is None or (isinstance(location_source, Path) and not location_source.exists()):
                st.info("Download the template, enter the latitude and longitude for each sensor device, then upload it here. This prevents the map from assigning made-up locations to real readings.")
                st.stop()
            points = apply_awd_status(prepare_water_distance_points(location_source, distance_statistic == "Latest reading per device", pd.Timestamp(selected_time)), irrigation_trigger)
            crs = boundaries.crs
            mae, rmse = validate_idw(points, crs)
            metric_one, metric_two, metric_three = st.columns(3)
            metric_one.metric("Stations used", len(points))
            metric_two.metric("IDW validation MAE", f"{mae:.2f} cm")
            metric_three.metric("IDW validation RMSE", f"{rmse:.2f} cm")
            if map_layer == "Model confidence (nearest sensor)":
                values, transform = nearest_sensor_distance(points, selected, crs, grid_size)
                title, unit = f"Model confidence — distance to nearest sensor — {place}", "metres"
            elif spatial_model == "Inverse-distance weighting (recommended)":
                values, transform = interpolate_idw(points, "longitude", "latitude", "Water Distance (cm)", selected, crs, grid_size)
                model_label = "IDW model"
            else:
                values, transform = interpolate_sheet(points, "longitude", "latitude", "Water Distance (cm)", selected, crs, grid_size)
                model_label = "linear model"
            if map_layer == "AWD wet-season water sufficiency":
                raw_reading = values.copy()
                values = np.maximum(raw_reading - SOIL_SURFACE_READING_CM, 0)
                valid = raw_reading[np.isfinite(raw_reading)]
                sufficient = np.mean(valid <= SOIL_SURFACE_READING_CM) * 100
                watch = np.mean((valid > SOIL_SURFACE_READING_CM) & (valid <= irrigation_trigger)) * 100
                needed = np.mean(valid > irrigation_trigger) * 100
                status_one, status_two, status_three = st.columns(3)
                status_one.metric("Sufficient / at soil", f"{sufficient:.1f}%")
                status_two.metric("Watch", f"{watch:.1f}%")
                status_three.metric("Irrigation needed", f"{needed:.1f}%")
                title, unit = f"AWD water deficit below soil surface — {place}", "cm"
                if needed > 0:
                    st.warning(f"Irrigation advisory: {needed:.1f}% of the modelled area is beyond the selected wet-season trigger. Check the red/orange zones and the matching station markers first.")
                else:
                    st.success("Field advisory: no modelled area is beyond the selected irrigation trigger. Continue routine AWD monitoring.")
                st.markdown("**Rice AWD guide:** 🔵 water at/above soil surface &nbsp; • &nbsp; 🟢 small drawdown—monitor &nbsp; • &nbsp; 🟡 approaching trigger &nbsp; • &nbsp; 🟠🔴 irrigation needed")
            elif map_layer == "Water-distance estimate":
                title, unit = f"Water distance ({model_label}) — {place}", "cm"
            st.caption(f"AWD calibration: raw reading {SOIL_SURFACE_READING_CM:.0f} cm = soil surface; AWD water deficit = max(raw reading − {SOIL_SURFACE_READING_CM:.0f}, 0). In the sufficiency layer, blue means at/near the soil surface and warmer colours mean a larger estimated deficit. Irrigation-needed threshold: raw reading > {irrigation_trigger:.0f} cm.")
        if source == "Terrain surface (reference style)":
            if not DEM_FILE.exists():
                st.error("The local DEM file is unavailable.")
                st.stop()
            values, transform, crs = dem_preview(selected)
            title, unit = f"Midsayap surface — {place}", "metres elevation"
            st.caption("Reference-style terrain surface using the supplied Midsayap elevation raster. Blue = lower terrain; cyan/yellow/orange/red = higher terrain.")
        if source == "Distance raster (.tif)":
            raster = upload or (DISTANCE_RASTERS[0] if DISTANCE_RASTERS else None)
            if raster is None:
                st.info("Upload a local GeoTIFF containing water distance values to create the map.")
                st.stop()
            values, transform, crs = raster_distance(raster, selected)
            title = f"Water distance — {place}"
        elif source == "Exported Google Sheet (CSV/XLSX)":
            if upload is None:
                st.info("Export the read-only Google Sheet manually as CSV or XLSX, then upload it here.")
                st.stop()
            table = read_sheet(upload)
            columns = list(table.columns)
            if len(columns) < 3:
                raise ValueError("The spreadsheet needs longitude, latitude, and distance columns.")
            st.sidebar.markdown("**Map the spreadsheet columns**")
            x_col = st.sidebar.selectbox("Longitude (EPSG:4326)", columns, index=0)
            y_col = st.sidebar.selectbox("Latitude (EPSG:4326)", columns, index=min(1, len(columns) - 1))
            value_col = st.sidebar.selectbox("Water distance", columns, index=min(2, len(columns) - 1))
            crs = boundaries.crs
            values, transform = interpolate_sheet(table, x_col, y_col, value_col, selected, crs, grid_size)
            title = f"Interpolated water distance — {place}"

        if not np.isfinite(values).any():
            raise ValueError("No valid values remain inside the chosen place.")
        is_awd_layer = source == "Water distance map (Google Sheet)" and map_layer.startswith("AWD ")
        st.plotly_chart(make_figure(
            values, transform, crs, selected, title, unit,
            points if source == "Water distance map (Google Sheet)" else None,
            RICE_AWD_COLORS if is_awd_layer else None,
            40 if is_awd_layer else None,
            satellite_mode if source == "Water distance map (Google Sheet)" else False,
        ), use_container_width=True)
        st.caption("Coloured surface = model estimate; outlined, labelled station markers = actual sensor observations. Hover for values, pan/zoom to inspect fields, and use the toolbar to export the map.")
    except Exception as error:
        st.error(f"Could not draw the selected data: {error}")


if __name__ == "__main__":
    main()
