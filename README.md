# Midsayap Water-Distance Explorer

A local interactive Python map for water-distance data in Midsayap. The map uses the supplied Midsayap boundaries, has a deep-blue → cyan → yellow → orange/red colour ramp matching the reference, and can zoom to individual irrigation/place areas.

## Privacy and data source

This application **does not use a Google Sheets API**. It can read the linked public sheet as a CSV snapshot, or you can download a read-only Google Sheet as **CSV** or **XLSX** and upload that local file.

## Run

Run the app from this folder:

    .\.venv\Scripts\python.exe -m streamlit run app.py

## Water-distance source options

- **Forecast-aware irrigation plan:** The default operational workspace. It joins the newest valid IoT water-distance reading for each located device with a 3–14 day Open-Meteo daily forecast. It adapts to **Wet Season** (May – Oct) and **Dry Season** (Nov – Apr) cropping cycles by adjusting default irrigation triggers (35 cm vs 30 cm raw sensor distance), ET0 evapotranspiration weights (0.15 vs 0.20), and rain-deferral thresholds (≥8 mm vs ≥15 mm). It ranks stations as irrigate today, schedule within 48 hours, hold for forecast rain, or monitor. The projected distance is a transparent decision-support estimate based on recent station drawdown, forecast reference ET, and rainfall; confirm it against field and canal conditions. The delivery-volume figure is an adjustable planning scenario using the representative area and planned application depth in the sidebar.

- **Terrain surface (reference style):** Renders the bundled Midsayap elevation raster in the supplied deep-blue → cyan → yellow → orange/red visual style and supports place-based zooming. This layer represents elevation, not water distance.
- **Water distance map (Google Sheet):** Uses the actual `Water Distance (cm)` values from the public sheet. Upload `device_locations.csv`, containing the `Device`, `longitude`, and `latitude` for every sensor. The app then interpolates each device's latest or median real measurement across the selected Midsayap place with the reference colour gradient. Download the included [device_locations_template.csv](device_locations_template.csv) from the app or use it as the starting file.
- **Public Google Sheet (read-only):** Reads the supplied public CSV export without an API and displays its water-distance readings by time and device. The current sheet has no coordinates or place column, so it cannot be mapped geographically.
- **Distance raster (.tif):** Upload a GeoTIFF where pixel values are distance to water. The app crops it to the selected place.
- **Exported Google Sheet:** Upload a CSV/XLSX with at least three columns: longitude, latitude, and water distance. Select those fields in the sidebar. Coordinates must be WGS 84 longitude/latitude (EPSG:4326). The map interpolates the measurements only inside the selected Midsayap boundary.

### Example spreadsheet

| longitude | latitude | water_distance_m |
| ---: | ---: | ---: |
| 124.5231 | 7.1902 | 86.5 |
| 124.5487 | 7.1753 | 241.0 |
| 124.5573 | 7.2048 | 502.7 |

## Interaction

Choose **Adjust view by place** in the sidebar, then use the map toolbar to pan, zoom, reset, or download the current visualization. Hover over cells to inspect their distance value.
