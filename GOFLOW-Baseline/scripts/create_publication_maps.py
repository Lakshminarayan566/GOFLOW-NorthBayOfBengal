import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================

HIMAWARI_DIR = "data/himawari/dec2022_himawari"
FRONT_DIR = "outputs/daily_fronts"
OUT_DIR = "outputs/publication_maps_9km"
os.makedirs(OUT_DIR, exist_ok=True)

LAT_MIN = 19.0
LAT_MAX = 23.0
LON_MIN = 86.0
LON_MAX = 95.0

# ==========================================================
# HELPERS
# ==========================================================

def find_nc_file(timestamp: str) -> str:
    hits = glob.glob(os.path.join(HIMAWARI_DIR, f"{timestamp}*.nc"))
    if not hits:
        raise FileNotFoundError(f"No Himawari file found for timestamp {timestamp}")
    return hits[0]

# ==========================================================
# LOOP OVER FRONT FILES
# ==========================================================

front_files = sorted(glob.glob(os.path.join(FRONT_DIR, "front_*.npy")))

print(f"\nFound {len(front_files)} front files")

# Keep only 00 UTC, 12 UTC and 21 UTC
front_files = [
    f for f in front_files
    if os.path.basename(f)[14:16] in ["00", "12", "21"]
]

print(f"Selected {len(front_files)} publication timestamps")


sst_cmap = plt.cm.turbo.copy()
sst_cmap.set_bad("white")

front_cmap = plt.cm.jet.copy()
front_cmap.set_bad("white")

log_cmap = plt.cm.jet.copy()
log_cmap.set_bad("white")

for front_path in front_files:
    timestamp = os.path.basename(front_path).replace("front_", "").replace(".npy", "")
    print(f"\nProcessing {timestamp}")

    nc_file = find_nc_file(timestamp)
    ds = xr.open_dataset(nc_file)

    lat = ds["lat"].values
    lon = ds["lon"].values

    lat_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lon_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]

    lat_sub = lat[lat_idx]
    lon_sub = lon[lon_idx]

    sst = ds["sea_surface_temperature"][0].values.astype(np.float32)
    quality = ds["quality_level"][0].values.astype(np.float32)

    sst = sst[np.ix_(lat_idx, lon_idx)]
    quality = quality[np.ix_(lat_idx, lon_idx)]

    # North-up display
    if lat_sub[0] < lat_sub[-1]:
        sst = sst[::-1, :]
        quality = quality[::-1, :]
        lat_sub = lat_sub[::-1]

    sst = sst - 273.15
    sst[quality != 5] = np.nan

    front = np.load(front_path)
    log_front = np.load(os.path.join(FRONT_DIR, f"log_front_{timestamp}.npy"))

    front[~np.isfinite(front)] = np.nan
    log_front[~np.isfinite(log_front)] = np.nan

    # ---------------- SST ----------------
    plt.figure(figsize=(8, 5))
    plt.imshow(
        sst,
        cmap=sst_cmap,
        origin="upper",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        aspect="auto"
    )
    plt.colorbar(label="SST (°C)")
    plt.title(f"SST\n{timestamp}")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"sst_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -------------- Front ---------------
    plt.figure(figsize=(8, 5))
    plt.imshow(
        front,
        cmap=front_cmap,
        origin="lower",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        aspect="auto",
        vmin=np.nanpercentile(front, 1),
        vmax=np.nanpercentile(front, 99)
    )
    plt.colorbar(label="Gradient Magnitude (°C/pixel)")
    plt.title(f"Thermal Front Magnitude\n{timestamp}")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"front_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ------------ Log Front -------------
    plt.figure(figsize=(8, 5))
    plt.imshow(
        log_front,
        cmap=log_cmap,
        origin="lower",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        aspect="auto",
        vmin=np.nanpercentile(log_front, 1),
        vmax=np.nanpercentile(log_front, 99)
    )
    plt.colorbar(label="log(|∇T|)")
    plt.title(f"log(Thermal Front)\n{timestamp}")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"log_front_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("\n===================================")
print("MONTHLY PUBLICATION MAPS GENERATED")
print("===================================")