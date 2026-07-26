import os
import glob
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_DIR = "data/himawari/dec2022_himawari"
OUTPUT_DIR = "outputs/daily_fronts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAT_MIN = 19.0
LAT_MAX = 23.0
LON_MIN = 86.0
LON_MAX = 95.0
SIGMA = 0.6

# ==========================================================
# HYCOM GRID (common ~9 km grid)
# ==========================================================
hy = xr.open_dataset("data/hycom/dec2022_hycom.nc4")
HY_LAT = hy["lat"].values
HY_LON = hy["lon"].values

HY_LAT = HY_LAT[(HY_LAT >= LAT_MIN) & (HY_LAT <= LAT_MAX)]
HY_LON = HY_LON[(HY_LON >= LON_MIN) & (HY_LON <= LON_MAX)]

HY_LON2D, HY_LAT2D = np.meshgrid(HY_LON, HY_LAT)

# ==========================================================
# HELPERS
# ==========================================================

def save_front_products(file_path: str) -> None:
    timestamp = os.path.basename(file_path)[:14]  # YYYYMMDDHHMMSS
    print(f"\nProcessing {timestamp}")

    ds = xr.open_dataset(file_path)

    lat = ds["lat"].values
    lon = ds["lon"].values

    sst = ds["sea_surface_temperature"][0].values.astype(np.float32)
    quality = ds["quality_level"][0].values.astype(np.float32)

    lat_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lon_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]

    sst = sst[np.ix_(lat_idx, lon_idx)]
    quality = quality[np.ix_(lat_idx, lon_idx)]

    lat_sub = lat[lat_idx]
    lon_sub = lon[lon_idx]

    # Make the grid north-up
    if lat_sub[0] < lat_sub[-1]:
        sst = sst[::-1, :]
        quality = quality[::-1, :]
        lat_sub = lat_sub[::-1]

    # Kelvin to Celsius
    sst = sst - 273.15

    # Quality mask
    sst[quality != 5] = np.nan

    # NaN-aware Gaussian smoothing using weights
    mask = np.isnan(sst)
    filled = np.where(mask, 0, sst)
    weights = np.where(mask, 0, 1)

    smooth_data = gaussian_filter(filled, sigma=SIGMA)
    smooth_weights = gaussian_filter(weights, sigma=SIGMA)

    sst_smooth = np.where(
        smooth_weights > 1e-6,
        smooth_data / smooth_weights,
        np.nan
    )

    # ==========================================================
    # INTERPOLATE SST TO HYCOM GRID (~9 km)
    # ==========================================================

    interp = RegularGridInterpolator(
        (lat_sub, lon_sub),
        sst_smooth,
        bounds_error=False,
        fill_value=np.nan
    )

    sst_hy = interp(
        np.column_stack([
            HY_LAT2D.ravel(),
            HY_LON2D.ravel()
        ])
    ).reshape(HY_LAT2D.shape)

    # ==========================================================
    # COMPUTE FRONTS ON HYCOM GRID
    # ==========================================================

    mask = np.isnan(sst_hy)
    filled = np.where(mask, 0, sst_hy)
    weights = np.where(mask, 0, 1)

    smooth_data = gaussian_filter(filled, sigma=SIGMA)
    smooth_weights = gaussian_filter(weights, sigma=SIGMA)

    sst_hy = np.where(
        smooth_weights > 1e-6,
        smooth_data / smooth_weights,
        np.nan
    )

    gy, gx = np.gradient(sst_hy)
    front = np.sqrt(gx**2 + gy**2)
    front[~np.isfinite(front)] = np.nan
    front[front <= 0] = np.nan

    log_front = np.log(front)
    log_front[~np.isfinite(log_front)] = np.nan

    np.save(os.path.join(OUTPUT_DIR, f"front_{timestamp}.npy"), front)
    np.save(os.path.join(OUTPUT_DIR, f"log_front_{timestamp}.npy"), log_front)

    print(f"Saved front_{timestamp}.npy and log_front_{timestamp}.npy")

# ==========================================================
# MAIN
# ==========================================================

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nc")))
print(f"\nFound {len(files)} Himawari files")

for file_path in files:
    save_front_products(file_path)

print("\n=================================")
print("MONTHLY HIMAWARI FRONT PROCESSING COMPLETE")
print("=================================")