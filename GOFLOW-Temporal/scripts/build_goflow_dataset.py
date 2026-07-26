import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# =====================================================
# CONFIGURATION
# =====================================================

HYCOM_FILE = "data/hycom/dec2022_hycom.nc4"
LOG_FRONT_DIR = "data/himawari_9km"

OUT_DIR = "training_data"
os.makedirs(OUT_DIR, exist_ok=True)

LAT_MIN, LAT_MAX = 19.0, 23.0
LON_MIN, LON_MAX = 86.0, 95.0

# =====================================================
# HELPERS
# =====================================================

def ts_to_str(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M%S")

def load_log_front(timestamp: str) -> np.ndarray:
    path = os.path.join(LOG_FRONT_DIR, f"log_front_{timestamp}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    arr = np.load(path).astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr

def interpolate_to_hycom(field: np.ndarray, target_points: np.ndarray,
                         hy_lat_len: int, hy_lon_len: int) -> np.ndarray:
    """
    Interpolate a 2D field from the front grid to the HYCOM grid.
    The front grid is assumed to span the same NBOB box:
    lat 19–23, lon 86–95.
    """
    front_lat = np.linspace(LAT_MIN, LAT_MAX, field.shape[0])
    front_lon = np.linspace(LON_MIN, LON_MAX, field.shape[1])

    field_fill = np.nan_to_num(field, nan=0.0)

    interp = RegularGridInterpolator(
        (front_lat, front_lon),
        field_fill,
        bounds_error=False,
        fill_value=np.nan
    )

    out = interp(target_points).reshape(hy_lat_len, hy_lon_len)
    return out

# =====================================================
# LOAD HYCOM
# =====================================================

hycom = xr.open_dataset(HYCOM_FILE)

hy_time = pd.to_datetime(hycom.time.values)
hy_lat = hycom.lat.values
hy_lon = hycom.lon.values

hy_lon_grid, hy_lat_grid = np.meshgrid(hy_lon, hy_lat)
target_points = np.column_stack([hy_lat_grid.ravel(), hy_lon_grid.ravel()])

print(f"HYCOM timesteps: {len(hy_time)}")
print(f"HYCOM grid: {len(hy_lat)} x {len(hy_lon)}")

# =====================================================
# LOOP OVER HYCOM TIMES
# =====================================================

sample_count = 0
skipped_count = 0

for idx, timestamp in enumerate(hy_time):
    ts = pd.Timestamp(timestamp)
    ts_str = ts_to_str(ts)

    ts_minus = ts - pd.Timedelta(hours=1)
    ts_plus = ts + pd.Timedelta(hours=1)

    ts_minus_str = ts_to_str(ts_minus)
    ts_plus_str = ts_to_str(ts_plus)

    path_minus = os.path.join(LOG_FRONT_DIR, f"log_front_{ts_minus_str}.npy")
    path_t = os.path.join(LOG_FRONT_DIR, f"log_front_{ts_str}.npy")
    path_plus = os.path.join(LOG_FRONT_DIR, f"log_front_{ts_plus_str}.npy")

    # Need all 3 consecutive hourly inputs
    if not (os.path.exists(path_minus) and os.path.exists(path_t) and os.path.exists(path_plus)):
        skipped_count += 1
        continue

    # Load input log-front fields
    log_minus = load_log_front(ts_minus_str)
    log_t = load_log_front(ts_str)
    log_plus = load_log_front(ts_plus_str)

    # Interpolate each input to HYCOM grid
    
    log_minus_hy = log_minus
    log_t_hy = log_t
    log_plus_hy = log_plus

    # HYCOM target currents
    water_u = hycom["water_u"].isel(time=idx, depth=0).values.astype(np.float32)
    water_v = hycom["water_v"].isel(time=idx, depth=0).values.astype(np.float32)

    water_u = np.where(np.isfinite(water_u), water_u, np.nan)
    water_v = np.where(np.isfinite(water_v), water_v, np.nan)

    # Final sanity check
    if (
        log_minus_hy.shape != (len(hy_lat), len(hy_lon)) or
        log_t_hy.shape != (len(hy_lat), len(hy_lon)) or
        log_plus_hy.shape != (len(hy_lat), len(hy_lon)) or
        water_u.shape != (len(hy_lat), len(hy_lon)) or
        water_v.shape != (len(hy_lat), len(hy_lon))
    ):
        print(f"Shape mismatch at {ts_str}, skipping.")
        skipped_count += 1
        continue

    # Stack GOFLOW inputs and targets
    X = np.stack([
        log_minus_hy,
        log_t_hy,
        log_plus_hy
    ]).astype(np.float32)

    Y = np.stack([
        water_u,
        water_v
    ]).astype(np.float32)

    out_file = os.path.join(OUT_DIR, f"sample_{sample_count:04d}.npz")

    np.savez_compressed(
        out_file,
        X=X,
        Y=Y,
        timestamp=ts_str,
        source_times=np.array([ts_minus_str, ts_str, ts_plus_str])
    )

    sample_count += 1

    if sample_count % 20 == 0:
        print(f"Created {sample_count} samples")

print("\n================================")
print("GOFLOW DATASET BUILD COMPLETE")
print("================================")
print(f"Samples created: {sample_count}")
print(f"Samples skipped: {skipped_count}")