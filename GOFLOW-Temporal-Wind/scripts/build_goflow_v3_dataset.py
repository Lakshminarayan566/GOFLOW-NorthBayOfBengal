import os
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import RegularGridInterpolator

# =====================================================
# PATHS
# =====================================================

LOG_FRONT_DIR = "data/log_fronts_9km"
ERA5_FILE = "data/era5/dec2022_era5.grib"
HYCOM_FILE = "data/hycom/dec2022_hycom.nc4"

OUT_DIR = "training_data_9km"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# LOAD HYCOM
# =====================================================

hy = xr.open_dataset(HYCOM_FILE)

hy_time = pd.to_datetime(hy.time.values)

hy_lat = hy.lat.values
hy_lon = hy.lon.values

# =====================================================
# LOAD ERA5
# =====================================================

era = xr.open_dataset(
    ERA5_FILE,
    engine="cfgrib"
)

era_time = pd.to_datetime(
    era.time.values
)

era_lat = era.latitude.values
era_lon = era.longitude.values

# ERA5 latitude descending
if era_lat[0] > era_lat[-1]:
    era_lat = era_lat[::-1]

# =====================================================
# TARGET GRID
# =====================================================

hy_lon_grid, hy_lat_grid = np.meshgrid(
    hy_lon,
    hy_lat
)

target_points = np.column_stack([
    hy_lat_grid.ravel(),
    hy_lon_grid.ravel()
])

# =====================================================
# HELPERS
# =====================================================

def ts_to_str(ts):

    return pd.Timestamp(ts).strftime(
        "%Y%m%d%H%M%S"
    )

def load_logfront(timestamp):

    file = os.path.join(
        LOG_FRONT_DIR,
        f"log_front_{timestamp}.npy"
    )

    arr = np.load(file)

    arr = np.nan_to_num(arr)

    return arr.astype(np.float32)

def interp_front(front):

    lat = np.linspace(
        19,
        23,
        front.shape[0]
    )

    lon = np.linspace(
        86,
        95,
        front.shape[1]
    )

    f = RegularGridInterpolator(
        (lat, lon),
        front,
        bounds_error=False,
        fill_value=0
    )

    out = f(
        target_points
    ).reshape(
        len(hy_lat),
        len(hy_lon)
    )

    return out

def interp_era(field):

    if field.shape[0] != len(era_lat):
        field = np.flipud(field)

    f = RegularGridInterpolator(
        (era_lat, era_lon),
        field,
        bounds_error=False,
        fill_value=0
    )

    out = f(
        target_points
    ).reshape(
        len(hy_lat),
        len(hy_lon)
    )

    return out

# =====================================================
# BUILD DATASET
# =====================================================

count = 0
skipped = 0

for i, ts in enumerate(hy_time):

    t = pd.Timestamp(ts)

    t_minus = t - pd.Timedelta(hours=1)
    t_plus = t + pd.Timedelta(hours=1)

    s0 = ts_to_str(t_minus)
    s1 = ts_to_str(t)
    s2 = ts_to_str(t_plus)

    f0 = os.path.join(
        LOG_FRONT_DIR,
        f"log_front_{s0}.npy"
    )

    f1 = os.path.join(
        LOG_FRONT_DIR,
        f"log_front_{s1}.npy"
    )

    f2 = os.path.join(
        LOG_FRONT_DIR,
        f"log_front_{s2}.npy"
    )

    if not (
        os.path.exists(f0)
        and os.path.exists(f1)
        and os.path.exists(f2)
    ):
        skipped += 1
        continue

    # -----------------------------------------
    # LOG FRONTS
    # -----------------------------------------

    lf0 = load_logfront(s0)
    lf1 = load_logfront(s1)
    lf2 = load_logfront(s2)
    
    if (
    lf0.shape != (len(hy_lat), len(hy_lon))
    or lf1.shape != (len(hy_lat), len(hy_lon))
    or lf2.shape != (len(hy_lat), len(hy_lon))
    ):
      print(f"Shape mismatch: {s1}")
      skipped += 1
      continue

    # -----------------------------------------
    # ERA5
    # -----------------------------------------

    era_idx = np.argmin(
        np.abs(
            era_time - t
        )
    )

    u10 = era["u10"].isel(
        time=era_idx
    ).values

    v10 = era["v10"].isel(
        time=era_idx
    ).values

    u10 = interp_era(u10)
    v10 = interp_era(v10)

    # -----------------------------------------
    # HYCOM TARGET
    # -----------------------------------------

    hy_u = hy["water_u"].isel(
        time=i,
        depth=0
    ).values

    hy_v = hy["water_v"].isel(
        time=i,
        depth=0
    ).values

    hy_u = np.nan_to_num(
        hy_u
    )

    hy_v = np.nan_to_num(
        hy_v
    )

    # -----------------------------------------
    # STACK
    # -----------------------------------------

    X = np.stack([

        lf0,
        lf1,
        lf2,

        u10,
        v10

    ]).astype(np.float32)

    Y = np.stack([

        hy_u,
        hy_v

    ]).astype(np.float32)

    np.savez_compressed(

        os.path.join(
            OUT_DIR,
            f"sample_{count:04d}.npz"
        ),

        X=X,
        Y=Y,
        timestamp=s1
    )

    count += 1

    if count % 20 == 0:

        print(
            f"Created {count}"
        )

print("\n=======================")
print("GOFLOW V3 COMPLETE")
print("=======================")

print(
    f"Samples: {count}"
)

print(
    f"Skipped: {skipped}"
)