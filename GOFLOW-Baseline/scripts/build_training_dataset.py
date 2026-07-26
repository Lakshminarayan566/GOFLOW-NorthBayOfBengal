import os
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import RegularGridInterpolator

# =====================================================
# FILES
# =====================================================

HYCOM_FILE = "data/hycom/dec2022_hycom.nc4"
ERA5_FILE = "data/era5/dec2022_era5.nc"

FRONT_DIR = "outputs/daily_fronts"
OUT_DIR = "training_data"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# LOAD HYCOM
# =====================================================

hycom = xr.open_dataset(HYCOM_FILE)

hy_time = pd.to_datetime(hycom.time.values)

hy_lat = hycom.lat.values
hy_lon = hycom.lon.values

hy_lon_grid, hy_lat_grid = np.meshgrid(hy_lon, hy_lat)

target_points = np.column_stack(
    [hy_lat_grid.ravel(), hy_lon_grid.ravel()]
)

# =====================================================
# LOAD ERA5
# =====================================================

era = xr.open_dataset(ERA5_FILE)

era_time = pd.to_datetime(era.time.values)

era_lat = era.lat.values
era_lon = era.lon.values

# =====================================================
# LOOP THROUGH HYCOM TIMES
# =====================================================

sample_count = 0

for idx, timestamp in enumerate(hy_time):

    ts_string = pd.Timestamp(timestamp).strftime("%Y%m%d%H%M%S")

    front_file = os.path.join(
        FRONT_DIR,
        f"front_{ts_string}.npy"
    )

    log_file = os.path.join(
        FRONT_DIR,
        f"log_front_{ts_string}.npy"
    )

    if not os.path.exists(front_file):
        continue

    if not os.path.exists(log_file):
        continue

    # =================================================
    # LOAD FRONTS
    # =================================================

    front = np.load(front_file)
    log_front = np.load(log_file)
    
    print("Original front :", front.shape)
    print("Original log   :", log_front.shape)
    print("ERA lat/lon    :", era_lat.shape, era_lon.shape)
    print("HYCOM lat/lon  :", hy_lat.shape, hy_lon.shape)

    # Front grid coordinates
    front_lat = np.linspace(19.0, 23.0, front.shape[0])
    front_lon = np.linspace(86.0, 95.0, front.shape[1])

    front_fill = np.nan_to_num(front, nan=0.0)
    log_fill = np.nan_to_num(log_front, nan=0.0)

    front_interp = RegularGridInterpolator(
        (front_lat, front_lon),
        front_fill,
        bounds_error=False,
        fill_value=np.nan
    )

    log_interp = RegularGridInterpolator(
        (front_lat, front_lon),
        log_fill,
        bounds_error=False,
        fill_value=np.nan
    )

    front_hy = front_interp(target_points).reshape(
        len(hy_lat),
        len(hy_lon)
    )

    log_hy = log_interp(target_points).reshape(
        len(hy_lat),
        len(hy_lon)
    )

    # =================================================
    # ERA5 MATCH
    # =================================================

    era_idx = np.argmin(
        np.abs(era_time - timestamp)
    )

    u10 = era["var165"][era_idx].values
    v10 = era["var166"][era_idx].values

    # ERA5 latitude is descending
    interp_u = RegularGridInterpolator(
        (era_lat[::-1], era_lon),
        u10[::-1, :],
        bounds_error=False,
        fill_value=np.nan
    )

    interp_v = RegularGridInterpolator(
        (era_lat[::-1], era_lon),
        v10[::-1, :],
        bounds_error=False,
        fill_value=np.nan
    )

    u10_hy = interp_u(target_points).reshape(
        len(hy_lat),
        len(hy_lon)
    )

    v10_hy = interp_v(target_points).reshape(
        len(hy_lat),
        len(hy_lon)
    )

    # =================================================
    # HYCOM TARGETS
    # =================================================

    water_u = hycom["water_u"][idx, 0].values
    water_v = hycom["water_v"][idx, 0].values

    # =================================================
    # SHAPE CHECK
    # =================================================

    print(
        front_hy.shape,
        log_hy.shape,
        u10_hy.shape,
        v10_hy.shape,
        water_u.shape,
        water_v.shape
    )

    # =================================================
    # STACK INPUTS
    # =================================================

    X = np.stack([
        front_hy,
        log_hy,
        u10_hy,
        v10_hy
    ])

    Y = np.stack([
        water_u,
        water_v
    ])

    np.savez_compressed(
        os.path.join(
            OUT_DIR,
            f"sample_{idx:04d}.npz"
        ),
        X=X,
        Y=Y,
        timestamp=ts_string
    )

    sample_count += 1

    if sample_count % 20 == 0:
        print(f"Created {sample_count} samples")

print("\n================================")
print("DATASET BUILD COMPLETE")
print("================================")
print(f"Samples: {sample_count}")