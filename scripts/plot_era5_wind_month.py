import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================================
# SETTINGS
# ==========================================================

DATA_DIR = "training_data"
OUT_DIR = "outputs/figures_9km/winds"
HYCOM_FILE = "data/hycom/dec2022_hycom.nc4"

os.makedirs(OUT_DIR, exist_ok=True)

LAT_MIN = 19.0
LAT_MAX = 23.0
LON_MIN = 86.0
LON_MAX = 95.0

# ==========================================================
# LOAD HYCOM GRID
# ==========================================================

ds = xr.open_dataset(HYCOM_FILE)

hycom_lat = ds["lat"].values
hycom_lon = ds["lon"].values

# ==========================================================
# GET FILES
# ==========================================================

sample_files = sorted(
    [f for f in os.listdir(DATA_DIR)
     if f.endswith(".npz")]
)

print(f"\nFound {len(sample_files)} samples")

# Keep only 00, 12 and 21 UTC
sample_files = [
    f for f in sample_files
    if str(np.load(os.path.join(DATA_DIR, f))["timestamp"])[8:10]
    in ["00", "12", "21"]
]

print(f"Selected {len(sample_files)} publication timestamps")

# ==========================================================
# LOOP
# ==========================================================

for fname in sample_files:

    path = os.path.join(DATA_DIR, fname)

    d = np.load(path)

    timestamp = str(d["timestamp"])

    print(f"\nProcessing {timestamp}")

    X = d["X"]

    u = X[2]
    v = X[3]

    u = np.where(np.abs(u) > 100, np.nan, u)
    v = np.where(np.abs(v) > 100, np.nan, v)

    speed = np.sqrt(u**2 + v**2)

    speed_smooth = gaussian_filter(
        np.where(np.isnan(speed), 0, speed),
        sigma=1.0
    )

    speed_smooth = np.where(
        np.isnan(speed),
        np.nan,
        speed_smooth
    )

    lat_plot = hycom_lat.copy()
    speed_plot = speed_smooth.copy()

    if lat_plot[0] > lat_plot[-1]:

        lat_plot = lat_plot[::-1]

        speed_plot = np.flipud(speed_plot)

        u = np.flipud(u)
        v = np.flipud(v)

    # =====================================================
    # FIGURE
    # =====================================================

    fig = plt.figure(
        figsize=(11, 6),
        dpi=300
    )

    fig.patch.set_facecolor("white")

    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree()
    )

    ax.set_facecolor("white")
    ax.set_aspect("auto")

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    levels = np.linspace(
        0,
        np.nanmax(speed_plot),
        30
    )

    im = ax.contourf(
        hycom_lon,
        lat_plot,
        speed_plot,
        levels=levels,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        antialiased=True,
        zorder=1
    )

    ax.streamplot(
        hycom_lon,
        lat_plot,
        u,
        v,
        color="black",
        density=1.5,
        linewidth=1.0,
        arrowsize=1.2,
        transform=ccrs.PlateCarree(),
        zorder=2
    )

    ax.add_feature(
        cfeature.LAND.with_scale("10m"),
        facecolor="white",
        zorder=3
    )

    ax.set_extent(
        [
            LON_MIN,
            LON_MAX,
            LAT_MIN,
            LAT_MAX
        ],
        crs=ccrs.PlateCarree()
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0,
        zorder=5
    )

    gl.xlines = False
    gl.ylines = False

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}

    gl.xlocator = mticker.FixedLocator(
        range(86, 96)
    )

    gl.ylocator = mticker.FixedLocator(
        [19, 20, 21, 22, 23]
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.03,
        shrink=1.0,
        aspect=25
    )

    cbar.set_label(
        "Wind Speed (m/s)",
        fontsize=12
    )

    cbar.ax.tick_params(
        labelsize=10
    )

    cbar.ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.2f")
    )

    ax.set_title(
        f"Surface Wind Speed\n"
        f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} "
        f"{timestamp[8:10]} UTC",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    outfile = os.path.join(
        OUT_DIR,
        f"wind_{timestamp}.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved {outfile}")

print("\n======================================")
print("WIND MAPS GENERATED")
print("======================================")