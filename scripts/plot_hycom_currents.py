import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =====================================================
# SETTINGS
# =====================================================

EVENTS = [
    "20221219150000",
    "20221231180000"
]

DATA_DIR = "training_data"
OUT_DIR = "outputs/figures_9km/surface_currents"
HYCOM_FILE = "data/hycom/dec2022_hycom.nc4"

os.makedirs(OUT_DIR, exist_ok=True)

LAT_MIN, LAT_MAX = 19.0, 23.0
LON_MIN, LON_MAX = 86.0, 95.0

# =====================================================
# LOAD HYCOM GRID
# =====================================================

ds = xr.open_dataset(HYCOM_FILE)

hycom_lat = ds["lat"].values
hycom_lon = ds["lon"].values

# =====================================================
# LOOP
# =====================================================

for event in EVENTS:

    print(f"Processing {event}")

    sample_file = None

    for f in sorted(os.listdir(DATA_DIR)):

        if not f.endswith(".npz"):
            continue

        path = os.path.join(DATA_DIR, f)

        d = np.load(path)

        if str(d["timestamp"]) == event:
            sample_file = path
            break

    if sample_file is None:
        print(f"Missing {event}")
        continue

    d = np.load(sample_file)

    Y = d["Y"]

    u = Y[0]
    v = Y[1]

    u = np.where(np.abs(u) > 10, np.nan, u)
    v = np.where(np.abs(v) > 10, np.nan, v)

    speed = np.sqrt(u**2 + v**2)

    speed_smooth = gaussian_filter(
        np.where(np.isnan(speed), 0, speed),
        sigma=1.5
    )

    speed_smooth = np.where(
        np.isnan(speed),
        np.nan,
        speed_smooth
    )

    speed_smooth = np.clip(
        speed_smooth,
        0,
        0.75
    )

    hycom_lat_plot = hycom_lat.copy()
    speed_plot = speed_smooth.copy()

    if hycom_lat_plot[0] > hycom_lat_plot[-1]:

        hycom_lat_plot = hycom_lat_plot[::-1]

        speed_plot = np.flipud(speed_plot)

        u = np.flipud(u)
        v = np.flipud(v)

    # =====================================================
    # PUBLICATION FIGURE
    # =====================================================

    fig = plt.figure(
        figsize=(11, 6),
        dpi=600
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
        0.75,
        40
    )

    im = ax.contourf(
        hycom_lon,
        hycom_lat_plot,
        speed_plot,
        levels=levels,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        antialiased=True,
        zorder=1
    )

    # -------------------------------------------------
    # VECTORS
    # -------------------------------------------------

    skip = 10

    lon_q = hycom_lon[::skip]
    lat_q = hycom_lat_plot[::skip]

    u_q = u[::skip, ::skip]
    v_q = v[::skip, ::skip]

    lon_grid, lat_grid = np.meshgrid(
        lon_q,
        lat_q
    )

    valid = np.isfinite(u_q) & np.isfinite(v_q)

    ax.quiver(
        lon_grid[valid],
        lat_grid[valid],
        u_q[valid],
        v_q[valid],
        color="black",
        scale=9,
        width=0.002,
        headwidth=4,
        headlength=5,
        headaxislength=4,
        transform=ccrs.PlateCarree(),
        zorder=2
    )

    # -------------------------------------------------
    # LAND
    # -------------------------------------------------

    ax.add_feature(
        cfeature.LAND.with_scale("10m"),
        facecolor="white",
        zorder=4
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
        zorder=6
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

    # -------------------------------------------------
    # COLORBAR
    # -------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.03,
        shrink=1.0,
        aspect=25,
        ticks=np.arange(
            0,
            0.76,
            0.1
        )
    )

    cbar.set_label(
        "Current Speed (m/s)",
        fontsize=12
    )

    cbar.ax.tick_params(
        labelsize=10
    )

    cbar.ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.2f")
    )

    # -------------------------------------------------
    # TITLE
    # -------------------------------------------------

    ax.set_title(
        f"HYCOM Surface Current Speed\n"
        f"{event[:4]}-{event[4:6]}-{event[6:8]} {event[8:10]} UTC",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    outfile = os.path.join(
        OUT_DIR,
        f"hycom_speed_{event}.png"
    )

    plt.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved {outfile}")

print("\n======================================")
print("PUBLICATION MAPS GENERATED")
print("======================================")
