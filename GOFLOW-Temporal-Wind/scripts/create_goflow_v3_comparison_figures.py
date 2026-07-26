import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy.ndimage import gaussian_filter

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
PRED_DIR = "outputs/predictions_9km"
OUT_DIR = "outputs/comparison_figures_publication_9km"
os.makedirs(OUT_DIR, exist_ok=True)

# Geographic limits matching the study domain bounds exactly
LAT_MIN, LAT_MAX = 19.0, 23.0
LON_MIN, LON_MAX = 86.0, 95.0

files = sorted(glob.glob(os.path.join(PRED_DIR, "prediction_*.npz")))
print(f"Files found for processing: {len(files)}")

# ==============================================================================
# PROCESSING LOOP
# ==============================================================================
for file in files:
    d = np.load(file)
    timestamp = str(d["timestamp"]).strip()

    # Parse raw timestamp string (e.g., 20221201060000) to standard publication format
    if len(timestamp) >= 12:
        dt_obj = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        formatted_ts = dt_obj.strftime("%B %d, %Y %H:%M UTC")
    else:
        formatted_ts = timestamp

    pred_u, pred_v = d["pred_u"], d["pred_v"]
    true_u, true_v = d["true_u"], d["true_v"]

    # Calculate speeds
    pred_speed = np.sqrt(pred_u**2 + pred_v**2)
    true_speed = np.sqrt(true_u**2 + true_v**2)

    # CRITICAL CLEANUP: Identify land grids from the ground-truth data
    land_mask = np.isnan(true_speed) | (true_speed == 0)
    
    # Apply a subtle spatial smoothing filter to reduce high-frequency ML noise
    pred_speed = gaussian_filter(pred_speed, sigma=0.5)
    
    # Enforce land masks over both matrices
    pred_speed[land_mask] = np.nan
    true_speed[land_mask] = np.nan

    # Calculate absolute spatial error over ocean cells only
    abs_error = np.abs(pred_speed - true_speed)
    abs_error[land_mask] = np.nan

    # Establish uniform colorbar scales to make cross-day visual comparisons valid
    vmax_speed = 0.35  
    vmax_error = 0.10

    # Dynamically generate coordinate grids matching the array dimensions
    ny, nx = true_speed.shape
    lons = np.linspace(LON_MIN, LON_MAX, nx)
    lats = np.linspace(LAT_MIN, LAT_MAX, ny)

    # ==========================================================================
    # FIGURE STRUCTURING (GEOGRAPHIC SUBPLOTS)
    # ==========================================================================
    fig, axes = plt.subplots(
        1, 3, 
        figsize=(18, 7.5), 
        dpi=300, 
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    fig.patch.set_facecolor('white')

    # Standardize map limits, land features, and grid labels across panels
    for i, ax in enumerate(axes):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        
        # Add high-resolution geographical elements over the data layer
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#EAEAEA', edgecolor='#333333', linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, zorder=4)
        ax.set_aspect('auto')  

        # Setup publication gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--', zorder=5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        gl.xlocator = mticker.FixedLocator([87, 89, 91, 93])
        gl.ylocator = mticker.FixedLocator([19.5, 20.5, 21.5, 22.5])
        
        # Hide y-axis labels on inner panels to optimize horizontal white space
        if i > 0:
            gl.left_labels = False

    # Setup color maps and assign explicit white background padding for bad/land cells
    cmap_speed = plt.get_cmap("turbo").copy()
    cmap_speed.set_bad("white")
    
    cmap_error = plt.get_cmap("Reds").copy()
    cmap_error.set_bad("white")

    # Fixed quantization levels for uniform color distribution
    speed_levels = np.linspace(0, vmax_speed, 35)
    error_levels = np.linspace(0, vmax_error, 35)

    # --- PANEL A: Actual Ground Truth ---
    im1 = axes[0].contourf(
        lons, lats, true_speed,
        levels=speed_levels,
        cmap=cmap_speed,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    axes[0].set_title("(A) Actual HYCOM Surface Currents", fontsize=12, fontweight="bold", pad=12)

    # --- PANEL B: GOFLOW Model Prediction ---
    im2 = axes[1].contourf(
        lons, lats, pred_speed,
        levels=speed_levels,
        cmap=cmap_speed,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    axes[1].set_title("(B) GOFLOW V3 Prediction Pattern", fontsize=12, fontweight="bold", pad=12)

    # --- PANEL C: Absolute Spatial Error ---
    im3 = axes[2].contourf(
        lons, lats, abs_error,
        levels=error_levels,
        cmap=cmap_error,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    axes[2].set_title("(C) Absolute Prediction Error", fontsize=12, fontweight="bold", pad=12)

    # ==========================================================================
    # BALANCED COLORBAR PLACEMENT (EQUAL DIMENSIONS FOR ALL 3 PANELS)
    # ==========================================================================
    # Individual horizontal colorbars force equal width modifications 
    # across every single subplot, preventing spatial scale compression.
    
    cbar1 = fig.colorbar(
        im1, ax=axes[0], orientation="horizontal", pad=0.12, shrink=0.85, aspect=20
    )
    cbar1.set_label("Actual Speed (m/s)", fontsize=11, fontweight="bold", labelpad=6)
    cbar1.ax.tick_params(labelsize=9)
    cbar1.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    cbar2 = fig.colorbar(
        im2, ax=axes[1], orientation="horizontal", pad=0.12, shrink=0.85, aspect=20
    )
    cbar2.set_label("Predicted Speed (m/s)", fontsize=11, fontweight="bold", labelpad=6)
    cbar2.ax.tick_params(labelsize=9)
    cbar2.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    cbar3 = fig.colorbar(
        im3, ax=axes[2], orientation="horizontal", pad=0.12, shrink=0.85, aspect=20
    )
    cbar3.set_label("Absolute Error (m/s)", fontsize=11, fontweight="bold", labelpad=6)
    cbar3.ax.tick_params(labelsize=9)
    cbar3.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ==========================================================================
    # EXPORT CONTROL & MARGIN ALLOCATION
    # ==========================================================================
    plt.suptitle(f"Model Validation Analysis — {formatted_ts}", fontsize=15, fontweight="bold", y=0.96)

    # tight_layout consolidates map geometry first
    plt.tight_layout()
    # subplots_adjust overrides the top and horizontal borders safely
    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.18)

    outfile = os.path.join(OUT_DIR, f"comparison_pub_{timestamp}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Saved Balanced Publication Layout Matrix: {outfile}")

print("\n=============================================")
print("SUCCESS: ALL SYSTEM COMPARISON FIGURES ARCHIVED")
print("=============================================")