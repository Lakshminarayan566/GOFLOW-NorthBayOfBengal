import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import xarray as xr
from datetime import datetime

# =====================================================
# CONFIGURATION & PATHS
# =====================================================
DATA_DIR = "training_data"
HYCOM_FILE = "data/hycom/dec2022_hycom.nc4"

MODEL_FILE = "models/best_unet_dec2022.pth"
OUT_DIR = "outputs/unet_predictions_publication"
PRED_ARRAY_DIR = os.path.join(OUT_DIR, "arrays")
PRED_FIG_DIR = os.path.join(OUT_DIR, "figures")

os.makedirs(PRED_ARRAY_DIR, exist_ok=True)
os.makedirs(PRED_FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVENTS = [
    "20221219150000",
    "20221231150000",
    "20221231180000",
]

# =====================================================
# MODEL ARCHITECTURE (U-Net)
# =====================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(4, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.final = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def gaussian_nan_smooth(arr, sigma=1.0):
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    weights = valid.astype(np.float32)

    smooth_data = gaussian_filter(filled, sigma=sigma)
    smooth_weights = gaussian_filter(weights, sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = smooth_data / smooth_weights

    out[smooth_weights == 0] = np.nan
    return out

def find_sample_by_timestamp(data_dir, timestamp):
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".npz"):
            continue  
        path = os.path.join(data_dir, f)
        d = np.load(path)
        if str(d["timestamp"]) == timestamp:
            return path
    return None

def format_ts(ts):
    dt_obj = datetime.strptime(ts, "%Y%m%d%H%M%S")
    return dt_obj.strftime("%B %d, %Y %H:%M UTC")

# =====================================================
# LOAD GEOGRAPHIC REF AND WEIGHTS
# =====================================================
hycom = xr.open_dataset(HYCOM_FILE)
hy_lat = hycom["lat"].values
hy_lon = hycom["lon"].values

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

print("\nModel initialization successful.")
print(f"Execution Device: {DEVICE}")

# =====================================================
# MAIN INFERENCE AND GENERATION LOOP
# =====================================================
for event in EVENTS:
    sample_path = find_sample_by_timestamp(DATA_DIR, event)

    if sample_path is None:
        print(f"Missing configuration sample for target event: {event}")
        continue

    d = np.load(sample_path)

    # Clean array indexing for tracking ocean/land configurations
    raw_y = d["Y"].astype(np.float32)
    land_mask = np.isnan(raw_y[0]) | (raw_y[0] == 0)

    X = np.nan_to_num(d["X"].astype(np.float32))
    x_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy()[0]

    true_u, true_v = raw_y[0], raw_y[1]
    pred_u, pred_v = pred[0], pred[1]

    true_speed = np.sqrt(true_u**2 + true_v**2)
    pred_speed = np.sqrt(pred_u**2 + pred_v**2)
    
    # Enforce static land bounds 
    true_speed[land_mask] = np.nan
    pred_speed[land_mask] = np.nan
    err_speed = np.abs(pred_speed - true_speed)
    err_speed[land_mask] = np.nan

    np.savez_compressed(
        os.path.join(PRED_ARRAY_DIR, f"prediction_{event}.npz"),
        timestamp=event, pred_u=pred_u, pred_v=pred_v,
        true_u=true_u, true_v=true_v,
        pred_speed=pred_speed, true_speed=true_speed, error_speed=err_speed
    )

    true_plot = gaussian_nan_smooth(true_speed, sigma=0.5)
    pred_plot = gaussian_nan_smooth(pred_speed, sigma=0.5)
    err_plot = gaussian_nan_smooth(err_speed, sigma=0.5)

    hy_lat_plot = hy_lat.copy()
    if hy_lat_plot[0] > hy_lat_plot[-1]:
        hy_lat_plot = hy_lat_plot[::-1]
        true_plot = np.flipud(true_plot)
        pred_plot = np.flipud(pred_plot)
        err_plot = np.flipud(err_plot)

    # =====================================================
    # VISUALIZATION GRAPHICS CANVAS SETUP
    # =====================================================
    fig, axes = plt.subplots(
        1, 3, 
        figsize=(18, 7.5), 
        dpi=300, 
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    fig.patch.set_facecolor('white')

    vmax_speed = 0.35  
    vmax_error = 0.10
    
    speed_levels = np.linspace(0.0, vmax_speed, 35)
    error_levels = np.linspace(0.0, vmax_error, 35)

    for i, ax in enumerate(axes):
        ax.set_extent([86.0, 95.0, 19.0, 23.0], crs=ccrs.PlateCarree())
        ax.set_aspect('auto')  # Eliminates physical matrix distortion across map frames
        
        # Add publication background assets
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#EAEAEA", edgecolor="#333333", linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, zorder=4)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--', zorder=5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([87, 89, 91, 93])
        gl.ylocator = mticker.FixedLocator([19.5, 20.5, 21.5, 22.5])
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}
        
        if i > 0:
            gl.left_labels = False

    cmap_speed = plt.get_cmap("turbo").copy()
    cmap_speed.set_bad("white")  
    
    cmap_error = plt.get_cmap("Reds").copy()
    cmap_error.set_bad("white")

    # --- Panel A: Actual Ground Truth ---
    im1 = axes[0].contourf(
        hy_lon, hy_lat_plot, true_plot,
        levels=speed_levels, cmap=cmap_speed, extend="max",
        transform=ccrs.PlateCarree(), zorder=1
    )
    axes[0].set_title("(A) Actual HYCOM Surface Currents", fontsize=12, fontweight="bold", pad=12)

    # --- Panel B: Model Prediction Output ---
    im2 = axes[1].contourf(
        hy_lon, hy_lat_plot, pred_plot,
        levels=speed_levels, cmap=cmap_speed, extend="max",
        transform=ccrs.PlateCarree(), zorder=1
    )
    axes[1].set_title("(B) Baseline Prediction Pattern", fontsize=12, fontweight="bold", pad=12)

    # --- Panel C: Calculated Absolute Spatial Error ---
    im3 = axes[2].contourf(
        hy_lon, hy_lat_plot, err_plot,
        levels=error_levels, cmap=cmap_error, extend="max",
        transform=ccrs.PlateCarree(), zorder=1
    )
    axes[2].set_title("(C) Absolute Prediction Error", fontsize=12, fontweight="bold", pad=12)

    # =====================================================
    # FIXED BALANCED COLORBAR GRID PLACEMENT 
    # =====================================================
    # Attaching dedicated colorbars directly to individual axes forces 
    # matplotlib to apply an identical bounding box reduction to all 3 plots.
    
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

    # =====================================================
    # LAYOUT ADJUSTMENT (PREVENTS HEADINGS FROM MIXING)
    # =====================================================
    plt.suptitle(
        f"UNet Validation Analysis Model Output\n{format_ts(event)}",
        fontsize=15, fontweight="bold", y=0.96
    )

    # Pack elements cleanly, then force clear margins at the top and bottom bounds
    plt.tight_layout()
    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.18)

    # Save out configuration output vectors
    outfile = os.path.join(PRED_FIG_DIR, f"prediction_compare_pub_{event}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Generated Balanced Layout Analysis Plate: {outfile}")

print("\n================================================")
print("SUCCESS: ALL GEOGRAPHIC IMAGES PROCESSED CLEANLY")
print("================================================")