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

# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = "training_data"
MODEL_FILE = "models/baseline_9km/best_unet_dec2022_9km.pth"
HYCOM_FILE = "data/hycom/dec2022_hycom.nc4"
OUT_DIR = "outputs/unet_predictions_publication_9km"
PRED_ARRAY_DIR = os.path.join(OUT_DIR, "arrays")
PRED_FIG_DIR = os.path.join(OUT_DIR, "figures")

os.makedirs(PRED_ARRAY_DIR, exist_ok=True)
os.makedirs(PRED_FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose timestamps that exist in HYCOM (3-hourly)
EVENTS = [
    "20221219150000",
    "20221231150000",
    "20221231180000",
]

# =====================================================
# MODEL ARCHITECTURE
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
# HELPERS
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
            return None
        path = os.path.join(data_dir, f)
        d = np.load(path)
        if str(d["timestamp"]) == timestamp:
            return path
    return None

def format_ts(ts):
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]} UTC"

# =====================================================
# LOAD DATA AND MODEL
# =====================================================

hycom = xr.open_dataset(HYCOM_FILE)
hy_lat = hycom["lat"].values
hy_lon = hycom["lon"].values

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

print("\nModel loaded successfully.")
print(f"Device: {DEVICE}")
print(f"Output folder: {OUT_DIR}")

# =====================================================
# LOOP THROUGH EVENTS
# =====================================================

for event in EVENTS:
    sample_path = find_sample_by_timestamp(DATA_DIR, event)

    if sample_path is None:
        print(f"Missing sample for {event}")
        continue

    d = np.load(sample_path)

    X = np.nan_to_num(d["X"].astype(np.float32))
    Y = np.nan_to_num(d["Y"].astype(np.float32))

    x_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy()[0]

    true_u = Y[0]
    true_v = Y[1]
    pred_u = pred[0]
    pred_v = pred[1]

    true_speed = np.sqrt(true_u**2 + true_v**2)
    pred_speed = np.sqrt(pred_u**2 + pred_v**2)
    err_speed = np.abs(pred_speed - true_speed)

    # Save compressed arrays
    np.savez_compressed(
        os.path.join(PRED_ARRAY_DIR, f"prediction_{event}.npz"),
        timestamp=event,
        pred_u=pred_u,
        pred_v=pred_v,
        true_u=true_u,
        true_v=true_v,
        pred_speed=pred_speed,
        true_speed=true_speed,
        error_speed=err_speed
    )

    # Publication smoothing
    true_plot = gaussian_nan_smooth(true_speed, sigma=1.0)
    pred_plot = gaussian_nan_smooth(pred_speed, sigma=1.0)
    err_plot = gaussian_nan_smooth(err_speed, sigma=1.0)

    # If lat is descending, flip for display consistency
    hy_lat_plot = hy_lat.copy()
    if hy_lat_plot[0] > hy_lat_plot[-1]:
        hy_lat_plot = hy_lat_plot[::-1]
        true_plot = np.flipud(true_plot)
        pred_plot = np.flipud(pred_plot)
        err_plot = np.flipud(err_plot)

    # =====================================================
    # OPTIMIZED PUBLICATION-QUALITY VISUALIZATION
    # =====================================================

    # Perfect aspect canvas ratio for side-by-side wide map extents
    fig = plt.figure(figsize=(22, 5.5), dpi=300) 

    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())

    # Build clean, standardized tick markers
    main_ticks = np.arange(0.0, 0.8, 0.1)
    err_max = np.nanmax(err_plot) if np.isfinite(err_plot).any() else 0.25
    err_ticks = np.linspace(0.0, np.round(err_max, 2), 6)

    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_facecolor("#f4f4f4")  # Soft baseline for areas with missing data
        ax.set_extent([86.0, 95.0, 19.0, 23.0], crs=ccrs.PlateCarree())
        
        # Distinct light-gray masking for beautiful, distinct landmass profiles
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#dcdcdc", zorder=5)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, edgecolor="#555555", zorder=6)
        
        # High-precision gridlining setup
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gainsboro", linestyle="--", zorder=7)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([86, 88, 90, 92, 94])
        gl.ylocator = mticker.FixedLocator([19, 20, 21, 22, 23])
        gl.xlabel_style = {"size": 11}
        gl.ylabel_style = {"size": 11}
        
        # Turn off redundant intermediate labels to maximize figure whitespace
        if i > 0:
            gl.left_labels = False

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#dcdcdc")  # Ensure data-less boundaries adapt to land palette seamlessly

    levels_main = np.linspace(0.0, 0.75, 100)
    levels_err = np.linspace(0.0, err_max, 100)

    # Subplot 1: True Velocities
    im1 = ax1.contourf(
        hy_lon, hy_lat_plot, true_plot,
        levels=levels_main, cmap=cmap, transform=ccrs.PlateCarree(),
        antialiased=True, extend="max", zorder=1
    )

    # Subplot 2: Model Prediction Model output
    im2 = ax2.contourf(
        hy_lon, hy_lat_plot, pred_plot,
        levels=levels_main, cmap=cmap, transform=ccrs.PlateCarree(),
        antialiased=True, extend="max", zorder=1
    )

    # Subplot 3: Structural Deviation
    im3 = ax3.contourf(
        hy_lon, hy_lat_plot, err_plot,
        levels=levels_err, cmap="magma", transform=ccrs.PlateCarree(),
        antialiased=True, extend="max", zorder=1
    )

    # Section Headers
    ax1.set_title("True HYCOM Speed", fontsize=15, fontweight="bold", pad=12)
    ax2.set_title("Predicted Speed", fontsize=15, fontweight="bold", pad=12)
    ax3.set_title("Absolute Error", fontsize=15, fontweight="bold", pad=12)

    # Colorbar Generation with Clean Decimal Resolution Formatting
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.04, ticks=main_ticks)
    cbar1.set_label("Current Speed (m/s)", fontsize=11, weight="bold")
    cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04, ticks=main_ticks)
    cbar2.set_label("Current Speed (m/s)", fontsize=11, weight="bold")
    cbar2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.04, ticks=err_ticks)
    cbar3.set_label("Absolute Error (m/s)", fontsize=11, weight="bold")
    cbar3.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Main Figure Title
    plt.suptitle(
        f"HYCOM Current Prediction Comparison\n{format_ts(event)}",
        fontsize=18, fontweight="bold", y=0.98
    )

    # Clean adjustments and save handling
    outfile = os.path.join(PRED_FIG_DIR, f"prediction_compare_{event}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved {outfile}")

print("\n===================================")
print("PUBLICATION MAPS SUCCESSFULLY GENERATED")
print("===================================")

