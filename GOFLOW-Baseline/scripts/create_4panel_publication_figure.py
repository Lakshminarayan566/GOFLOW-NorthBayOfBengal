import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =====================================================
# CONFIGURATION
# =====================================================

OUT_DIR = "outputs/publication_figures"
os.makedirs(OUT_DIR, exist_ok=True)

FRONT_DIR = "outputs/publication_maps"
CURRENT_DIR = "outputs/hycom_currents_publication"
WIND_DIR = "outputs/wind_maps_publication"

# Take every timestamp for which all 4 images exist
front_files = sorted(glob.glob(os.path.join(FRONT_DIR, "front_*.png")))

front_files = [
    f for f in front_files
    if os.path.basename(f)[14:16] in ["00", "12", "21"]
]

print(f"Selected {len(front_files)} timestamps")

# =====================================================
# LOOP
# =====================================================

for front_path in front_files:
    timestamp = os.path.basename(front_path).replace("front_", "").replace(".png", "")

    log_path = os.path.join(FRONT_DIR, f"log_front_{timestamp}.png")
    current_path = os.path.join(CURRENT_DIR, f"current_{timestamp}.png")
    wind_path = os.path.join(WIND_DIR, f"wind_{timestamp}.png")

    if not (os.path.exists(log_path) and os.path.exists(current_path) and os.path.exists(wind_path)):
        print(f"Skipping {timestamp} (missing one or more panels)")
        continue

    print(f"Generating 4-panel figure for {timestamp}")

    front_img = mpimg.imread(front_path)
    log_img = mpimg.imread(log_path)
    current_img = mpimg.imread(current_path)
    wind_img = mpimg.imread(wind_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    axes[0, 0].imshow(front_img)
    axes[0, 0].set_title("(A) Thermal Front Magnitude", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(log_img)
    axes[0, 1].set_title("(B) Log(dT)", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(current_img)
    axes[1, 0].set_title("(C) HYCOM Surface Currents", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(wind_img)
    axes[1, 1].set_title("(D) ERA5 Surface Wind Field", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle(
        f"North Bay of Bengal Thermal Front Analysis ({timestamp})\n"
        "Himawari SST, HYCOM Currents and ERA5 Winds",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    outfile = os.path.join(OUT_DIR, f"figure_4panel_{timestamp}.png")

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outfile}")

print("\n===================================")
print("ALL 4-PANEL FIGURES GENERATED")
print("===================================")