import os
import pandas as pd
import shutil

# ==========================================
# FINAL METRICS
# ==========================================

metrics = {
    "Metric": [
        "RMSE_U",
        "RMSE_V",
        "MAE_U",
        "MAE_V",
        "CORR_U",
        "CORR_V"
    ],
    "Value": [
        0.067118,
        0.061541,
        0.039712,
        0.037825,
        0.7134,
        0.6287
    ]
}

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

FINAL_DIR = "outputs/final_results"

os.makedirs(FINAL_DIR, exist_ok=True)

# ==========================================
# SAVE CSV
# ==========================================

df = pd.DataFrame(metrics)

csv_path = os.path.join(
    FINAL_DIR,
    "final_metrics.csv"
)

df.to_csv(csv_path, index=False)

# ==========================================
# COPY PUBLICATION FIGURES
# ==========================================

FIGURES = [
    "20221219150000",
    "20221231150000",
    "20221231180000"
]

SOURCE_DIR = "outputs/unet_predictions_publication"

for ts in FIGURES:

    src = os.path.join(
        SOURCE_DIR,
        f"prediction_compare_{ts}.png"
    )

    dst = os.path.join(
        FINAL_DIR,
        f"prediction_compare_{ts}.png"
    )

    if os.path.exists(src):

        shutil.copy2(src, dst)

        print(f"Copied: {ts}")

    else:

        print(f"Missing: {ts}")

# ==========================================
# SUMMARY
# ==========================================

print("\n===================================")
print("FINAL RESULTS PACKAGE CREATED")
print("===================================")

print("\nMetrics:")
print(df)

print("\nSaved Folder:")
print(FINAL_DIR)