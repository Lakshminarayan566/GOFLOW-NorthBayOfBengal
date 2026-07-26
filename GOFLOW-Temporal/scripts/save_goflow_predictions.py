import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import random_split

# =====================================================
# SETTINGS
# =====================================================

DATA_DIR = "training_data"
MODEL_FILE = "models/goflow_v2_9km/best_goflow_unet_9km.pth"

OUT_DIR = "outputs/predictions_9km"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# DATASET
# =====================================================

class GOFLOWDataset(Dataset):

    def __init__(self, data_dir):

        self.files = sorted(
            glob.glob(
                os.path.join(data_dir, "*.npz")
            )
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        d = np.load(self.files[idx])

        X = np.nan_to_num(
            d["X"].astype(np.float32)
        )

        Y = np.nan_to_num(
            d["Y"].astype(np.float32)
        )

        timestamp = str(d["timestamp"])

        return (
            torch.tensor(X),
            torch.tensor(Y),
            timestamp
        )

# =====================================================
# MODEL
# =====================================================

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(3,32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32,64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64,128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128,256)

        self.up3 = nn.ConvTranspose2d(
            256,128,2,stride=2
        )

        self.dec3 = DoubleConv(
            256,128
        )

        self.up2 = nn.ConvTranspose2d(
            128,64,2,stride=2
        )

        self.dec2 = DoubleConv(
            128,64
        )

        self.up1 = nn.ConvTranspose2d(
            64,32,2,stride=2
        )

        self.dec1 = DoubleConv(
            64,32
        )

        self.final = nn.Conv2d(
            32,2,1
        )

    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)

        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(
                d3,
                size=e3.shape[2:]
            )

        d3 = torch.cat([d3,e3],1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[2:]
            )

        d2 = torch.cat([d2,e2],1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[2:]
            )

        d1 = torch.cat([d1,e1],1)
        d1 = self.dec1(d1)

        return self.final(d1)

# =====================================================
# LOAD TEST SET
# =====================================================

dataset = GOFLOWDataset(DATA_DIR)

train_size = 197
val_size = 24
test_size = 26

_, _, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# =====================================================
# LOAD MODEL
# =====================================================

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model.eval()

# =====================================================
# SAVE PREDICTIONS
# =====================================================

saved = 0

with torch.no_grad():

    for X, Y, timestamp in test_ds:

        X = X.unsqueeze(0).to(DEVICE)

        pred = model(X)

        pred = pred.cpu().numpy()[0]
        truth = Y.numpy()

        np.savez_compressed(
            os.path.join(
                OUT_DIR,
                f"prediction_{timestamp}.npz"
            ),
            pred_u=pred[0],
            pred_v=pred[1],
            true_u=truth[0],
            true_v=truth[1],
            timestamp=timestamp
        )

        saved += 1

print("\n==============================")
print("GOFLOW PREDICTIONS SAVED")
print("==============================")
print("Files:", saved)
print("Folder:", OUT_DIR)