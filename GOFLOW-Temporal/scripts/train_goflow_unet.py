import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = "training_data"

BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = "models/goflow_v2_9km"
os.makedirs(MODEL_DIR, exist_ok=True)

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

        return (
            torch.tensor(X),
            torch.tensor(Y)
        )

# =====================================================
# U-NET BLOCK
# =====================================================

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)

# =====================================================
# U-NET
# =====================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            128,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            2,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(b)

        if d3.shape[2:] != e3.shape[2:]:

            d3 = F.interpolate(
                d3,
                size=e3.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[2:] != e2.shape[2:]:

            d2 = F.interpolate(
                d2,
                size=e2.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[2:] != e1.shape[2:]:

            d1 = F.interpolate(
                d1,
                size=e1.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)

# =====================================================
# LOAD DATA
# =====================================================

dataset = GOFLOWDataset(DATA_DIR)

print(f"\nTotal Samples = {len(dataset)}")

train_size = int(
    0.8 * len(dataset)
)

val_size = int(
    0.1 * len(dataset)
)

test_size = (
    len(dataset)
    - train_size
    - val_size
)

train_ds, val_ds, test_ds = random_split(

    dataset,

    [
        train_size,
        val_size,
        test_size
    ],

    generator=torch.Generator().manual_seed(42)
)

print(f"Train Samples : {len(train_ds)}")
print(f"Val Samples   : {len(val_ds)}")
print(f"Test Samples  : {len(test_ds)}")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE
)

# =====================================================
# MODEL
# =====================================================

model = UNet().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val = np.inf

# =====================================================
# TRAIN
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for X, Y in train_loader:

        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(
            pred,
            Y
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X, Y in val_loader:

            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            pred = model(X)

            loss = criterion(
                pred,
                Y
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(

        f"Epoch {epoch+1:03d} | "
        f"Train Loss = {train_loss:.6f} | "
        f"Val Loss = {val_loss:.6f}"

    )

    if val_loss < best_val:

        best_val = val_loss

        torch.save(

            model.state_dict(),

            os.path.join(
                MODEL_DIR,
                "best_goflow_unet_9km.pth"
            )

        )

        print(
            f"Saved Best Model "
            f"(Val Loss = {val_loss:.6f})"
        )

print("\n===================================")
print("GOFLOW TRAINING COMPLETE")
print("===================================")

print(
    f"Best Validation Loss: "
    f"{best_val:.6f}"
)

print(
    "Model Saved To:"
)

print(
    f"{MODEL_DIR}/best_goflow_unet_9km.pth"
)