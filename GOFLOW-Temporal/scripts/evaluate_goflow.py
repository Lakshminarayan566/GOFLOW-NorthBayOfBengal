import os
import glob
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import random_split

# =====================================================
# SETTINGS
# =====================================================

DATA_DIR = "training_data"

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

        return (
            torch.tensor(X),
            torch.tensor(Y)
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

        import torch.nn.functional as F

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(
            self.pool3(e3)
        )

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

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

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
            "models/goflow_v2_9km/best_goflow_unet_9km.pth",
            map_location=DEVICE
    )
)

model.eval()

# =====================================================
# METRICS
# =====================================================

pred_u = []
pred_v = []

true_u = []
true_v = []

with torch.no_grad():

    for X, Y in test_ds:

        X = X.unsqueeze(0).to(DEVICE)

        pred = model(X)

        pred = pred.cpu().numpy()[0]
        truth = Y.numpy()

        pred_u.append(pred[0].ravel())
        pred_v.append(pred[1].ravel())

        true_u.append(truth[0].ravel())
        true_v.append(truth[1].ravel())

pred_u = np.concatenate(pred_u)
pred_v = np.concatenate(pred_v)

true_u = np.concatenate(true_u)
true_v = np.concatenate(true_v)

rmse_u = np.sqrt(
    np.mean((pred_u-true_u)**2)
)

rmse_v = np.sqrt(
    np.mean((pred_v-true_v)**2)
)

mae_u = np.mean(
    np.abs(pred_u-true_u)
)

mae_v = np.mean(
    np.abs(pred_v-true_v)
)

corr_u = np.corrcoef(
    pred_u,
    true_u
)[0,1]

corr_v = np.corrcoef(
    pred_v,
    true_v
)[0,1]

print("\n==========================")
print("GOFLOW TEST RESULTS")
print("==========================")

print(f"RMSE U : {rmse_u:.6f}")
print(f"RMSE V : {rmse_v:.6f}")

print(f"MAE U  : {mae_u:.6f}")
print(f"MAE V  : {mae_v:.6f}")

print(f"CORR U : {corr_u:.4f}")
print(f"CORR V : {corr_v:.4f}")