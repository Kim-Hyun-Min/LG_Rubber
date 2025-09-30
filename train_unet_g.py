# 맨 위 import 근처에 추가
import os, glob, random, warnings
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
from torchvision.transforms import InterpolationMode

MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
IMG_EXTS  = [".png", ".jpg", ".jpeg", ".bmp"]

def find_mask_for(stem: str, mask_dir: str):
    """같은 stem의 마스크 파일을 확장자 순회하며 찾는다."""
    for ext in MASK_EXTS:
        p = Path(mask_dir) / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None

class RubberSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512, augment=False, strict=False):
        self.size = size
        self.augment = augment
        self.pairs = []

        # 이미지 목록 수집
        img_paths = []
        for ext in IMG_EXTS:
            img_paths.extend(sorted(Path(img_dir).glob(f"*{ext}")))
        img_paths = [p for p in img_paths if p.is_file()]

        missing = []
        for p in img_paths:
            stem = p.stem
            mask_path = find_mask_for(stem, mask_dir)
            if mask_path is None:
                missing.append(stem)
                if strict:
                    raise FileNotFoundError(f"Mask not found for: {stem}")
                else:
                    continue
            self.pairs.append((str(p), mask_path))

        if missing:
            warnings.warn(f"마스크 누락 {len(missing)}개: 예) {missing[:5]} ...")

        if not self.pairs:
            raise RuntimeError("유효한 (이미지-마스크) 쌍을 찾지 못했습니다.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 0/255

        img_t = TF.to_tensor(img)                     # (3,H,W) float[0..1]
        mask_t = TF.to_tensor(mask)                   # (1,H,W) float[0..1]

        # 리사이즈 (이미지는 bilinear+antialias, 마스크는 nearest)
        img_t  = TF.resize(img_t,  [self.size, self.size],
                           interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask_t = TF.resize(mask_t, [self.size, self.size],
                           interpolation=InterpolationMode.NEAREST, antialias=False)

        # 이진화
        mask_t = (mask_t > 0.5).float()

        # 간단 Aug
        if self.augment and random.random() < 0.5:
            img_t = TF.hflip(img_t); mask_t = TF.hflip(mask_t)
        if self.augment and random.random() < 0.5:
            img_t = TF.vflip(img_t); mask_t = TF.vflip(mask_t)

        return img_t, mask_t


# -------------------------
# Simple UNet (경량)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = DoubleConv(base*2, base)

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)

        return self.outc(d1)  # logits

# -------------------------
# Loss (BCE + Dice)
# -------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.
        num = 2 * (probs*targets).sum() + smooth
        den = probs.sum() + targets.sum() + smooth
        dice = 1 - (num / den)
        return bce + dice

# -------------------------
# Train
# -------------------------
def main():
    img_dir = 'data_g/images'
    mask_dir = 'data_g/masks'
    save_dir = 'runs_seg'
    os.makedirs(save_dir, exist_ok=True)

    size = 512
    epochs = 30
    batch_size = 4
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = RubberSegDataset(img_dir, mask_dir, size=size, augment=True)
    # 간단 split (8:2)
    n = len(dataset)
    n_train = int(n*0.8)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss()

    best_val = 1e9
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optim.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optim.step()
            tr_loss += loss.item()*imgs.size(0)
        tr_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [valid]"):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()*imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{ep}] train={tr_loss:.4f}  val={val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'size': size}, os.path.join(save_dir, 'best_unet.pth'))
            print(f"  -> saved best to {os.path.join(save_dir,'best_unet.pth')}")

if __name__ == "__main__":
    main()
