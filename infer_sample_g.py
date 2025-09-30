# infer_alpha.py
import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import cv2

# -------------------------
# UNet 정의 (train_unet.py와 동일)
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
        b  = self.bottleneck(self.pool(e4))
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.outc(d1)

# -------------------------
# 로드 / 추론
# -------------------------
def load_model(weight_path, device):
    ckpt = torch.load(weight_path, map_location='cpu')
    size = ckpt.get('size', 512)
    model = UNet().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, size

def infer_one(model, img_pil, size, device='cuda', thresh=0.5):
    w, h = img_pil.size
    img = TF.to_tensor(img_pil).unsqueeze(0)
    img_resized = TF.resize(img, [size, size], antialias=True)
    with torch.no_grad():
        logits = model(img_resized.to(device))
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask01 = (prob >= thresh).astype(np.uint8)
    return mask01

# -------------------------
# 마스크 후처리
# -------------------------
def postprocess_mask(mask01: np.ndarray, min_area=500, close_k=3):
    """
    mask01: (H,W) 0/1
    - 가장 큰 성분만 남기고 노이즈 제거
    - CLOSE 연산으로 경계 매끈화
    """
    m = (mask01.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num > 1:
        keep = np.zeros_like(m)
        best_idx, best_area = -1, 0
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area; best_idx = i
        if best_idx > 0 and best_area >= min_area:
            keep[labels == best_idx] = 255
        m = keep
    if close_k > 0:
        kernel = np.ones((close_k, close_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return (m > 127).astype(np.uint8)

# -------------------------
# 저장 (alpha만)
# -------------------------
def save_alpha(img_pil, mask01, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    img_np = np.array(img_pil)
    mask01 = postprocess_mask(mask01, min_area=500, close_k=3)
    alpha = (mask01 * 255).astype(np.uint8)
    rgba = np.dstack([img_np, alpha])   # (H,W,4)
    Image.fromarray(rgba).save(os.path.join(out_dir, f"{stem}_alpha.png"))

# -------------------------
# 메인
# -------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, size = load_model('runs_seg/best_unet.pth', device=device)

    in_dir  = 'out_brightness'   # 입력 폴더
    out_dir = 'pred_alpha'       # 결과 폴더
    limit   = 30                 # 앞에서 30장만
    thresh  = 0.5

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    paths = [p for p in sorted(glob.glob(os.path.join(in_dir, '*'))) if p.lower().endswith(exts)]
    paths = paths[:limit]

    print(f"Infer {len(paths)} images -> alpha PNG 저장")
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        img_pil = Image.open(p).convert('RGB')
        mask01 = infer_one(model, img_pil, size=size, device=device, thresh=thresh)
        save_alpha(img_pil, mask01, out_dir, stem)
        print(f" -> saved: {stem}_alpha.png")

if __name__ == "__main__":
    main()
