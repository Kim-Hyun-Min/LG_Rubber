#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import cv2
from pathlib import Path

# -------------------------
# UNet 정의
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
def load_model(weight_path: Path, device: str):
    ckpt = torch.load(str(weight_path), map_location='cpu')
    size = ckpt.get('size', 512)
    model = UNet().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, size

@torch.inference_mode()
def infer_one(model, img_pil: Image.Image, size: int, device='cuda', thresh=0.5):
    w, h = img_pil.size
    img = TF.to_tensor(img_pil).unsqueeze(0)                     # (1,3,H,W) float32[0..1]
    img_resized = TF.resize(img, [size, size], antialias=True)
    logits = model(img_resized.to(device))
    prob = torch.sigmoid(logits)[0,0].float().cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask01 = (prob >= float(thresh)).astype(np.uint8)            # 0/1
    return mask01

# -------------------------
# 마스크 후처리
# -------------------------
def postprocess_mask(mask01: np.ndarray, min_area=500, close_k=3):
    m = (mask01.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num > 1:
        keep = np.zeros_like(m)
        best_idx, best_area = -1, 0
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area; best_idx = i
        if best_idx > 0 and best_area >= int(min_area):
            keep[labels == best_idx] = 255
        m = keep
    if close_k and close_k > 0:
        kernel = np.ones((close_k, close_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return (m > 127).astype(np.uint8)  # 0/1

# -------------------------
# 저장 (RGBA + whitebg 둘 다 옵션)
# -------------------------
def save_outputs(img_pil, mask01, out_dir: Path, stem: str, save_rgba: bool, save_whitebg: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    mask01 = postprocess_mask(mask01, min_area=500, close_k=3)
    img_np = np.array(img_pil)

    if save_rgba:
        a8 = (mask01 * 255).astype(np.uint8)
        rgba = np.dstack([img_np, a8])
        Image.fromarray(rgba).save(out_dir / f"{stem}_alpha.png")

    if save_whitebg:
        alpha = mask01.astype(np.float32)[..., None]
        white_bg = np.ones_like(img_np, dtype=np.float32) * 255.0
        out_rgb = (img_np * alpha + white_bg * (1 - alpha)).clip(0, 255).astype(np.uint8)
        Image.fromarray(out_rgb).save(out_dir / f"{stem}_whitebg.png")

# -------------------------
# 메인
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Infer alpha masks for a folder")
    ap.add_argument("--inp", required=True, help="입력 폴더")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--weights", default="runs_seg/best_unet.pth", help="모델 가중치 경로")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (0이면 전체)")
    ap.add_argument("--thresh", type=float, default=0.5, help="마스크 임계값")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.tif,.tiff", help="확장자 목록(쉼표)")
    ap.add_argument("--save-rgba", action="store_true", help="RGBA(alpha) PNG 저장")
    ap.add_argument("--save-whitebg", action="store_true", help="흰 배경 합성본 저장")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    in_dir  = Path(args.inp)
    out_dir = Path(args.out)
    weight  = Path(args.weights)

    if not in_dir.exists():
        print(f"[X] 입력 폴더 없음: {in_dir}"); return
    if not weight.exists():
        print(f"[X] 가중치 없음: {weight}"); return
    if not (args.save_rgba or args.save_whitebg):
        # 기본: RGBA 저장
        args.save_rgba = True

    model, size = load_model(weight, device=device)

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    paths = [p for p in sorted(in_dir.glob("*")) if p.suffix.lower() in exts]
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    print(f"Infer {len(paths)} images from {in_dir} → {out_dir}")
    for i, p in enumerate(paths, 1):
        stem = p.stem
        img_pil = Image.open(p).convert('RGB')
        mask01 = infer_one(model, img_pil, size=size, device=device, thresh=args.thresh)
        save_outputs(img_pil, mask01, out_dir, stem, save_rgba=args.save_rgba, save_whitebg=args.save_whitebg)
        print(f"[{i:04}/{len(paths)}] saved: {stem}")


if __name__ == "__main__":
    main()
