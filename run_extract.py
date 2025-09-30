#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원클릭 추론 파이프라인 (학습 건너뜀)
- single:  한 폴더 입력 → 밝기 정규화 → UNet 추론(알파 PNG) → (선택) RGB 평탄화/타이트크롭
- split : 윗/아랫 폴더 쌍 → 각자 정규화 → 스티칭 → UNet 추론 → (선택) 평탄화/타이트크롭

필수:
- runs_seg/best_unet.pth (미리 학습된 가중치)
- normalize_brightness.py, (선택) prep_rubber_crops.py, improved_rubber_stamp_stitching.py (동일 폴더)
"""

import argparse
import sys, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision.transforms.functional as TF
import cv2

# ---------- 경로 상수 ----------
PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent

# import 경로 보장
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 외부 스크립트 절대경로
SCRIPT_STITCH = ROOT / "improved_rubber_stamp_stitching.py"
SCRIPT_PREP   = ROOT / "prep_rubber_crops.py"

# normalize_brightness 함수 직접 호출
from normalize_brightness import process_directory as nb_process_dir

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------- UNet (train_unet_g.py와 동일 아키텍처) ----------
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


# ---------- 모델 로드 ----------
def load_model(weight_path: Path, device: str):
    # FutureWarning 회피: weights_only=True 지원 시 우선 시도
    try:
        ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=True)  # torch>=2.4
    except TypeError:
        ckpt = torch.load(str(weight_path), map_location='cpu')  # 호환 모드

    size = 512
    state = None
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        state = ckpt['model']
        size = int(ckpt.get('size', 512))
    elif isinstance(ckpt, dict):  # 순수 state_dict 저장 형태
        state = ckpt
    else:
        state = ckpt

    model = UNet().to(device)
    model.load_state_dict(state)
    model.eval()
    return model, size


# ---------- 추론(폴더→알파PNG) ----------
def infer_dir_to_alpha(in_dir: Path, out_dir: Path, model, size: int, device='cuda', thresh=0.5):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted(in_dir.glob("*")) if p.suffix.lower() in IMG_EXTS]
    print(f"  추론 대상 {len(paths)}장")
    for p in paths:
        try:
            img_pil = Image.open(p).convert('RGB')
        except Exception:
            print(f"   [skip] 불러오기 실패: {p.name}")
            continue
        w, h = img_pil.size
        img = TF.to_tensor(img_pil).unsqueeze(0)
        img_resized = TF.resize(img, [size, size], antialias=True)
        with torch.no_grad():
            logits = model(img_resized.to(device))
            prob = torch.sigmoid(logits)[0,0].cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask01 = (prob >= thresh).astype(np.uint8)

        # 후처리: 가장 큰 성분 유지 + CLOSE
        m = (mask01 * 255).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if num > 1:
            best_idx, best_area = -1, 0
            for i in range(1, num):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > best_area:
                    best_area = area; best_idx = i
            keep = np.zeros_like(m)
            if best_idx > 0:
                keep[labels == best_idx] = 255
            m = keep
        kernel = np.ones((3,3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        rgba = np.dstack([np.array(img_pil), m])
        save_path = out_dir / f"{p.stem}_alpha.png"
        Image.fromarray(rgba).save(save_path)
    print(f"  -> 알파 PNG 저장: {out_dir}")


# ---------- 밝기 정규화 ----------
def normalize_folder(src: Path, dst: Path,
                     method="l-abs",
                     clip_limit=2.0, tile_grid=8,
                     target_L=None, target_std=None,
                     abs_p_low=5.0, abs_p_high=95.0,
                     abs_t_low=60.0, abs_t_high=200.0):
    dst.mkdir(parents=True, exist_ok=True)
    print(f"\n[정규화] {src} → {dst} (method={method})")
    nb_process_dir(
        input_dir=src,
        output_dir=dst,
        method=method,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid,
        reference_path=None,
        overwrite=True,
        target_L_mean=target_L,
        target_L_std=target_std,
        abs_p_low=abs_p_low,
        abs_p_high=abs_p_high,
        abs_t_low=abs_t_low,
        abs_t_high=abs_t_high,
    )


# ---------- 스티칭 호출 ----------
def stitch_pairs(top_dir: Path, bot_dir: Path, out_dir: Path,
                 original_height: int|None, height_file: Path|None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not SCRIPT_STITCH.exists():
        raise FileNotFoundError(f"스티칭 스크립트를 찾을 수 없습니다: {SCRIPT_STITCH}")
    args = [PYTHON, str(SCRIPT_STITCH),
            "--top-dir", str(top_dir),
            "--bot-dir", str(bot_dir),
            "--out", str(out_dir)]
    if original_height:
        args += ["--original-height", str(original_height)]
    if height_file:
        args += ["--height-file", str(height_file)]
    print(f"\n[스티칭] {top_dir.name} + {bot_dir.name} → {out_dir.name}")
    subprocess.run(args, check=True)


# ---------- 알파→RGB 평탄화/타이트크롭 호출 ----------
def flatten_alpha(in_dir: Path, out_dir: Path, bg="white", tight=True, jpg=False, quality=95):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not SCRIPT_PREP.exists():
        raise FileNotFoundError(f"prep_rubber_crops.py를 찾을 수 없습니다: {SCRIPT_PREP}")
    args = [PYTHON, str(SCRIPT_PREP),
            "--inp", str(in_dir),
            "--out", str(out_dir),
            "--bg", bg]
    if tight:
        args.append("--tight")
    if jpg:
        args += ["--jpg", "--quality", str(quality)]
    print(f"\n[평탄화/크롭] {in_dir} → {out_dir}  (script={SCRIPT_PREP})")
    subprocess.run(args, check=True)


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single","split"], required=True,
                    help="single: 한 폴더 입력 / split: 윗조각-아랫조각 폴더 쌍")
    ap.add_argument("--inp", type=str, help="[single] 원본 이미지 폴더")
    ap.add_argument("--top-dir", type=str, help="[split] 윗조각 폴더")
    ap.add_argument("--bot-dir", type=str, help="[split] 아랫조각 폴더")
    ap.add_argument("--stitch-out", type=str, default="simple_overlap",
                    help="[split] 스티칭 결과 폴더 (기본: simple_overlap)")
    ap.add_argument("--original-height", type=int, default=None,
                    help="[split] 원본 고무 높이 힌트")
    ap.add_argument("--height-file", type=str, default=None,
                    help="[split] 파일별 높이 CSV(stem,height)")
    ap.add_argument("--model", type=str, default=str(ROOT / "runs_seg" / "best_unet.pth"),
                    help="미리 학습된 UNet 가중치 경로")
    ap.add_argument("--bg", type=str, default="white",
                    help="평탄화 배경색: white/black/gray/#RRGGBB/auto")
    ap.add_argument("--no-tight", action="store_true",
                    help="타이트 크롭 비활성화")
    ap.add_argument("--jpg", action="store_true",
                    help="결과를 JPG로 저장(기본 PNG)")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPG 품질")
    ap.add_argument("--skip-flatten", action="store_true",
                    help="알파→RGB 평탄화/타이트크롭 단계를 생략")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, size = load_model(Path(args.model), device=device)

    if args.mode == "single":
        if not args.inp:
            ap.error("--inp 폴더를 지정하세요.")
        inp = Path(args.inp)
        out_b   = ROOT / "out_brightness"
        preddir = ROOT / "pred_alpha"
        out_rgb = ROOT / "data_carbon" / "images"

        # 1) 밝기 정규화
        normalize_folder(inp, out_b, method="l-abs")

        # 2) 추론(알파 PNG)
        infer_dir_to_alpha(out_b, preddir, model, size=size, device=device, thresh=0.5)

        # 3) (선택) 평탄화/타이트크롭
        if not args.skip_flatten:
            flatten_alpha(preddir, out_rgb, bg=args.bg, tight=(not args.no_tight),
                          jpg=args.jpg, quality=args.quality)
            print("\n✅ 완료(single):")
            print(f" - 알파: {preddir}")
            print(f" - RGB : {out_rgb}")
        else:
            print("\n✅ 완료(single, 평탄화 생략):")
            print(f" - 알파: {preddir}")

    else:  # split
        if not args.top_dir or not args.bot_dir:
            ap.error("--top-dir --bot-dir 모두 지정하세요.")
        top = Path(args.top_dir)
        bot = Path(args.bot_dir)

        out_top     = ROOT / "out_brightness_top"
        out_bot     = ROOT / "out_brightness_bot"
        stitch_out  = Path(args.stitch_out)
        preddir     = ROOT / "pred_alpha_stitched"
        out_rgb     = ROOT / "data_carbon" / "images_stitched"

        # 1) 각 폴더 밝기 정규화
        normalize_folder(top, out_top, method="l-abs")
        normalize_folder(bot, out_bot, method="l-abs")

        # 2) 스티칭(정규화 결과 기준)
        height_file = Path(args.height_file) if args.height_file else None
        stitch_pairs(out_top, out_bot, stitch_out,
                     original_height=args.original_height,
                     height_file=height_file)

        # 3) 추론(알파 PNG)
        infer_dir_to_alpha(stitch_out, preddir, model, size=size, device=device, thresh=0.5)

        # 4) (선택) 평탄화/타이트크롭
        if not args.skip_flatten:
            flatten_alpha(preddir, out_rgb, bg=args.bg, tight=(not args.no_tight),
                          jpg=args.jpg, quality=args.quality)
            print("\n✅ 완료(split):")
            print(f" - 스티칭: {stitch_out}")
            print(f" - 알파  : {preddir}")
            print(f" - RGB   : {out_rgb}")
        else:
            print("\n✅ 완료(split, 평탄화 생략):")
            print(f" - 스티칭: {stitch_out}")
            print(f" - 알파  : {preddir}")

if __name__ == "__main__":
    main()
