# make_split_pairs_fixed.py
import os, argparse, random
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(d: Path):
    paths = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_save(img_bgr: np.ndarray, out1: Path, out2: Path, stem: str,
               ratio: float, as_png: bool, quality: int):
    """
    ratio: 보이는 높이 비율 (0.55~0.65 범위)
    out1: 위쪽 조각 저장 폴더
    out2: 아래쪽 조각 저장 폴더
    """
    h, w = img_bgr.shape[:2]
    crop_h = max(1, int(round(h * ratio)))
    crop_h = min(h, crop_h)

    # 위쪽 조각: [0:crop_h]
    top = img_bgr[0:crop_h, 0:w]
    # 아래쪽 조각: [h-crop_h:h]
    bottom = img_bgr[h - crop_h : h, 0:w]

    if as_png:
        cv2.imwrite(str(out1 / f"{stem}.png"), top)
        cv2.imwrite(str(out2 / f"{stem}.png"), bottom)
    else:
        cv2.imwrite(str(out1 / f"{stem}.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, quality])
        cv2.imwrite(str(out2 / f"{stem}.jpg"), bottom, [cv2.IMWRITE_JPEG_QUALITY, quality])

def main():
    ap = argparse.ArgumentParser(description="원본 이미지를 위/아래로 55~65% 범위로 랜덤 분할")
    ap.add_argument("--src", type=str, default="data_carbon/images", help="입력 폴더")
    ap.add_argument("--out1", type=str, default="s_data/img1", help="출력 폴더(윗부분)")
    ap.add_argument("--out2", type=str, default="s_data/img2", help="출력 폴더(아랫부분)")
    ap.add_argument("--ratio-min", type=float, default=0.55, help="최소 보이는 비율")
    ap.add_argument("--ratio-max", type=float, default=0.65, help="최대 보이는 비율")
    ap.add_argument("--mean", type=float, default=0.60, help="랜덤 비율의 평균(가우시안 샘플 중심)")
    ap.add_argument("--std", type=float, default=0.03, help="랜덤 비율의 표준편차")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    ap.add_argument("--jpg", action="store_true", help="JPG로 저장(기본 PNG)")
    ap.add_argument("--quality", type=int, default=95, help="JPG 품질")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src)
    out1 = Path(args.out1)
    out2 = Path(args.out2)
    ensure_dir(out1); ensure_dir(out2)

    imgs = list_images(src)
    if not imgs:
        print(f"[!] 입력 폴더에 이미지가 없습니다: {src}")
        return

    # 범위 정리
    rmin = max(0.55, float(args.ratio_min))
    rmax = min(0.65, float(args.ratio_max))
    if rmax <= rmin:
        raise ValueError("--ratio-max는 --ratio-min보다 커야 합니다.")

    print(f"[INFO] 입력 {len(imgs)}장 | 비율 범위 [{rmin:.2f}, {rmax:.2f}] | 평균 {args.mean:.2f} ± {args.std:.2f}")
    cnt = 0
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[skip] 읽기 실패: {p.name}")
            continue

        # RGBA면 RGB로 변환
        if img.ndim == 3 and img.shape[2] == 4:
            b,g,r,a = cv2.split(img)
            img = cv2.merge([b,g,r])

        # 가우시안 샘플 → 범위 클리핑 (55~65%)
        r = float(np.random.normal(args.mean, args.std))
        r = max(rmin, min(rmax, r))

        # 저장
        stem = p.stem
        try:
            split_save(img, out1, out2, stem, r, as_png=not args.jpg, quality=args.quality)
            cnt += 2
            print(f"  {stem}: ratio={r:.3f} (겹침 {(2*r-1)*100:.1f}%)")
        except Exception as e:
            print(f"[skip] {p.name}: {e}")

    print(f"\n[DONE] 생성 완료: {cnt} 파일 (각 원본당 위/아래 1장씩)")

if __name__ == "__main__":
    main()