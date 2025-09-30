#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubber image brightness-only normalization batch tool
- 색상(a/b)은 그대로 두고 LAB의 L(밝기)만 조정
- 지원 방식:
  auto(=CLAHE-L), clahe, l-fixed, l-histmatch, l-abs
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


# ---------------- Common utils ----------------

# 출력 경로가 없으면 상위 폴더까지 생성
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# 확장자 검사로 이미지 파일인지 판별
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 하위 폴더까지 모두 순회하며 이미지 파일 목록을 수집·정렬
def collect_images(input_dir: Path) -> list[Path]:
    all_paths: list[Path] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if is_image_file(p):
                all_paths.append(p)
    return sorted(all_paths)


# ---------------- LAB helpers (brightness-only) ----------------

# BGR → LAB 변환 후 각 채널(L, a, b)을 분리.
def bgr2lab_channels(img_bgr: np.ndarray):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return l, a, b

# L/a/b 채널을 다시 합쳐 LAB → BGR로 되돌립니다.
def lab_merge_to_bgr(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------- Methods (operate on L only) ----------------

def apply_clahe_L(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """L 채널에만 CLAHE 적용(색상 보존)"""
    l, a, b = bgr2lab_channels(image_bgr)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l)
    return lab_merge_to_bgr(l_eq, a, b)


def apply_L_fixed(image_bgr: np.ndarray, target_mean: float = 170.0, target_std: float | None = None) -> np.ndarray:
    """
    L(0~255)의 평균/표준편차를 지정 타깃에 맞춤 (색상 보존)
    - target_std가 None이면 평균만 일치
    - target_std가 지정되면 평균/표준편차 모두 매칭
    """
    l, a, b = bgr2lab_channels(image_bgr)
    l_f = l.astype(np.float32)

    cur_mean = float(l_f.mean())
    cur_std = float(l_f.std() + 1e-6)

    if target_std is None:
        shift = (target_mean - cur_mean)
        l_new = l_f + shift
    else:
        normed = (l_f - cur_mean) / cur_std
        l_new = normed * float(target_std) + float(target_mean)

    l_new = np.clip(l_new, 0, 255).astype(np.uint8)
    return lab_merge_to_bgr(l_new, a, b)


def build_cdf(img_channel: np.ndarray) -> np.ndarray:
    """0~255 L 채널의 누적분포(CDF) 계산"""
    hist = cv2.calcHist([img_channel], [0], None, [256], [0, 256]).ravel()
    cdf = hist.cumsum()
    cdf /= (cdf[-1] + 1e-6)
    return cdf


def lut_from_cdf_match(cdf_src: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """src CDF를 ref CDF에 매칭하는 256 크기 LUT 생성"""
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and cdf_ref[j] < cdf_src[i]:
            j += 1
        lut[i] = j
    return lut


def apply_L_histmatch(image_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    """
    참조 이미지의 L 히스토그램으로만 매칭 (색상 보존)
    scikit-image 불필요: OpenCV 히스토그램 + LUT로 구현
    """
    l, a, b = bgr2lab_channels(image_bgr)
    l_ref, _, _ = bgr2lab_channels(reference_bgr)

    cdf_src = build_cdf(l)
    cdf_ref = build_cdf(l_ref)
    lut = lut_from_cdf_match(cdf_src, cdf_ref)
    l_new = cv2.LUT(l, lut)

    return lab_merge_to_bgr(l_new, a, b)


def percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x.reshape(-1), q))


def apply_L_absolute(
    image_bgr: np.ndarray,
    p_low: float = 5.0,
    p_high: float = 95.0,
    t_low: float = 60.0,
    t_high: float = 200.0,
) -> np.ndarray:
    """
    퍼센타일 기반 절대 밝기 정렬 (체감 밝기 통일에 가장 효과적)
    - 소스 L에서 [p_low, p_high] 구간을 고정 목표 [t_low, t_high]로 선형 매핑
    - 색상(a/b) 불변
    - 모든 값은 OpenCV LAB(0~255) 기준
    """
    l, a, b = bgr2lab_channels(image_bgr)

    l_f = l.astype(np.float32)
    pl = percentile(l_f, p_low)
    ph = percentile(l_f, p_high)
    if ph - pl < 1e-6:
        # 분포가 너무 좁으면 약한 CLAHE로 대체
        return apply_clahe_L(image_bgr, clip_limit=1.5, tile_grid_size=8)

    scale = float(t_high - t_low) / float(ph - pl)
    l_new = (l_f - pl) * scale + t_low
    l_new = np.clip(l_new, 0, 255).astype(np.uint8)

    return lab_merge_to_bgr(l_new, a, b)


# ---------------- Dispatcher ----------------

def normalize_image(
    image_bgr: np.ndarray,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
    reference_bgr: np.ndarray | None,
    target_L_mean: float | None,
    target_L_std: float | None,
    abs_p_low: float,
    abs_p_high: float,
    abs_t_low: float,
    abs_t_high: float,
) -> np.ndarray:
    if method == 'auto':
        # 안전 기본값: CLAHE-L
        return apply_clahe_L(image_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method == 'clahe':
        return apply_clahe_L(image_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method == 'l-fixed':
        tm = 170.0 if target_L_mean is None else float(target_L_mean)
        ts = None if target_L_std is None else float(target_L_std)
        return apply_L_fixed(image_bgr, target_mean=tm, target_std=ts)
    elif method == 'l-histmatch':
        if reference_bgr is None:
            raise ValueError('l-histmatch 방식은 --reference 이미지가 필요합니다.')
        return apply_L_histmatch(image_bgr, reference_bgr)
    elif method == 'l-abs':
        return apply_L_absolute(
            image_bgr,
            p_low=abs_p_low, p_high=abs_p_high,
            t_low=abs_t_low, t_high=abs_t_high,
        )
    else:
        raise ValueError(f'Unknown method: {method}. Use one of: auto, clahe, l-fixed, l-histmatch, l-abs')


# ---------------- Batch processing ----------------

def process_directory(
    input_dir: Path,
    output_dir: Path,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
    reference_path: Path | None,
    overwrite: bool,
    target_L_mean: float | None,
    target_L_std: float | None,
    abs_p_low: float,
    abs_p_high: float,
    abs_t_low: float,
    abs_t_high: float,
) -> None:
    ensure_dir(output_dir)
    reference_bgr = None
    if method == 'l-histmatch':
        if reference_path is None or not reference_path.exists():
            raise FileNotFoundError('히스토그램 매칭용 --reference 이미지가 존재하지 않습니다.')
        reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference_bgr is None:
            raise RuntimeError('참조 이미지를 읽을 수 없습니다.')

    inputs = collect_images(input_dir)
    if not inputs:
        raise RuntimeError('입력 디렉터리에서 이미지 파일을 찾지 못했습니다.')

    for src_path in tqdm(inputs, desc=f'Normalizing ({method})'):
        rel = src_path.relative_to(input_dir)
        dst_path = output_dir / rel
        ensure_dir(dst_path.parent)
        if dst_path.exists() and not overwrite:
            continue
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            out = normalize_image(
                img,
                method=method,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                reference_bgr=reference_bgr,
                target_L_mean=target_L_mean,
                target_L_std=target_L_std,
                abs_p_low=abs_p_low,
                abs_p_high=abs_p_high,
                abs_t_low=abs_t_low,
                abs_t_high=abs_t_high,
            )
            cv2.imwrite(str(dst_path), out)
        except Exception as e:
            print(f'Failed {src_path}: {e}')


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rubber image brightness-only normalization batch tool')
    parser.add_argument('-m', '--method', type=str, default='auto',
                        choices=['auto', 'clahe', 'l-fixed', 'l-histmatch', 'l-abs'],
                        help='정규화 방식 (밝기 전용)')
    # CLAHE
    parser.add_argument('--clip-limit', type=float, default=2.0, help='CLAHE clipLimit')
    parser.add_argument('--tile-grid', type=int, default=8, help='CLAHE tileGridSize (정사각 크기)')
    # L-fixed
    parser.add_argument('--target-L', type=float, default=None, help='l-fixed: 목표 L 평균(0~255)')
    parser.add_argument('--target-std', type=float, default=None, help='l-fixed: 목표 L 표준편차(선택)')
    # L-histmatch
    parser.add_argument('--reference', type=str, default=None, help='l-histmatch: L 히스토그램 매칭 참조 이미지 경로')
    # L-absolute
    parser.add_argument('--abs-p-low', type=float, default=5.0, help='l-abs: 소스 L 퍼센타일 하한(기본 5)')
    parser.add_argument('--abs-p-high', type=float, default=95.0, help='l-abs: 소스 L 퍼센타일 상한(기본 95)')
    parser.add_argument('--abs-t-low', type=float, default=60.0, help='l-abs: 목표 L 하한(0~255, 기본 60)')
    parser.add_argument('--abs-t-high', type=float, default=200.0, help='l-abs: 목표 L 상한(0~255, 기본 200)')
    # misc
    parser.add_argument('--overwrite', action='store_true', help='기존 출력 파일 덮어쓰기')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
     # --- 고정 경로 설정 ---
    input_dir = Path("data")       # 입력 폴더 이름
    output_dir = Path("out_brightness")     # 출력 폴더 이름
    method = "l-abs"                         # 원하는 방식: auto / clahe / l-fixed / l-histmatch / l-abs
    reference_path = None                   # l-histmatch 쓸 때만 지정

    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        method=method,
        clip_limit=args.clip_limit,
        tile_grid_size=args.tile_grid,
        reference_path=reference_path,
        overwrite=args.overwrite,
        target_L_mean=args.target_L,
        target_L_std=args.target_std,
        abs_p_low=args.abs_p_low,
        abs_p_high=args.abs_p_high,
        abs_t_low=args.abs_t_low,
        abs_t_high=args.abs_t_high,
    )


if __name__ == '__main__':
    main()
