import argparse, cv2, numpy as np
from pathlib import Path

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

def list_images(d: Path):
    return [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]

def imread_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"읽기 실패: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def find_rubber_content_bounds(img, bg_threshold=245):
    """고무 내용의 정확한 상하 경계 찾기"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # 각 행에서 고무 픽셀 비율
    content_ratios = []
    for y in range(h):
        row = gray[y, :]
        content_ratio = np.sum(row < bg_threshold) / w
        content_ratios.append(content_ratio)
    
    # 의미있는 내용이 있는 행 찾기 (5% 이상)
    significant_threshold = 0.05
    significant_rows = [i for i, ratio in enumerate(content_ratios) if ratio > significant_threshold]
    
    if not significant_rows:
        return None, None
    
    return significant_rows[0], significant_rows[-1]



def calculate_simple_overlap(img_top, img_bot, original_height):
    """단순한 수학적 계산으로 겹침 구하기"""
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    # 고무 내용 경계
    top_start, top_end = find_rubber_content_bounds(img_top)
    bot_start, bot_end = find_rubber_content_bounds(img_bot)
    
    print(f"  Top content: {top_start}-{top_end} (total: {h1})")
    print(f"  Bot content: {bot_start}-{bot_end} (total: {h2})")
    
    if top_end is None or bot_start is None:
        print("  Using fallback overlap calculation")
        # 60%씩 찍혔다면 20% 겹침
        overlap = int(min(h1, h2) * 0.2)
        return overlap
    
    # 실제 고무 내용 높이
    top_content_height = top_end - top_start + 1
    bot_content_height = bot_end - bot_start + 1
    
    if original_height:
        # 원본 높이 기반 계산
        expected_total_content = original_height
        current_total_content = top_content_height + bot_content_height
        overlap = current_total_content - expected_total_content
        print(f"  Calculated overlap from original height: {overlap}")
    else:
        # 60% 가정하에 20% 겹침
        overlap = int((top_content_height + bot_content_height) * 0.2)
        print(f"  Calculated overlap from 60% assumption: {overlap}")
    
    # 합리적인 범위로 제한
    min_overlap = 10
    max_overlap = min(top_content_height, bot_content_height) // 2
    overlap = max(min_overlap, min(overlap, max_overlap))
    
    return overlap

def create_alpha_blend(img_top, img_bot, overlap_pixels):
    """알파 블렌딩으로 겹치는 영역 자연스럽게 합성"""
    if overlap_pixels <= 0:
        return img_top, img_bot
    
    h1 = img_top.shape[0]
    h2 = img_bot.shape[0]
    
    # 겹치는 영역 추출
    top_overlap_region = img_top[h1-overlap_pixels:h1, :].copy()
    bot_overlap_region = img_bot[0:overlap_pixels, :].copy()
    
    # 크기 맞춤
    min_h = min(top_overlap_region.shape[0], bot_overlap_region.shape[0])
    min_w = min(top_overlap_region.shape[1], bot_overlap_region.shape[1])
    
    if min_h <= 0 or min_w <= 0:
        return img_top, img_bot
    
    top_overlap_region = top_overlap_region[:min_h, :min_w]
    bot_overlap_region = bot_overlap_region[:min_h, :min_w]
    
    # 알파 블렌딩
    blended_region = np.zeros_like(top_overlap_region, dtype=np.float32)
    
    for y in range(min_h):
        # 위에서 아래로 갈수록 아래쪽 이미지 비중 증가
        alpha = y / (min_h - 1) if min_h > 1 else 0.5
        
        blended_region[y] = (
            (1 - alpha) * top_overlap_region[y].astype(np.float32) +
            alpha * bot_overlap_region[y].astype(np.float32)
        )
    
    blended_region = np.clip(blended_region, 0, 255).astype(np.uint8)
    
    # 결과 이미지 생성
    img_top_result = img_top.copy()
    img_bot_result = img_bot.copy()
    
    # 블렌딩된 영역 적용
    img_top_result[h1-min_h:h1, :min_w] = blended_region
    img_bot_result[0:min_h, :min_w] = blended_region
    
    return img_top_result, img_bot_result

def stitch_simple_overlap(top_path, bot_path, out_path, original_height=None, align="center", dx=0):
    """단순 겹침 기반 합성"""
    print(f"\nProcessing: {top_path.stem}")
    
    img_top = imread_rgb(str(top_path))
    img_bot = imread_rgb(str(bot_path))
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    print(f"  Images - Top: {w1}x{h1}, Bot: {w2}x{h2}")
    
    # 폭 맞춤
    target_w = min(w1, w2)
    if w1 > target_w:
        scale = target_w / w1
        new_h = int(h1 * scale)
        img_top = cv2.resize(img_top, (target_w, new_h), interpolation=cv2.INTER_AREA)
        h1 = new_h
    
    if w2 > target_w:
        scale = target_w / w2  
        new_h = int(h2 * scale)
        img_bot = cv2.resize(img_bot, (target_w, new_h), interpolation=cv2.INTER_AREA)
        h2 = new_h
    
    # 겹침 계산
    overlap_pixels = calculate_simple_overlap(img_top, img_bot, original_height)
    print(f"  Using overlap: {overlap_pixels} pixels")
    
    # 블렌딩
    img_top_blended, img_bot_blended = create_alpha_blend(img_top, img_bot, overlap_pixels)
    
    # 최종 캔버스
    canvas_w = target_w
    canvas_h = h1 + h2 - overlap_pixels
    canvas = np.full((canvas_h, canvas_w, 3), (248, 245, 240), dtype=np.uint8)
    
    print(f"  Final size: {canvas_w}x{canvas_h}")
    
    # 배치
    # 위쪽 이미지
    canvas[0:h1, :] = img_top_blended
    
    # 아래쪽 이미지 (겹침만큼 위로)
    bot_start_y = h1 - overlap_pixels
    canvas[bot_start_y:bot_start_y+h2, :] = img_bot_blended
    
    # 저장
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_path.name}")
    
    return canvas_h, overlap_pixels

def load_original_heights(p: Path|None):
    d = {}
    if not p:
        return d
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [t.strip() for t in line.split(",")]
        if len(parts) >= 2:
            stem = Path(parts[0]).stem.lower()
            height = int(parts[1])
            d[stem] = height
    return d

def main():
    ap = argparse.ArgumentParser(description="단순 겹침 기반 고무 도장 합성")
    ap.add_argument("--top-dir", default="./s_data/img1", help="윗부분 폴더")
    ap.add_argument("--bot-dir", default="./s_data/img2", help="아랫부분 폴더")
    ap.add_argument("--out", default="stitched", help="출력 폴더")
    ap.add_argument("--original-height", type=int, help="원본 높이")
    ap.add_argument("--height-file", type=str, help="높이 파일: stem,height")
    
    args = ap.parse_args()

    top_dir = Path(args.top_dir)
    bot_dir = Path(args.bot_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tops = {p.stem.lower(): p for p in list_images(top_dir)}
    bots = {p.stem.lower(): p for p in list_images(bot_dir)}
    common = sorted(set(tops.keys()) & set(bots.keys()))
    
    if not common:
        print("매칭되는 파일명이 없습니다.")
        return

    # 원본 높이 정보
    original_heights = {}
    if args.height_file:
        original_heights = load_original_heights(Path(args.height_file))
    elif args.original_height:
        original_heights = {stem: args.original_height for stem in common}

    print(f"Processing {len(common)} pairs...")

    ok = 0
    for stem in common:
        try:
            top_p = tops[stem]
            bot_p = bots[stem]
            original_h = original_heights.get(stem, None)
            
            out_path = out_dir / f"{stem}_simple.png"
            
            final_h, overlap = stitch_simple_overlap(
                top_p, bot_p, out_path, 
                original_height=original_h
            )
            
            ok += 1
            
        except Exception as e:
            print(f"FAILED {stem}: {e}")

    print(f"\nCompleted: {ok}/{len(common)} successful")

if __name__ == "__main__":
    main()