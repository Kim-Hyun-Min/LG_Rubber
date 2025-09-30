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

def find_rubber_bounds_advanced(img, bg_threshold=245):
    """고무 영역과 조명 정보 통합 분석"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    rubber_mask = gray < bg_threshold
    if not np.any(rubber_mask):
        return None
    
    # 각 행별 고무 픽셀 수
    row_counts = np.sum(rubber_mask, axis=1)
    rubber_rows = np.where(row_counts >= 5)[0]
    
    if len(rubber_rows) == 0:
        return None
    
    # 연속 구간 찾기
    segments = []
    current_start = rubber_rows[0]
    
    for i in range(1, len(rubber_rows)):
        if rubber_rows[i] - rubber_rows[i-1] > 2:
            segments.append((current_start, rubber_rows[i-1]))
            current_start = rubber_rows[i]
    segments.append((current_start, rubber_rows[-1]))
    
    # 가장 긴 구간이 메인 고무 영역
    main_segment = max(segments, key=lambda x: x[1] - x[0]) if segments else (rubber_rows[0], rubber_rows[-1])
    
    return {
        'top': main_segment[0],
        'bottom': main_segment[1],
        'height': main_segment[1] - main_segment[0] + 1
    }

def estimate_original_height_from_source(original_dir: Path, stem: str, bg_threshold=245):
    """원본 이미지에서 고무 영역 높이 추정"""
    # 가능한 확장자 시도
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']:
        original_path = original_dir / f"{stem}{ext}"
        if original_path.exists():
            try:
                img = imread_rgb(str(original_path))
                bounds = find_rubber_bounds_advanced(img, bg_threshold)
                if bounds:
                    return bounds['height'], str(original_path)
            except Exception as e:
                print(f"    Warning: 원본 분석 실패 {original_path.name}: {e}")
    return None, None

def find_optimal_overlap_point(top_bounds, bot_bounds, original_height=None):
    """최적의 겹침 지점 찾기"""
    if not top_bounds or not bot_bounds:
        return 50, "fallback"
    
    top_height = top_bounds['height']
    bot_height = bot_bounds['height']
    
    print(f"  Content heights - Top: {top_height}, Bot: {bot_height}")
    
    if original_height:
        base_overlap = (top_height + bot_height) - original_height
        
        # 겹침이 음수면 정보 손실 케이스
        if base_overlap < 0:
            print(f"  ⚠️  WARNING: 정보 손실 감지! {abs(base_overlap)}px 누락")
            print(f"      Top ends at ~{top_height}px, Bot starts at ~{bot_height}px from bottom")
            print(f"      Missing region estimated: {abs(base_overlap)}px")
            # 그래도 시도는 해봄 (최소 1픽셀 겹침)
            optimal_overlap = 1
            method = "info_loss_detected"
        else:
            optimal_overlap = max(1, base_overlap)
            method = "original_height_based"
    else:
        # 원본 높이가 없을 경우, 기본 20% 기본값 사용
        optimal_overlap = int((top_height + bot_height) * 0.2)
        method = "base_20_percent_fallback"

    print(f"  Final overlap: {optimal_overlap:.0f} pixels ({method})")
    return int(optimal_overlap), method

def create_profile_aware_blend(img_top, img_bot, overlap_pixels, top_bounds, bot_bounds):
    """프로파일 인식 블렌딩"""
    if overlap_pixels <= 1:
        return img_top, img_bot
    
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    # 겹침 영역 추출
    actual_overlap = min(overlap_pixels, h1, h2)
    if actual_overlap <= 1:
        return img_top, img_bot
    
    top_region = img_top[-actual_overlap:, :].astype(np.float32)
    bot_region = img_bot[:actual_overlap, :].astype(np.float32)
    
    min_w = min(top_region.shape[1], bot_region.shape[1])
    top_region = top_region[:, :min_w]
    bot_region = bot_region[:, :min_w]
    
    blended_region = np.zeros_like(top_region)
    
    # 각 행별로 적응적 블렌딩
    for i in range(actual_overlap):
        progress = i / (actual_overlap - 1) if actual_overlap > 1 else 0.5
        
        # 행별 밝기 분석
        top_row_brightness = np.mean(top_region[i])
        bot_row_brightness = np.mean(bot_region[i])
        
        # 밝기 차이에 따른 블렌딩 가중치 조정
        brightness_diff = abs(top_row_brightness - bot_row_brightness)
        
        if brightness_diff > 30:
            if top_row_brightness > bot_row_brightness:
                alpha = progress * 0.6
            else:
                alpha = 0.4 + progress * 0.6
        else:
            alpha = progress
        
        # 부드러운 전환을 위한 S-curve
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        
        blended_region[i] = (1 - alpha) * top_region[i] + alpha * bot_region[i]
    
    blended_region = np.clip(blended_region, 0, 255).astype(np.uint8)
    
    # 적용
    result_top = img_top.copy()
    result_bot = img_bot.copy()
    
    result_top[-actual_overlap:, :min_w] = blended_region
    result_bot[:actual_overlap, :min_w] = blended_region
    
    return result_top, result_bot

def resize_proportionally(img_top, img_bot):
    """비례적 리사이즈"""
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    target_w = min(w1, w2)
    
    if w1 != target_w:
        scale = target_w / w1
        new_h1 = int(h1 * scale)
        img_top = cv2.resize(img_top, (target_w, new_h1), interpolation=cv2.INTER_LANCZOS4)
        print(f"  Top resized: {w1}×{h1} → {target_w}×{new_h1}")
    
    if w2 != target_w:
        scale = target_w / w2
        new_h2 = int(h2 * scale)
        img_bot = cv2.resize(img_bot, (target_w, new_h2), interpolation=cv2.INTER_LANCZOS4)
        print(f"  Bot resized: {w2}×{h2} → {target_w}×{new_h2}")
    
    return img_top, img_bot

def stitch_with_profile_analysis(top_path, bot_path, out_path, original_height=None, 
                                 original_dir=None, bg_threshold=245):
    """프로파일 분석 기반 합성"""
    print(f"\nProcessing: {top_path.stem}")
    
    # 원본 높이 추정 시도
    estimated_height = None
    if original_dir and not original_height:
        estimated_height, original_path = estimate_original_height_from_source(
            original_dir, top_path.stem, bg_threshold
        )
        if estimated_height:
            print(f"  ✓ 원본 높이 추정: {estimated_height}px (from {Path(original_path).name})")
            original_height = estimated_height
        else:
            print(f"  ⚠️  원본 이미지 없음 - 기본 추정 사용")
    
    # 로드
    img_top = imread_rgb(str(top_path))
    img_bot = imread_rgb(str(bot_path))
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    print(f"  Input: Top {w1}×{h1}, Bot {w2}×{h2}")
    
    # 1. 고급 분석
    top_bounds = find_rubber_bounds_advanced(img_top, bg_threshold)
    bot_bounds = find_rubber_bounds_advanced(img_bot, bg_threshold)
    
    if not top_bounds or not bot_bounds:
        print("  Warning: Analysis failed")
        return None, None
    
    # 2. 리사이즈
    img_top, img_bot = resize_proportionally(img_top, img_bot)
    
    # 3. 리사이즈 후 재분석
    top_bounds = find_rubber_bounds_advanced(img_top, bg_threshold)
    bot_bounds = find_rubber_bounds_advanced(img_bot, bg_threshold)
    
    # 4. 최적 겹침 계산
    overlap, method = find_optimal_overlap_point(top_bounds, bot_bounds, original_height)
    
    # 5. 고급 블렌딩
    img_top_final, img_bot_final = create_profile_aware_blend(
        img_top, img_bot, overlap, top_bounds, bot_bounds
    )
    
    # 6. 합성
    h1_final, w1_final = img_top_final.shape[:2]
    h2_final, w2_final = img_bot_final.shape[:2]
    
    canvas_h = h1_final + h2_final - overlap
    canvas_w = w1_final
    
    # 배경
    canvas = np.full((canvas_h, canvas_w, 3), (248, 245, 240), dtype=np.uint8)
    
    # 배치
    canvas[:h1_final, :] = img_top_final
    start_y = h1_final - overlap
    canvas[start_y:start_y+h2_final, :] = img_bot_final
    
    # 결과 검증
    print(f"  Result: {canvas_w}×{canvas_h}")
    if original_height:
        error = canvas_h - original_height
        error_pct = (error / original_height) * 100
        print(f"  Accuracy: {error:+d}px ({error_pct:+.1f}%)")
    
    # 저장
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Saved: {out_path.name}")
    
    return canvas_h, overlap

def load_original_heights(height_file_path):
    """높이 정보 로드"""
    heights = {}
    if not height_file_path or not Path(height_file_path).exists():
        return heights
    
    try:
        with open(height_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    filename = Path(parts[0]).stem.lower()
                    try:
                        height = int(parts[1])
                        heights[filename] = height
                    except ValueError:
                        print(f"  Warning: Line {line_num} invalid")
    except Exception as e:
        print(f"  Height file error: {e}")
    
    return heights

def main():
    parser = argparse.ArgumentParser(description="프로파일 분석 기반 고무도장 합성 (원본 높이 자동 추정)")
    parser.add_argument("--top-dir", default="./s_data/img1", help="윗부분 폴더")
    parser.add_argument("--bot-dir", default="./s_data/img2", help="아랫부분 폴더")
    parser.add_argument("--out", default="synsetic_final", help="출력 폴더")
    parser.add_argument("--original-dir", help="원본 이미지 폴더 (높이 추정용)")
    parser.add_argument("--original-height", type=int, help="공통 원본 높이")
    parser.add_argument("--height-file", help="개별 높이 파일")
    parser.add_argument("--bg-threshold", type=int, default=245, help="배경 임계값")
    
    args = parser.parse_args()
    
    # 경로 처리
    top_dir = Path(args.top_dir)
    bot_dir = Path(args.bot_dir)
    out_dir = Path(args.out)
    original_dir = Path(args.original_dir) if args.original_dir else None
    
    if not top_dir.exists() or not bot_dir.exists():
        print("❌ Input directories not found")
        return
    
    if original_dir and not original_dir.exists():
        print(f"⚠️  Warning: 원본 폴더 없음: {original_dir}")
        original_dir = None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 수집
    top_images = {p.stem.lower(): p for p in list_images(top_dir)}
    bot_images = {p.stem.lower(): p for p in list_images(bot_dir)}
    common = sorted(set(top_images.keys()) & set(bot_images.keys()))
    
    if not common:
        print("❌ No matching pairs found")
        return
    
    print(f"Found {len(common)} matching pairs")
    if original_dir:
        print(f"원본 폴더: {original_dir}")
    
    # 높이 정보
    heights = load_original_heights(args.height_file)
    if args.original_height:
        for name in common:
            heights[name] = args.original_height
        print(f"Using common height: {args.original_height}")
    elif heights:
        print(f"Loaded {len(heights)} height entries")
    
    # 처리
    success = 0
    total_error = 0
    info_loss_count = 0
    
    for i, name in enumerate(common, 1):
        try:
            print(f"\n[{i:3}/{len(common)}]", end=" ")
            
            result_h, overlap = stitch_with_profile_analysis(
                top_images[name], 
                bot_images[name], 
                out_dir / f"{name}_restored.png",
                heights.get(name),
                original_dir,
                args.bg_threshold
            )
            
            if result_h and name in heights:
                total_error += abs(result_h - heights[name])
            
            success += 1
            
        except Exception as e:
            print(f"❌ FAILED {name}: {e}")
    
    print(f"\n{'='*50}")
    print(f"SUCCESS: {success}/{len(common)}")
    if success and heights:
        print(f"Avg error: {total_error/success:.1f}px")
    print(f"Output: {out_dir}")

if __name__ == "__main__":
    main()