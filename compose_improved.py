import argparse, cv2, numpy as np
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

def _process_one(args):
    """단일 페어 처리 (프로세스 풀에서 실행될 함수)"""
    (name, top_path, bot_path, out_path, height_for_name, original_dir, bg_threshold) = args
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        result_h, overlap = stitch_with_improved_blending(
            top_path, bot_path, out_path,
            height_for_name, original_dir, bg_threshold
        )
        return {
            "name": name,
            "ok": True,
            "result_h": result_h,
            "overlap": overlap,
            "height": height_for_name,
            "error": None
        }
    except Exception as e:
        return {
            "name": name,
            "ok": False,
            "result_h": None,
            "overlap": None,
            "height": height_for_name,
            "error": str(e)
        }

def list_images(d: Path):
    return [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]

def imread_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"읽기 실패: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def find_rubber_bounds_advanced(img, bg_threshold=245):
    """고무 영역 검출"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    rubber_mask = gray < bg_threshold
    if not np.any(rubber_mask):
        return None
    
    row_counts = np.sum(rubber_mask, axis=1)
    rubber_rows = np.where(row_counts >= 5)[0]
    
    if len(rubber_rows) == 0:
        return None
    
    segments = []
    current_start = rubber_rows[0]
    
    for i in range(1, len(rubber_rows)):
        if rubber_rows[i] - rubber_rows[i-1] > 2:
            segments.append((current_start, rubber_rows[i-1]))
            current_start = rubber_rows[i]
    segments.append((current_start, rubber_rows[-1]))
    
    main_segment = max(segments, key=lambda x: x[1] - x[0]) if segments else (rubber_rows[0], rubber_rows[-1])
    
    return {
        'top': main_segment[0],
        'bottom': main_segment[1],
        'height': main_segment[1] - main_segment[0] + 1
    }

def estimate_original_height_from_source(original_dir: Path, stem: str, bg_threshold=245):
    """원본 이미지에서 고무 영역 높이 추정"""
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

def compute_normalized_correlation(img1, img2):
    """정규화된 상관계수 계산 - 더 안정적"""
    # 그레이스케일 변환
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2
    
    # 평균 프로파일 계산
    profile1 = np.mean(gray1, axis=1).astype(np.float32)
    profile2 = np.mean(gray2, axis=1).astype(np.float32)
    
    # 정규화
    profile1 = (profile1 - np.mean(profile1)) / (np.std(profile1) + 1e-6)
    profile2 = (profile2 - np.mean(profile2)) / (np.std(profile2) + 1e-6)
    
    # 상관계수
    if len(profile1) > 1 and len(profile2) > 1:
        correlation = np.corrcoef(profile1, profile2)[0, 1]
        return max(0, correlation)  # 음수 상관계수는 0으로
    return 0.0

def find_best_overlap_robust(img_top, img_bot, original_height=None):
    """개선된 겹침 찾기 - 더 안정적이고 간단한 방법"""
    h1, h2 = img_top.shape[:2]
    w = min(img_top.shape[1], img_bot.shape[1])
    
    # 탐색 범위 설정
    min_overlap = max(5, int(min(h1, h2) * 0.05))  # 최소 5% 
    max_overlap = min(h1, h2, int(min(h1, h2) * 0.40))  # 최대 40%
    
    print(f"  Overlap search range: {min_overlap}~{max_overlap}px")
    
    best_overlap = min_overlap
    best_score = -1
    
    # 기하학적 초기 추정
    geometric_hint = None
    if original_height:
        top_bounds = find_rubber_bounds_advanced(img_top)
        bot_bounds = find_rubber_bounds_advanced(img_bot)
        if top_bounds and bot_bounds:
            estimated_overlap = (top_bounds['height'] + bot_bounds['height']) - original_height
            if estimated_overlap > 0:
                geometric_hint = max(min_overlap, min(max_overlap, estimated_overlap))
                print(f"  Geometric hint: {geometric_hint}px")
    
    # 1차: 거친 탐색 (step=3)
    coarse_step = max(3, (max_overlap - min_overlap) // 20)
    coarse_candidates = []
    
    for overlap in range(min_overlap, max_overlap + 1, coarse_step):
        # 겹침 영역 추출
        top_region = img_top[-overlap:, :w]
        bot_region = img_bot[:overlap, :w]
        
        # 상관계수 계산
        correlation = compute_normalized_correlation(top_region, bot_region)
        
        # 추가 지표: 밝기 차이
        brightness_diff = abs(np.mean(top_region) - np.mean(bot_region))
        brightness_score = max(0, 1 - brightness_diff / 100)  # 0~1
        
        # 종합 점수
        combined_score = 0.7 * correlation + 0.3 * brightness_score
        
        coarse_candidates.append((overlap, combined_score))
        
        if combined_score > best_score:
            best_score = combined_score
            best_overlap = overlap
    
    print(f"  Coarse search best: {best_overlap}px (score={best_score:.3f})")
    
    # 2차: 정밀 탐색 (best ±10px 범위)
    fine_start = max(min_overlap, best_overlap - 10)
    fine_end = min(max_overlap, best_overlap + 10)
    
    for overlap in range(fine_start, fine_end + 1):
        top_region = img_top[-overlap:, :w]
        bot_region = img_bot[:overlap, :w]
        
        correlation = compute_normalized_correlation(top_region, bot_region)
        brightness_diff = abs(np.mean(top_region) - np.mean(bot_region))
        brightness_score = max(0, 1 - brightness_diff / 100)
        
        combined_score = 0.7 * correlation + 0.3 * brightness_score
        
        if combined_score > best_score:
            best_score = combined_score
            best_overlap = overlap
    
    # 기하학적 힌트가 있고 점수가 비슷하면 힌트 우선
    if geometric_hint and abs(best_overlap - geometric_hint) <= 5:
        print(f"  Using geometric hint: {geometric_hint}px (close to best: {best_overlap}px)")
        best_overlap = geometric_hint
    
    print(f"  Final overlap: {best_overlap}px (score={best_score:.3f})")
    return best_overlap

def create_smooth_blend(img_top, img_bot, overlap_pixels):
    """개선된 부드러운 블렌딩 - 단순하고 효과적"""
    if overlap_pixels <= 1:
        return img_top, img_bot
    
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    actual_overlap = min(overlap_pixels, h1, h2)
    if actual_overlap <= 1:
        return img_top, img_bot
    
    # 겹침 영역 추출
    top_region = img_top[-actual_overlap:, :].astype(np.float32)
    bot_region = img_bot[:actual_overlap, :].astype(np.float32)
    
    min_w = min(top_region.shape[1], bot_region.shape[1])
    top_region = top_region[:, :min_w]
    bot_region = bot_region[:, :min_w]
    
    # === 개선된 블렌딩 전략 ===
    blended_region = np.zeros_like(top_region)
    
    for i in range(actual_overlap):
        # 선형 진행도 (0 → 1)
        progress = i / (actual_overlap - 1) if actual_overlap > 1 else 0.5
        
        # === 방법 1: S-curve 블렌딩 (기본) ===
        # 부드러운 S자 곡선으로 자연스러운 전환
        alpha = progress ** 2 * (3.0 - 2.0 * progress)
        
        # === 방법 2: 픽셀별 적응 (고품질) ===
        # 각 픽셀별로 미세 조정
        pixel_diff = np.abs(top_region[i] - bot_region[i])
        similarity = np.exp(-pixel_diff / 50.0)  # 0~1, 유사할수록 1
        
        # 유사한 픽셀은 더 부드럽게, 다른 픽셀은 더 빠르게 전환
        adaptive_alpha = alpha * similarity + (1 - similarity) * (progress ** 0.5)
        adaptive_alpha = np.clip(adaptive_alpha, 0, 1)
        
        # 최종 블렌딩
        blended_region[i] = (1 - adaptive_alpha) * top_region[i] + adaptive_alpha * bot_region[i]
    
    blended_region = np.clip(blended_region, 0, 255).astype(np.uint8)
    
    # 결과 적용
    result_top = img_top.copy()
    result_bot = img_bot.copy()
    
    result_top[-actual_overlap:, :min_w] = blended_region
    result_bot[:actual_overlap, :min_w] = blended_region
    
    return result_top, result_bot

def resize_with_aspect_preservation(img_top, img_bot):
    """종횡비 보존하면서 리사이즈 - 더 정확"""
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    # 더 작은 너비에 맞춤
    target_w = min(w1, w2)
    
    result_top = img_top
    result_bot = img_bot
    
    # 상단 이미지 리사이즈
    if w1 != target_w:
        scale = target_w / w1
        new_h1 = int(round(h1 * scale))  # round() 추가로 더 정확
        result_top = cv2.resize(img_top, (target_w, new_h1), interpolation=cv2.INTER_LANCZOS4)
        print(f"  Top resized: {w1}×{h1} → {target_w}×{new_h1} (scale={scale:.4f})")
    
    # 하단 이미지 리사이즈  
    if w2 != target_w:
        scale = target_w / w2
        new_h2 = int(round(h2 * scale))  # round() 추가로 더 정확
        result_bot = cv2.resize(img_bot, (target_w, new_h2), interpolation=cv2.INTER_LANCZOS4)
        print(f"  Bot resized: {w2}×{h2} → {target_w}×{new_h2} (scale={scale:.4f})")
    
    return result_top, result_bot

def stitch_with_improved_blending(top_path, bot_path, out_path, original_height=None, 
                                 original_dir=None, bg_threshold=245):
    """개선된 블렌딩으로 합성"""
    print(f"\nProcessing: {top_path.stem}")
    
    # 원본 높이 추정
    estimated_height = None
    if original_dir and not original_height:
        estimated_height, original_path = estimate_original_height_from_source(
            original_dir, top_path.stem, bg_threshold
        )
        if estimated_height:
            print(f"  ✓ 원본 높이 추정: {estimated_height}px (from {Path(original_path).name})")
            original_height = estimated_height
        else:
            print(f"  ⚠️ 원본 이미지 없음")
    
    # 이미지 로드
    img_top = imread_rgb(str(top_path))
    img_bot = imread_rgb(str(bot_path))
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    print(f"  Input: Top {w1}×{h1}, Bot {w2}×{h2}")
    
    # 크기 조정 (종횡비 보존)
    img_top, img_bot = resize_with_aspect_preservation(img_top, img_bot)
    
    # 최적 겹침 찾기
    overlap = find_best_overlap_robust(img_top, img_bot, original_height)
    
    # 부드러운 블렌딩 적용
    img_top_final, img_bot_final = create_smooth_blend(img_top, img_bot, overlap)
    
    # 최종 합성
    h1_final, w1_final = img_top_final.shape[:2]
    h2_final, w2_final = img_bot_final.shape[:2]
    
    canvas_h = h1_final + h2_final - overlap
    canvas_w = w1_final
    
    # 캔버스 생성 (더 자연스러운 배경색)
    canvas = np.full((canvas_h, canvas_w, 3), (248, 245, 240), dtype=np.uint8)
    
    # 이미지 배치
    canvas[:h1_final, :] = img_top_final
    start_y = h1_final - overlap
    canvas[start_y:start_y+h2_final, :] = img_bot_final
    
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
    parser = argparse.ArgumentParser(description="개선된 블렌딩 합성")
    parser.add_argument("--top-dir", default="./s_data/img1", help="윗부분 폴더")
    parser.add_argument("--bot-dir", default="./s_data/img2", help="아랫부분 폴더")
    parser.add_argument("--out", default="compose_copied", help="출력 폴더")
    parser.add_argument("--original-dir", help="원본 이미지 폴더 (높이 추정용)")
    parser.add_argument("--original-height", type=int, help="공통 원본 높이")
    parser.add_argument("--height-file", help="개별 높이 파일")
    parser.add_argument("--bg-threshold", type=int, default=245, help="배경 임계값")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="병렬 처리 프로세스 수")
    
    args = parser.parse_args()
    
    top_dir = Path(args.top_dir)
    bot_dir = Path(args.bot_dir)
    out_dir = Path(args.out)
    original_dir = Path(args.original_dir) if args.original_dir else None
    
    if not top_dir.exists() or not bot_dir.exists():
        print("❌ Input directories not found")
        return
    
    if original_dir and not original_dir.exists():
        print(f"⚠️ Warning: 원본 폴더 없음: {original_dir}")
        original_dir = None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # split_top_, split_bot_ 접두사를 제거한 후 매칭
    def get_base_name(path):
        stem = path.stem.lower()
        if stem.startswith('split_top_'):
            return stem[10:]  # 'split_top_' 제거
        elif stem.startswith('split_bot_'):
            return stem[10:]  # 'split_bot_' 제거
        else:
            return stem
    
    top_images = {get_base_name(p): p for p in list_images(top_dir)}
    bot_images = {get_base_name(p): p for p in list_images(bot_dir)}
    common = sorted(set(top_images.keys()) & set(bot_images.keys()))
    
    if not common:
        print("❌ No matching pairs found")
        return
    
    print(f"Found {len(common)} matching pairs")
    if original_dir:
        print(f"원본 폴더: {original_dir}")
    
    heights = load_original_heights(args.height_file)
    if args.original_height:
        for name in common:
            heights[name] = args.original_height
        print(f"Using common height: {args.original_height}")
    elif heights:
        print(f"Loaded {len(heights)} height entries")
    
    print("\n" + "="*60)
    print("개선 전략: 단순하고 안정적인 블렌딩")
    print("="*60)
    
    # 작업 목록 만들기
    tasks = []
    for name in common:
        tasks.append((
            name,
            top_images[name],
            bot_images[name],
            out_dir / f"{name}_alpha_restored.png",
            heights.get(name),
            original_dir,
            args.bg_threshold
        ))

    workers = max(1, args.workers)
    success = 0
    total_error = 0

    print(f"\n[Parallel] start: {len(tasks)} pairs | workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_process_one, t): t[0] for t in tasks}
        done_cnt = 0
        for fut in as_completed(future_map):
            name = future_map[fut]
            done_cnt += 1
            res = fut.result()

            if res["ok"]:
                success += 1
                if res["result_h"] and res["height"]:
                    err = abs(res["result_h"] - res["height"])
                    total_error += err
                    print(f"[{done_cnt:3}/{len(tasks)}] ✓ {name} (overlap={res['overlap']}, err={err}px)")
                else:
                    print(f"[{done_cnt:3}/{len(tasks)}] ✓ {name} (overlap={res['overlap']})")
            else:
                print(f"[{done_cnt:3}/{len(tasks)}] ❌ FAILED {name}: {res['error']}")

    print(f"\n{'='*50}")
    print(f"SUCCESS: {success}/{len(tasks)}")
    if success and heights:
        print(f"Avg error: {total_error/success:.1f}px")
    print(f"Output: {out_dir}")

if __name__ == "__main__":
    main()