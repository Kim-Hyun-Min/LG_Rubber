import argparse, cv2, numpy as np
from pathlib import Path

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

# --- 병렬 처리용 헬퍼 (파일 상단에 추가) ---
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def _process_one(args):
    """
    단일 페어 처리 (프로세스 풀에서 실행될 함수)
    args: (name, top_path, bot_path, out_path, height_for_name, original_dir, bg_threshold)
    """
    (name, top_path, bot_path, out_path, height_for_name, original_dir, bg_threshold) = args
    # 프로세스 내 OpenCV 스레드 오버섭션 방지 (선택)
    try:
        import cv2
        cv2.setNumThreads(0)  # 또는 1
    except Exception:
        pass

    try:
        result_h, overlap = stitch_with_profile_analysis(
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
    """고무 영역과 조명 정보 통합 분석"""
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

def compute_affine_mse(top_region, bot_region):
    """조명 보정된 MSE: bot ≈ a·top + b 후 잔차"""
    top_flat = top_region.flatten().astype(np.float32)
    bot_flat = bot_region.flatten().astype(np.float32)
    
    # 선형 회귀: bot = a * top + b
    A = np.vstack([top_flat, np.ones(len(top_flat))]).T
    try:
        a, b = np.linalg.lstsq(A, bot_flat, rcond=None)[0]
        predicted = a * top_flat + b
        residual = bot_flat - predicted
        return np.mean(residual ** 2), a, b
    except:
        return np.mean((top_flat - bot_flat) ** 2), 1.0, 0.0

def compute_zscore_mse(top_region, bot_region):
    """표준화(Z-score) 후 MSE - 조명 불변"""
    def zscore(arr):
        arr_flat = arr.flatten().astype(np.float32)
        mean = np.mean(arr_flat)
        std = np.std(arr_flat)
        if std < 1e-6:
            return arr_flat
        return (arr_flat - mean) / std
    
    top_z = zscore(top_region)
    bot_z = zscore(bot_region)
    return np.mean((top_z - bot_z) ** 2)

def compute_gradient_mse(top_region, bot_region):
    """세로 그라디언트 MSE - 경계/형상 민감"""
    top_gray = cv2.cvtColor(top_region, cv2.COLOR_RGB2GRAY).astype(np.float32) if len(top_region.shape) == 3 else top_region.astype(np.float32)
    bot_gray = cv2.cvtColor(bot_region, cv2.COLOR_RGB2GRAY).astype(np.float32) if len(bot_region.shape) == 3 else bot_region.astype(np.float32)
    
    # Sobel 세로 그라디언트
    top_grad = cv2.Sobel(top_gray, cv2.CV_32F, 0, 1, ksize=3)
    bot_grad = cv2.Sobel(bot_gray, cv2.CV_32F, 0, 1, ksize=3)
    
    return np.mean((top_grad - bot_grad) ** 2)

def compute_stripe_scores(gray_top, gray_bot, overlap, num_stripes=7):
    """가로 스트라이프별 점수 계산 + 이상치 억제"""
    h1, w = gray_top.shape
    h2 = gray_bot.shape[0]
    
    if overlap > h1 or overlap > h2 or overlap < 1:
        return None, None
    
    top_region = gray_top[-overlap:, :]
    bot_region = gray_bot[:overlap, :]
    
    stripe_width = w // num_stripes
    scores = []
    
    for i in range(num_stripes):
        start_x = i * stripe_width
        end_x = (i + 1) * stripe_width if i < num_stripes - 1 else w
        
        top_stripe = top_region[:, start_x:end_x]
        bot_stripe = bot_region[:, start_x:end_x]
        
        # Z-score MSE (조명 불변)
        zscore_mse = compute_zscore_mse(top_stripe, bot_stripe)
        scores.append(zscore_mse)
    
    # 중앙값 (이상치 억제)
    median_score = np.median(scores)
    variance = np.var(scores)
    
    return median_score, variance

def find_overlap_two_stage(img_top, img_bot, initial_guess=None):
    """2단계 탐색: Coarse → Refine"""
    
    gray_top = cv2.cvtColor(img_top, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_bot = cv2.cvtColor(img_bot, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    h1, h2 = gray_top.shape[0], gray_bot.shape[0]
    w = min(gray_top.shape[1], gray_bot.shape[1])
    
    # === 1단계: 전역 거친 탐색 (5~40%) ===
    coarse_start = max(5, int(min(h1, h2) * 0.05))
    coarse_end = min(h1, h2, int(min(h1, h2) * 0.40))
    coarse_step = max(3, (coarse_end - coarse_start) // 30)  # 약 30회 샘플링
    
    print(f"  [Stage 1] Coarse search: {coarse_start}~{coarse_end}px (step={coarse_step})")
    
    best_coarse = coarse_start
    best_coarse_score = float('inf')
    coarse_scores = []
    
    for overlap in range(coarse_start, coarse_end + 1, coarse_step):
        # 혼합 점수 계산
        top_region = gray_top[-overlap:, :w]
        bot_region = gray_bot[:overlap, :w]
        
        # 1) Affine MSE
        aff_mse, a, b = compute_affine_mse(top_region, bot_region)
        
        # 2) Z-score MSE
        z_mse = compute_zscore_mse(top_region, bot_region)
        
        # 3) Gradient MSE
        grad_mse = compute_gradient_mse(top_region, bot_region)
        
        # 4) 스트라이프 점수
        stripe_score, stripe_var = compute_stripe_scores(gray_top, gray_bot, overlap)
        
        # 혼합 점수 (가중합)
        combined = 0.3 * aff_mse + 0.4 * z_mse + 0.2 * grad_mse + 0.1 * stripe_score
        
        coarse_scores.append((overlap, combined, stripe_var))
        
        if combined < best_coarse_score:
            best_coarse_score = combined
            best_coarse = overlap
    
    print(f"  [Stage 1] Best: {best_coarse}px (score={best_coarse_score:.2f})")
    
    # === 2단계: 정밀 탐색 (±2% 범위, 1px 스텝) ===
    margin_px = max(5, int(h1 * 0.02))
    refine_start = max(5, best_coarse - margin_px)
    refine_end = min(h1, h2, best_coarse + margin_px)
    
    print(f"  [Stage 2] Refine search: {refine_start}~{refine_end}px (±{margin_px}px)")
    
    best_overlap = best_coarse
    best_score = best_coarse_score
    best_metrics = {}
    
    for overlap in range(refine_start, refine_end + 1):
        top_region = gray_top[-overlap:, :w]
        bot_region = gray_bot[:overlap, :w]
        
        aff_mse, a, b = compute_affine_mse(top_region, bot_region)
        z_mse = compute_zscore_mse(top_region, bot_region)
        grad_mse = compute_gradient_mse(top_region, bot_region)
        stripe_score, stripe_var = compute_stripe_scores(gray_top, gray_bot, overlap)
        
        combined = 0.3 * aff_mse + 0.4 * z_mse + 0.2 * grad_mse + 0.1 * stripe_score
        
        if combined < best_score:
            best_score = combined
            best_overlap = overlap
            best_metrics = {
                'aff_mse': aff_mse,
                'z_mse': z_mse,
                'grad_mse': grad_mse,
                'stripe_score': stripe_score,
                'stripe_var': stripe_var,
                'affine_a': a,
                'affine_b': b
            }
    
    # 상관계수 계산 (Z-score 프로파일)
    top_profile = np.mean(gray_top[-best_overlap:, :w], axis=1)
    bot_profile = np.mean(gray_bot[:best_overlap, :w], axis=1)
    
    # Z-score 정규화
    top_profile = (top_profile - np.mean(top_profile)) / (np.std(top_profile) + 1e-6)
    bot_profile = (bot_profile - np.mean(bot_profile)) / (np.std(bot_profile) + 1e-6)
    
    corr_z = np.corrcoef(top_profile, bot_profile)[0, 1] if len(top_profile) > 1 else 0.0
    
    print(f"  [Stage 2] Best: {best_overlap}px (combined={best_score:.2f})")
    print(f"            Affine MSE={best_metrics.get('aff_mse', 0):.1f}, Z-MSE={best_metrics.get('z_mse', 0):.2f}")
    print(f"            Grad MSE={best_metrics.get('grad_mse', 0):.1f}, Corr(Z)={corr_z:.3f}")
    print(f"            Stripe var={best_metrics.get('stripe_var', 0):.2f}")
    
    return best_overlap, best_score, corr_z, best_metrics

def find_optimal_overlap_point(top_bounds, bot_bounds, img_top, img_bot, original_height=None):
    """하이브리드: 기하 추정 + 2단계 탐색 + 신뢰도 기반 블렌딩"""
    if not top_bounds or not bot_bounds:
        return 50, "fallback"
    
    top_height = top_bounds['height']
    bot_height = bot_bounds['height']
    
    print(f"  Content heights - Top: {top_height}, Bot: {bot_height}")
    
    # === 기하학적 초기 추정 ===
    geom_overlap = None
    geom_confidence = 0.0
    
    if original_height:
        base_overlap = (top_height + bot_height) - original_height
        
        if base_overlap < 0:
            print(f"  ⚠️ WARNING: 정보 손실 감지! {abs(base_overlap)}px 누락")
            geom_overlap = 1
            geom_confidence = 0.2  # 낮은 신뢰도
        else:
            geom_overlap = max(1, base_overlap)
            geom_confidence = 0.7  # 보통 신뢰도
        
        print(f"  Geometric estimate: {geom_overlap}px (confidence={geom_confidence:.1f})")
    
    # === 2단계 탐색 (항상 실행) ===
    matched_overlap, combined_score, corr_z, metrics = find_overlap_two_stage(img_top, img_bot, geom_overlap)
    
    # === 다중 지표 신뢰도 평가 ===
    match_confidence = 0.0
    
    aff_mse = metrics.get('aff_mse', 999)
    stripe_var = metrics.get('stripe_var', 999)
    
    # 조건 1: Affine MSE
    cond1 = aff_mse < 300
    # 조건 2: Z-score 상관계수
    cond2 = corr_z > 0.6
    # 조건 3: 스트라이프 분산
    cond3 = stripe_var < 50
    
    if cond1 and cond2 and cond3:
        match_confidence = 0.95  # 높은 신뢰도
        print(f"  ✓ High confidence match (all criteria passed)")
    elif cond1 and cond2:
        match_confidence = 0.75
        print(f"  ✓ Medium confidence match (2/3 criteria)")
    elif cond1 or cond2:
        match_confidence = 0.5
        print(f"  ⚠ Low confidence match (1/3 criteria)")
    else:
        match_confidence = 0.2
        print(f"  ✗ Very low confidence match")
    
    # === 하이브리드 결합 ===
    if geom_overlap is not None:
        # 기하값 검증: 매칭값과 크게 다르면 신뢰도 하향
        diff_ratio = abs(matched_overlap - geom_overlap) / max(geom_overlap, 1)
        if diff_ratio > 0.15:  # 15% 이상 차이
            print(f"  ⚠ Large discrepancy: {diff_ratio*100:.1f}% → lowering geom confidence")
            geom_confidence *= 0.5
        
        # 신뢰도 기반 가중 평균
        total_conf = geom_confidence + match_confidence
        alpha = match_confidence / total_conf if total_conf > 0 else 0.5
        
        final_overlap = int(alpha * matched_overlap + (1 - alpha) * geom_overlap)
        method = f"hybrid(α={alpha:.2f},geom={geom_overlap},match={matched_overlap})"
    else:
        final_overlap = matched_overlap
        method = "match_only"
    
    print(f"  Final overlap: {final_overlap}px ({method})")
    return int(final_overlap), method

def create_profile_aware_blend(img_top, img_bot, overlap_pixels, top_bounds, bot_bounds):
    """개선된 프로파일 인식 블렌딩 - 겹침 영역만 정확히 블렌딩"""
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
    
    # === 방법 1: 픽셀별 유사도 기반 적응적 블렌딩 ===
    # 유사한 픽셀은 블렌딩, 다른 픽셀은 sharp transition
    
    # 픽셀별 차이 계산
    pixel_diff = np.abs(top_region - bot_region)
    similarity = 255 - np.mean(pixel_diff, axis=2, keepdims=True)  # (H, W, 1)
    similarity = similarity / 255.0  # 0~1 정규화
    
    # 행별 유사도 프로파일
    row_similarity = np.mean(similarity, axis=1)  # (H, 1)
    
    blended_region = np.zeros_like(top_region)
    
    for i in range(actual_overlap):
        # 기본 선형 진행도
        progress = i / (actual_overlap - 1) if actual_overlap > 1 else 0.5
        
        # 행 유사도에 따른 블렌딩 강도 조절
        sim = row_similarity[i, 0]
        
        if sim > 0.8:  # 매우 유사 → 부드러운 블렌딩
            # S-curve로 부드러운 전환
            alpha = progress ** 2 * (3.0 - 2.0 * progress)
        elif sim > 0.5:  # 중간 유사도 → 적당한 블렌딩
            # 약한 S-curve
            alpha = progress ** 1.5
        else:  # 다름 → 거의 블렌딩 안 함 (sharp transition)
            # 중간 지점에서만 짧게 블렌딩
            if 0.4 < progress < 0.6:
                alpha = (progress - 0.4) * 5  # 0.4~0.6 구간만 0→1
                alpha = np.clip(alpha, 0, 1)
            elif progress < 0.5:
                alpha = 0.0  # 상단 이미지 그대로
            else:
                alpha = 1.0  # 하단 이미지 그대로
        
        # 블렌딩 적용
        blended_region[i] = (1 - alpha) * top_region[i] + alpha * bot_region[i]
    
    # === 방법 2: 추가 - 엣지 보존 ===
    # 겹침 영역 내 고무 경계선 보존
    if top_bounds and bot_bounds:
        # 상단 이미지의 하단 경계 확인
        top_edge = h1 - top_bounds['bottom']
        # 하단 이미지의 상단 경계 확인
        bot_edge = bot_bounds['top']
        
        # 경계선 근처는 블렌딩 최소화
        for i in range(actual_overlap):
            top_dist = abs((h1 - actual_overlap + i) - top_bounds['bottom'])
            bot_dist = abs(i - bot_edge)
            
            # 경계선 5px 이내는 원본 우선
            if top_dist < 5 or bot_dist < 5:
                progress = i / (actual_overlap - 1) if actual_overlap > 1 else 0.5
                if progress < 0.3:
                    blended_region[i] = top_region[i]  # 상단 원본
                elif progress > 0.7:
                    blended_region[i] = bot_region[i]  # 하단 원본
    
    blended_region = np.clip(blended_region, 0, 255).astype(np.uint8)
    
    # 결과 적용
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
    """개선된 프로파일 분석 기반 합성"""
    print(f"\nProcessing: {top_path.stem}")
    
    estimated_height = None
    if original_dir and not original_height:
        estimated_height, original_path = estimate_original_height_from_source(
            original_dir, top_path.stem, bg_threshold
        )
        if estimated_height:
            print(f"  ✓ 원본 높이 추정: {estimated_height}px (from {Path(original_path).name})")
            original_height = estimated_height
        else:
            print(f"  ⚠️ 원본 이미지 없음 - 2단계 탐색만 사용")
    
    img_top = imread_rgb(str(top_path))
    img_bot = imread_rgb(str(bot_path))
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bot.shape[:2]
    
    print(f"  Input: Top {w1}×{h1}, Bot {w2}×{h2}")
    
    top_bounds = find_rubber_bounds_advanced(img_top, bg_threshold)
    bot_bounds = find_rubber_bounds_advanced(img_bot, bg_threshold)
    
    if not top_bounds or not bot_bounds:
        print("  Warning: Analysis failed")
        return None, None
    
    img_top, img_bot = resize_proportionally(img_top, img_bot)
    
    top_bounds = find_rubber_bounds_advanced(img_top, bg_threshold)
    bot_bounds = find_rubber_bounds_advanced(img_bot, bg_threshold)
    
    overlap, method = find_optimal_overlap_point(top_bounds, bot_bounds, img_top, img_bot, original_height)
    
    img_top_final, img_bot_final = create_profile_aware_blend(
        img_top, img_bot, overlap, top_bounds, bot_bounds
    )
    
    h1_final, w1_final = img_top_final.shape[:2]
    h2_final, w2_final = img_bot_final.shape[:2]
    
    canvas_h = h1_final + h2_final - overlap
    canvas_w = w1_final
    
    canvas = np.full((canvas_h, canvas_w, 3), (248, 245, 240), dtype=np.uint8)
    
    canvas[:h1_final, :] = img_top_final
    start_y = h1_final - overlap
    canvas[start_y:start_y+h2_final, :] = img_bot_final
    
    print(f"  Result: {canvas_w}×{canvas_h}")
    if original_height:
        error = canvas_h - original_height
        error_pct = (error / original_height) * 100
        print(f"  Accuracy: {error:+d}px ({error_pct:+.1f}%)")
    
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
    parser = argparse.ArgumentParser(description="개선된 2단계 탐색 + 다중 지표 합성")
    parser.add_argument("--top-dir", default="./s_data/img1", help="윗부분 폴더")
    parser.add_argument("--bot-dir", default="./s_data/img2", help="아랫부분 폴더")
    parser.add_argument("--out", default="compose_final", help="출력 폴더")
    parser.add_argument("--original-dir", help="원본 이미지 폴더 (높이 추정용)")
    parser.add_argument("--original-height", type=int, help="공통 원본 높이")
    parser.add_argument("--height-file", help="개별 높이 파일")
    parser.add_argument("--bg-threshold", type=int, default=245, help="배경 임계값")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="병렬 처리 프로세스 수 (기본: CPU코어-1)")
    
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
    
    top_images = {p.stem.lower(): p for p in list_images(top_dir)}
    bot_images = {p.stem.lower(): p for p in list_images(bot_dir)}
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
    print("개선 전략: 2단계 탐색 + 조명 불변 점수 + 다중 지표")
    print("="*60)
    
    success = 0
    total_error = 0
    
    # 작업 목록 만들기
    tasks = []
    for name in common:
        tasks.append((
            name,
            top_images[name],
            bot_images[name],
            out_dir / f"{name}_restored.png",
            heights.get(name),
            original_dir,
            args.bg_threshold
        ))

    workers = max(1, getattr(args, "workers", 1))
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