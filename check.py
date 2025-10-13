import cv2
import sys
import numpy as np
from pathlib import Path
import os
import re

# ====== 추가: 키 생성 유틸 (앞의 숫자 블록 동일 여부) ======
# 예) 2025_09_17_10_56_57.218_27_alpha_alpha_restored.png
#  -> 키: 2025_09_17_10_56_57.218_27
_TS_HEAD = re.compile(
    r'^(?P<ts>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}'
    r'(?:\.\d+)?'   # .218 또는 .683 (밀리초 - 1자리 이상)
    r'(?:_\d+)?'    # _27 또는 _1 (인덱스)
    r')'
)

def _strip_prefix(s: str) -> str:
    s = s.lower()
    s = re.sub(r'^split_(top|bot)[-_]+', '', s)  # split_top_/split_bot_ 제거
    s = re.sub(r'^(top|bot)[-_]+', '', s)        # top_/bot_/top-/bot- 제거
    return s

def make_key(stem: str) -> str | None:
    """
    1) 접두사 제거(split_top_/split_bot_/top_/bot_)
    2) 선두의 숫자 블록 'YYYY_MM_DD_HH_MM_SS(.mmm)?(_idx)?'을 키로 사용
    3) (2)가 안 잡히면 과거 로직 호환: '_alpha' 앞부분 + '_alpha'
    4) 그래도 없으면 None
    """
    s = _strip_prefix(stem)
    m = _TS_HEAD.match(s)
    if m:
        return m.group('ts')
    if '_alpha' in s:
        return s.split('_alpha')[0] + '_alpha'
    return None
# ========================================================

def compare_folders(folder1, folder2):
    """두 폴더의 같은 '키(앞 숫자 블록)'를 갖는 이미지들을 비교"""
    p1 = Path(folder1)
    p2 = Path(folder2)
    if not p1.exists() or not p2.exists():
        print(f"[X] 폴더가 존재하지 않습니다: {folder1} 또는 {folder2}")
        return

    # png만이 아니라 자주 쓰는 확장자 모두 허용
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files1 = [f for f in p1.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
    files2 = [f for f in p2.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]

    # 파일명에서 '키' 추출하여 매핑
    files1_dict = {}
    files2_dict = {}

    for f in files1:
        k = make_key(f.stem)
        if k:
            files1_dict[k] = f

    for f in files2:
        k = make_key(f.stem)
        if k:
            files2_dict[k] = f

    # 공통 키 찾기
    common_files = sorted(set(files1_dict.keys()) & set(files2_dict.keys()))
    if not common_files:
        print("[X] 공통된 키(앞 숫자 부분)가 없습니다.")
        # 디버깅 힌트 조금 출력
        print("  예시(top):", list(files1_dict.keys())[:5])
        print("  예시(bot):", list(files2_dict.keys())[:5])
        return

    # 참고: 고아(한쪽에만 있는 키)도 알려줌
    only1 = sorted(set(files1_dict.keys()) - set(files2_dict.keys()))
    only2 = sorted(set(files2_dict.keys()) - set(files1_dict.keys()))
    if only1:
        print(f"[!] 폴더1 전용 키 {len(only1)}개 (예시 5개):", only1[:5])
    if only2:
        print(f"[!] 폴더2 전용 키 {len(only2)}개 (예시 5개):", only2[:5])

    print(f"[INFO] 총 {len(common_files)}개의 공통 키를 찾았습니다.")
    print("[INFO] 사용법:")
    print("  - 'd' 키: 다음 이미지 쌍")
    print("  - 'a' 키: 이전 이미지 쌍")
    print("  - 'q' 키: 종료")
    print("  - 다른 키: 다음 이미지로 이동")
    print("-" * 50)

    current_idx = 0

    try:
        while True:
            current_key = common_files[current_idx]
            file1_path = files1_dict[current_key]
            file2_path = files2_dict[current_key]

            img1 = cv2.imread(str(file1_path))
            img2 = cv2.imread(str(file2_path))

            if img1 is None or img2 is None:
                print(f"[X] 이미지를 읽을 수 없습니다: {current_key}")
                current_idx = (current_idx + 1) % len(common_files)
                continue

            # 화면 맞춤 리사이즈 (최대 높이 800)
            def resize_max_h(img, max_h=800):
                h, w = img.shape[:2]
                if h > max_h:
                    scale = max_h / h
                    return cv2.resize(img, (int(w*scale), max_h))
                return img

            img1 = resize_max_h(img1, 800)
            img2 = resize_max_h(img2, 800)

            # 높이 맞춤 (작은 쪽에 상하 패딩 X, 큰쪽에만 맞춰 아래 패딩)
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_h = max(h1, h2)
            if h1 < max_h:
                img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            if h2 < max_h:
                img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

            combined = np.hstack([img1, img2])

            title = f"[{current_idx+1}/{len(common_files)}] {current_key}"
            folder1_name = p1.name
            folder2_name = p2.name
            subtitle = f"{folder1_name} vs {folder2_name}"

            text_height = 60
            text_img = np.zeros((text_height, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(text_img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(text_img, subtitle, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 가운데 구분선
            line_x = img1.shape[1]
            cv2.line(combined, (line_x, 0), (line_x, combined.shape[0]), (0, 255, 0), 2)

            final_img = np.vstack([text_img, combined])
            cv2.imshow('Image Comparison', final_img)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                current_idx = (current_idx + 1) % len(common_files)
            elif key == ord('a'):
                current_idx = (current_idx - 1) % len(common_files)
            else:
                current_idx = (current_idx + 1) % len(common_files)

    except KeyboardInterrupt:
        print("\n[INFO] 사용자가 종료했습니다.")
    except Exception as e:
        print(f"[ERROR] 오류 발생: {e}")
    finally:
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 3:
        print("사용법: python check.py <폴더1> <폴더2>")
        print("예시: python check.py compose_final compose_improved")
        print("또는: python check.py C:/path/to/folder1 C:/path/to/folder2")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
    compare_folders(folder1, folder2)

if __name__ == "__main__":
    main()
