import cv2
import sys
import numpy as np
from pathlib import Path
import os

def compare_folders(folder1, folder2):
    """두 폴더의 같은 이름 이미지들을 비교하는 함수"""
    
    # 폴더 경로 확인
    p1 = Path(folder1)
    p2 = Path(folder2)
    
    if not p1.exists() or not p2.exists():
        print(f"[X] 폴더가 존재하지 않습니다: {folder1} 또는 {folder2}")
        return
    
    # 각 폴더의 이미지 파일들 찾기
    files1 = [f for f in p1.glob("*.png") if f.is_file()]
    files2 = [f for f in p2.glob("*.png") if f.is_file()]
    
    # 파일명으로 매칭 (_alpha 부분까지만 비교)
    files1_dict = {}
    files2_dict = {}
    
    # 첫 번째 폴더 파일들 처리
    for f in files1:
        stem = f.stem
        # _alpha 부분까지만 추출
        if '_alpha' in stem:
            key = stem.split('_alpha')[0] + '_alpha'
            files1_dict[key] = f
    
    # 두 번째 폴더 파일들 처리  
    for f in files2:
        stem = f.stem
        # _alpha 부분까지만 추출
        if '_alpha' in stem:
            key = stem.split('_alpha')[0] + '_alpha'
            files2_dict[key] = f
    
    # 공통 파일명 찾기
    common_files = set(files1_dict.keys()) & set(files2_dict.keys())
    common_files = sorted(common_files)
    
    if not common_files:
        print("[X] 공통된 파일명이 없습니다.")
        return
    
    print(f"[INFO] 총 {len(common_files)}개의 공통 파일을 찾았습니다.")
    print("[INFO] 사용법:")
    print("  - 'd' 키: 다음 이미지 쌍")
    print("  - 'a' 키: 이전 이미지 쌍") 
    print("  - 'q' 키: 종료")
    print("  - 다른 키: 다음 이미지로 이동")
    print("-" * 50)
    
    current_idx = 0
    
    try:
        while True:
            # 현재 파일 정보
            current_file = common_files[current_idx]
            file1_path = files1_dict[current_file]
            file2_path = files2_dict[current_file]
            
            # 이미지 읽기
            img1 = cv2.imread(str(file1_path))
            img2 = cv2.imread(str(file2_path))
            
            if img1 is None or img2 is None:
                print(f"[X] 이미지를 읽을 수 없습니다: {current_file}")
                current_idx = (current_idx + 1) % len(common_files)
                continue
            
            # 이미지 크기 조정 (화면에 맞게)
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # 최대 높이를 800으로 제한
            max_height = 800
            if h1 > max_height:
                scale = max_height / h1
                new_w1 = int(w1 * scale)
                img1 = cv2.resize(img1, (new_w1, max_height))
            
            if h2 > max_height:
                scale = max_height / h2
                new_w2 = int(w2 * scale)
                img2 = cv2.resize(img2, (new_w2, max_height))
            
            # 두 이미지를 나란히 배치
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # 높이를 맞춤
            max_h = max(h1, h2)
            if h1 < max_h:
                img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            if h2 < max_h:
                img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            
            # 이미지 합치기
            combined = np.hstack([img1, img2])
            
            # 제목 추가
            title = f"[{current_idx+1}/{len(common_files)}] {current_file}"
            folder1_name = p1.name
            folder2_name = p2.name
            subtitle = f"{folder1_name} vs {folder2_name}"
            
            # 텍스트 영역 추가
            text_height = 60
            text_img = np.zeros((text_height, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(text_img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(text_img, subtitle, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 구분선 추가 (가운데)
            line_x = img1.shape[1]
            cv2.line(combined, (line_x, 0), (line_x, combined.shape[0]), (0, 255, 0), 2)
            
            # 최종 이미지
            final_img = np.vstack([text_img, combined])
            
            # 화면에 표시
            cv2.imshow('Image Comparison', final_img)
            
            # 키 입력 대기
            key = cv2.waitKey(0) & 0xFF  # 다시 0으로 변경하여 키 입력까지 대기
            
            if key == ord('q'):  # 종료
                break
            elif key == ord('d'):  # 다음
                current_idx = (current_idx + 1) % len(common_files)
            elif key == ord('a'):  # 이전
                current_idx = (current_idx - 1) % len(common_files)
            else:  # 다른 키는 다음으로 처리
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