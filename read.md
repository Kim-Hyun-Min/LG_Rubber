# LG_Rubber 세그멘테이션 & 스티칭 파이프라인

이 문서는 **각 스크립트의 역할 / 입출력 / 실행법**을 한눈에 정리한 가이드입니다.  
(Windows PowerShell 기준 예시 포함)

---

## 📦 전체 파이프라인 (요약)

1. **밝기 보정** → `normalize_brightness.py`  
   입력: `data/` → 출력: `out_brightness/`

2. **라벨링** → `rubber_label_gui.py`  
   입력: `out_brightness/` → 출력: `labels/`, `labels/overlay/`

3. **데이터셋 정리** → `data_g/`  
   - `data_g/images/` : `out_brightness/` 복사  
   - `data_g/masks/`  : `labels/` 복사 (파일명 stem 일치)

4. **학습** → `train_unet_g.py`  
   입력: `data_g/images/`, `data_g/masks/`  
   출력: `runs_seg/best_unet.pth`

5. **추론**  
   - 샘플 30장 → `infer_sample_g.py`  
   - 알파 PNG 추출 → `pred_alpha/`  
   - (옵션) RGB 평탄화/크롭 → `prep_rubber_crops.py`

6. **분할 이미지 합성(스티칭)**  
   - 단순 겹침 → `improved_rubber_stamp_stitching.py`  
   - 조명/정렬/밝기보정 포함 고급 합성 → `synsetic.py`, `synsetic_advanced.py`  
   - GUI 도구 → `gui.py`

7. **통합 실행기** → `run_extract.py`  
   - 단일 입력(`--mode single`)  
   - 분할 입력(`--mode split`)  

---

## 🗂️ 권장 폴더 구조

project_root/
├─ normalize_brightness.py
├─ rubber_label_gui.py
├─ train_unet_g.py
├─ infer_sample_g.py
├─ improved_rubber_stamp_stitching.py
├─ synsetic.py
├─ synsetic_advanced.py
├─ gui.py
├─ run_extract.py
├─ make_split_pairs.py
├─ check.py
│
├─ data/ # 원본 이미지
├─ out_brightness/ # 밝기 보정 결과
├─ labels/ # 라벨 마스크(0/255)
│ └─ overlay/
├─ data_g/ # 학습셋
│ ├─ images/
│ └─ masks/
├─ runs_seg/ # 학습 결과
│ └─ best_unet.pth
├─ pred_alpha/ # 추론 알파 PNG
├─ simple_overlap/ # 기본 스티칭 결과
└─ synsetic_final/ # 고급 스티칭 결과

yaml
코드 복사

---

## ⚙️ 스크립트별 설명 & 실행 예시

### 1) 밝기 보정 — `normalize_brightness.py`
LAB L 채널만 보정 (색상 보존).  
지원 방식: `auto`, `clahe`, `l-fixed`, `l-histmatch`, `l-abs`  

```powershell
(.env) PS> python normalize_brightness.py
출력: out_brightness/

2) 라벨링 — rubber_label_gui.py
드래그 박스 지정 후 GrabCut / 임계값 / 색상 분할로 마스크 생성.
저장: labels/, labels/overlay/

키 조작:

좌클릭 드래그: 영역 지정

s: 마스크 저장

o: 오버레이 저장

a/d: 이전/다음

r: 리셋

powershell
코드 복사
(.env) PS> python rubber_label_gui.py
3) 데이터셋 정리
out_brightness/ → data_g/images/

labels/ → data_g/masks/

4) 학습 — train_unet_g.py
경량 UNet (BCE+Dice Loss).
출력: runs_seg/best_unet.pth

powershell
코드 복사
(.env) PS> python train_unet_g.py
5) 추론 — infer_sample_g.py
앞 30장만 알파 PNG 저장.

powershell
코드 복사
(.env) PS> python infer_sample_g.py
출력: pred_alpha/

6) 알파 → RGB 평탄화/크롭 — prep_rubber_crops.py
알파 PNG → RGB 변환 + 크롭.
옵션: --tight, --bg white

powershell
코드 복사
(.env) PS> python prep_rubber_crops.py --inp pred_alpha --out data_carbon/images --bg white --tight
7) 분할 이미지 합성 — improved_rubber_stamp_stitching.py
위/아래 조각 단순 스티칭.
옵션: --original-height (원본 고무 높이)

powershell
코드 복사
(.env) PS> python improved_rubber_stamp_stitching.py --top-dir s_data\img1 --bot-dir s_data\img2 --out s_stitched --original-height 800
고급 버전:

powershell
코드 복사
(.env) PS> python synsetic.py --top-dir s_data\img1 --bot-dir s_data\img2 --out synsetic
(.env) PS> python synsetic_advanced.py --top-dir s_data\img1 --bot-dir s_data\img2 --out synsetic_final
8) 통합 실행기 — run_extract.py
학습된 모델(runs_seg/best_unet.pth)로 자동 실행.

단일 모드(single)

powershell
코드 복사
(.env) PS> python run_extract.py --mode single --inp data --model runs_seg\best_unet.pth
분할 모드(split)

powershell
코드 복사
(.env) PS> python run_extract.py --mode split --top-dir s_data\img1 --bot-dir s_data\img2 --stitch-out s_stitched --original-height 800 --model runs_seg\best_unet.pth
옵션: --skip-flatten (평탄화 생략), --jpg (JPG 저장), --bg (배경색 지정)

9) 기타 유틸
make_split_pairs.py : 원본을 위/아래로 랜덤 분할 (60% ± 오차)

check.py : 알파 채널 존재 여부 및 픽셀 통계 확인

powershell
코드 복사
(.env) PS> python make_split_pairs.py --src data_carbon/images
(.env) PS> python check.py pred_alpha/sample_alpha.png