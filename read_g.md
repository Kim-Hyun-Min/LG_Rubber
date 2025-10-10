# 고무도장 이미지 합성 도구 (compose_advanced.py)

분할 촬영된 고무도장 이미지(상단/하단)를 원본 높이로 복원하는 고급 이미지 합성 프로그램입니다.

## 주요 특징

### 🎯 조명 차이 강건성
- **4가지 조명 불변 지표** 사용으로 밝기 차이가 있어도 정확한 매칭
- Affine MSE, Z-score MSE, Gradient MSE, Stripe Score 혼합 평가
- 반사광, 그림자 등 이상치에 강건한 알고리즘

### 🔍 2단계 탐색 전략
- **Stage 1 (Coarse)**: 5~40% 범위 전역 탐색으로 최적 후보 발견
- **Stage 2 (Refine)**: 후보 주변 ±2% 정밀 탐색으로 픽셀 단위 최적화
- 초기 추정값에 의존하지 않는 견고한 탐색

### 🧠 하이브리드 의사결정
- **기하학적 추정** (원본 높이 기반) + **프로파일 매칭** 결합
- 다중 지표 신뢰도 평가로 보수적 결정
- 정보 손실 감지 및 경고 시스템

---

## 설치

### 필수 패키지
```bash
pip install opencv-python numpy
```

### Python 버전
- Python 3.7 이상

---

## 사용 방법

### 기본 사용법

```bash
python compose_improved.py \
  --top-dir ./s_data/img1 \
  --bot-dir ./s_data/img2 \
  --out ./output
```

### 원본 높이 자동 추정

```bash
python compose_improved.py \
  --top-dir ./s_data/img1 \
  --bot-dir ./s_data/img2 \
  --original-dir ./original_stamps \
  --out ./output
```

### 공통 원본 높이 지정

```bash
python compose_improved.py \
  --top-dir ./s_data/img1 \
  --bot-dir ./s_data/img2 \
  --original-height 2000 \
  --out ./output
```

### 개별 높이 파일 사용

```bash
# heights.txt 형식:
# stamp001.png,2000
# stamp002.png,1850
# stamp003.png,2100

python compose_improved.py \
  --top-dir ./s_data/img1 \
  --bot-dir ./s_data/img2 \
  --height-file heights.txt \
  --out ./output
```

---

## 명령행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--top-dir` | 상단 이미지 폴더 | `./s_data/img1` |
| `--bot-dir` | 하단 이미지 폴더 | `./s_data/img2` |
| `--out` | 출력 폴더 | `compose_improved` |
| `--original-dir` | 원본 이미지 폴더 (높이 추정용) | None |
| `--original-height` | 공통 원본 높이 (픽셀) | None |
| `--height-file` | 개별 높이 정보 파일 | None |
| `--bg-threshold` | 배경 판단 임계값 (0-255) | 245 |

---

## 폴더 구조

```
project/
├── compose_improved.py
├── s_data/
│   ├── img1/              # 상단 이미지
│   │   ├── stamp001.png
│   │   ├── stamp002.png
│   │   └── ...
│   └── img2/              # 하단 이미지
│       ├── stamp001.png
│       ├── stamp002.png
│       └── ...
├── original_stamps/       # (선택) 원본 이미지
│   ├── stamp001.png
│   └── ...
└── output/                # 결과 출력
    ├── stamp001_restored.png
    └── ...
```

**중요**: 상단/하단 이미지는 **파일명이 동일**해야 합니다 (확장자는 달라도 무관).

---

분할 후 중앙값
- **효과**: 반사광, 그림자 등 이상치 억제

### 신뢰도 평가 기준

```python
✓ High confidence (0.95):
  - Affine MSE < 300
  - Z-score Correlation > 0.6
  - Stripe Variance < 50

⚠ Medium confidence (0.75):
  - 위 조건 중 2개 만족

⚠ Low confidence (0.50):
  - 위 조건 중 1개 만족

✗ Very low (0.20):
  - 모든 조건 불만족
```

---

## 출력 예시

```
Processing: stamp001

  Content heights - Top: 1050, Bot: 1120
  ✓ 원본 높이 추정: 2000px (from stamp001.png)

  [Stage 1] Coarse search: 50~800px (step=25)
  [Stage 1] Best: 325px (score=45.23)

  [Stage 2] Refine search: 285~365px (±40px)
  [Stage 2] Best: 318px (combined=42.15)
            Affine MSE=245.3, Z-MSE=0.38
            Grad MSE=182.7, Corr(Z)=0.823
            Stripe var=28.4

  ✓ High confidence match (all criteria passed)
  Geometric estimate: 320px (confidence=0.7)
  Final overlap: 319px (hybrid(α=0.58,geom=320,match=318))

  Result: 1500×1999
  Accuracy: -1px (-0.1%)
  ✓ Saved: stamp001_restored.png

==================================================
SUCCESS: 100/100
Avg error: 2.3px
Output: ./output
```

---

## 알고리즘 흐름도

```
┌─────────────────────────────────────────┐
│  1. 입력 이미지 로드 (Top / Bottom)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2. 고무 영역 경계 분석                   │
│     - 배경(bg_threshold) 기반 마스킹     │
│     - 연속 구간 추출 → 메인 영역 식별    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3. 원본 높이 추정 (선택)                │
│     - original-dir 이미지 분석           │
│     - 기하학적 초기값 계산               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  4. 이미지 리사이즈 (너비 정렬)           │
│     - 비례 리사이즈 (LANCZOS4)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  5. 2단계 겹침점 탐색                     │
│  ┌─────────────────────────────────┐    │
│  │ Stage 1: Coarse (5~40%)         │    │
│  │  - 4지표 혼합 평가               │    │
│  │  - 최적 후보 선정               │    │
│  └────────┬────────────────────────┘    │
│           ▼                              │
│  ┌─────────────────────────────────┐    │
│  │ Stage 2: Refine (±2%)           │    │
│  │  - 1px 단위 정밀 탐색            │    │
│  │  - 다중 지표 신뢰도 평가         │    │
│  └────────┬────────────────────────┘    │
└───────────┼──────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│  6. 하이브리드 의사결정                   │
│     - 기하 추정 vs 프로파일 매칭 비교    │
│     - 신뢰도 기반 가중 평균              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  7. 프로파일 인식 블렌딩                  │
│     - 행별 적응적 알파 블렌딩            │
│     - S-curve 부드러운 전환              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  8. 최종 합성 및 저장                     │
│     - 캔버스 생성 (배경: #F8F5F0)        │
│     - PNG 출력                           │
└─────────────────────────────────────────┘
```

---

## 문제 해결

### Q1. "No matching pairs found" 오류
**원인**: 상단/하단 폴더의 파일명이 다름  
**해결**: 파일명 (확장자 제외)을 동일하게 맞추세요

```bash
# 예시
img1/stamp001.png
img2/stamp001.jpg  # 확장자는 달라도 OK
```

### Q2. "Analysis failed" 경고
**원인**: 배경 임계값이 부적절  
**해결**: `--bg-threshold` 조정 (기본값: 245)

```bash
# 어두운 배경인 경우
python compose_improved.py --bg-threshold 200

# 밝은 배경인 경우
python compose_improved.py --bg-threshold 250
```

### Q3. 복원 높이 오차가 큼
**원인**: 원본 높이 정보 부재  
**해결**: `--original-dir` 또는 `--original-height` 제공

```bash
# 방법 1: 원본 폴더 제공 (자동 추정)
python compose_improved.py --original-dir ./originals

# 방법 2: 높이 직접 지정
python compose_improved.py --original-height 2000
```

### Q4. 조명 차이가 심해도 실패
**원인**: 매우 극단적인 조명 변화  
**해결**: 
1. 현재 알고리즘으로도 대부분 해결되지만
2. 필요시 `aff_mse`, `corr_z`, `stripe_var` 임계값 완화

```python
# compose_improved.py 내부 수정
# Line ~245 부근
cond1 = aff_mse < 500      # 300 → 500으로 완화
cond2 = corr_z > 0.4       # 0.6 → 0.4로 완화
cond3 = stripe_var < 100   # 50 → 100으로 완화
```

---

## 성능 최적화 팁

### 대량 배치 처리
```python
# 멀티프로세싱 추가 (향후 버전)
from multiprocessing import Pool

with Pool(processes=4) as pool:
    pool.map(stitch_with_profile_analysis, image_pairs)
```

### 메모리 절약
```bash
# 큰 이미지는 리사이즈 후 처리
# 내부 코드에서 자동으로 너비 정렬하므로 별도 작업 불필요
```

---

## 기술 세부사항

### 지원 이미지 형식
- PNG, JPG/JPEG, BMP, TIF/TIFF, WebP

### 색공간 처리
- 입력: RGB
- 분석: Grayscale 변환 후 처리
- 출력: RGB (배경 #F8F5F0)

### 보간 방법
- 리사이즈: LANCZOS4 (고품질)
- 블렌딩: S-curve 적응적 알파

---

## 라이선스

MIT License

---

## 버전 히스토리

### v2.0 (2025-10-02)
- ✨ 2단계 탐색 전략 (Coarse→Refine)
- ✨ 4가지 조명 불변 지표 (Affine, Z-score, Gradient, Stripe)
- ✨ 다중 지표 신뢰도 평가
- ✨ 하이브리드 의사결정 (기하+매칭)
- 🐛 조명 차이 환경에서 정확도 대폭 개선

### v1.0 (Initial)
- 기본 프로파일 매칭
- 단일 MSE 평가
- 원본 높이 기반 초기 추정

---

## 문의 및 기여

문제가 발생하거나 개선 제안이 있으시면 이슈를 등록해주세요.

---

## 참고 자료

### 알고리즘 논문
- Image Stitching using Affine Transformations
- Robust Image Matching under Illumination Variations
- Multi-Scale Feature Detection for Image Registration

### 사용 라이브러리
- [OpenCV](https://opencv.org/) - 이미지 처리
- [NumPy](https://numpy.org/) - 수치 연산