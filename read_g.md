 현재까지 진행된 작업 요약
1. 데이터 전처리 (밝기 보정)

스크립트: normalize_brightness.py

역할: 원본(data/) 이미지를 LAB 색공간에서 L(밝기)만 보정 → out_brightness/에 저장

방식: clahe, l-fixed, l-histmatch, l-abs 중 선택 가능 (기본 l-abs)

📌 CLAHE (Contrast Limited Adaptive Histogram Equalization)

풀네임: 대비 제한 적응형 히스토그램 평활화

일반 히스토그램 평활화(HE) 는 전체 이미지를 대상으로 밝기 분포를 균등하게 만드는 방법입니다.

하지만 HE는 과도한 대비 증가(노이즈 강조) 문제가 있습니다.

CLAHE는 이를 개선한 방식으로:

이미지를 작은 블록(tile, region)으로 나눔

각 블록마다 히스토그램 평활화 적용

clip limit(대비 제한 값) 으로 과도한 대비를 억제

블록 간 경계는 보간(interpolation) 으로 매끄럽게 연결

2. 라벨링 (고무 부분 마스크 생성)

스크립트: rubber_label_gui.py

역할: 드래그로 영역 지정 후 GrabCut/임계값/색상 기반 방법으로 고무 영역 마스크(0/255) 생성

입력: out_brightness/

출력: labels/ (마스크), labels/overlay/ (원본+마스크 오버레이)

3. 데이터셋 정리

구조: data_g/

data_g/images/ : out_brightness/ 복사

data_g/masks/ : labels/ 복사 (파일명 stem 일치 필요)

목적: 학습용 (이미지-마스크 1:1 매칭)

4. 학습 (고무 부분 추출)

스크립트: train_unet_g.py

모델: 경량 UNet (in_ch=3, out_ch=1)

손실: BCE + Dice Loss

학습 데이터: data_g/images, data_g/masks

출력: runs_seg/best_unet.pth

5. 추론 (고무 부분 확인)

스크립트: infer_sample_g.py

역할: 학습된 모델(best_unet.pth)로 out_brightness/ 앞 30장 추론

출력: pred_alpha/ 에 RGB+Alpha PNG 저장 (고무 영역만 투명 배경 분리)


make_split_pairs.py 랜덤하게 분리

>> imporoved_rubber_stamp_stiching.py 이미지 합성1 # 추후 이미지의 높이가 일정해질때 발전 가능성 up

1. 겹침 계산 방식: 고무 내용의 상·하 경계를 밝기 임계치로 찾아 내용 높이를 구한 뒤, 주어진 원본 높이가 있으면 top+bot-원본으로 겹침을 계산합니다. 원본 높이가 없으면 “60% 찍힘=20% 겹침” 가정으로 폴백합니다 .

2. 경계 추출: find_rubber_content_bounds에서 행별 비배경 비율>5% 구간을 내용으로 판단합니다 .

3. 계산: calculate_simple_overlap 에서 원본 높이 제공 시 정확 계산, 없으면 20% 가정 폴백을 사용합니다 .

4. 블렌딩: 겹치는 세로 구간을 위→아래로 알파를 선형 증가시키는 간단한 알파 블렌딩을 수행합니다(행마다 α를 0→1로 변화) .

5. 입출력/흐름: 폭을 더 작은 쪽에 맞춘 뒤, 겹침 계산→블렌딩→캔버스 배치→저장 순서입니다. CLI는 --original-height/--height-file 지원, 기본 출력은 _simple.png 입니다 .

python improved_rubber_stamp_stitching.py --top-dir ./s_data/img1 --bot-dir ./s_data/img2 


5. 입출력/옵션: --assume-aspect, --aspect-tol, --aspect-smooth, --aspect-calib, --blend-mode 등을 지원하여 현장 편차에 대한 방어막을 제공합니다

python improved_rubber_stamp_stitching_test.py --top-dir s_data\img1 --bot-dir s_data\img2 --blend-mode cosine


>> synsetic.py 이미지 합성3 # 추후 조명을 보다 명확히 하고 오차를 명확히 하면 발전 가능성 up

1. 이미지 로드 및 전처리
2. 고무 영역 분석 (조명 프로파일 포함)
3. 최적 겹침점 계산
4. 적응적 블렌딩
5. 최종 합성 및 저장


# 고무도장 이미지 분할 및 복원 시스템

고무도장 이미지를 위/아래로 분할한 후 다시 복원하는 파이프라인입니다.

## 개요

- **분할**: 원본 이미지를 55~65% 범위로 랜덤하게 위/아래로 자릅니다
- **복원**: 잘린 이미지들을 원본 높이 정보를 바탕으로 정확하게 복원합니다

## 요구사항

```bash
pip install opencv-python numpy
```

## 파일 구조

```
project/
├── make_split_pairs.py          # 이미지 분할 스크립트
├── synsetic_advanced.py         # 이미지 복원 스크립트
├── data_carbon/
│   └── images/                  # 원본 이미지 폴더
├── s_data/
│   ├── img1/                    # 분할된 위쪽 이미지
│   └── img2/                    # 분할된 아래쪽 이미지
└── restored_output/             # 복원된 이미지 출력
```

## 사용법

### 1단계: 이미지 분할

원본 이미지를 위/아래로 랜덤하게 자릅니다.

```bash
python make_split_pairs.py \
  --src data_carbon/images \
  --out1 s_data/img1 \
  --out2 s_data/img2 \
  --ratio-min 0.55 \
  --ratio-max 0.65
```

**옵션:**
- `--src`: 원본 이미지 폴더 (기본값: `data_carbon/images`)
- `--out1`: 위쪽 조각 저장 폴더 (기본값: `s_data/img1`)
- `--out2`: 아래쪽 조각 저장 폴더 (기본값: `s_data/img2`)
- `--ratio-min`: 최소 보이는 비율 (기본값: 0.55)
- `--ratio-max`: 최대 보이는 비율 (기본값: 0.65)
- `--mean`: 랜덤 비율의 평균 (기본값: 0.60)
- `--std`: 표준편차 (기본값: 0.03)
- `--seed`: 랜덤 시드 (기본값: 42)
- `--jpg`: JPG로 저장 (기본값: PNG)
- `--quality`: JPG 품질 (기본값: 95)

**결과:**
- 각 원본 이미지당 2개의 파일 생성 (위/아래)
- 55~65% 범위로 랜덤하게 자름 (평균 60%)
- 겹침 영역: 10~30% 보장

### 2단계: 이미지 복원

분할된 이미지를 원본으로 복원합니다.

```bash
python synsetic_advanced.py \
  --top-dir s_data/img1 \
  --bot-dir s_data/img2 \
  --out restored_output \
  --original-dir data_carbon/images
```

**옵션:**
- `--top-dir`: 위쪽 이미지 폴더 (기본값: `./s_data/img1`)
- `--bot-dir`: 아래쪽 이미지 폴더 (기본값: `./s_data/img2`)
- `--out`: 출력 폴더 (기본값: `synsetic`)
- `--original-dir`: 원본 이미지 폴더 (높이 자동 추정용)
- `--original-height`: 모든 이미지에 공통 높이 지정
- `--height-file`: 개별 높이 정보 파일 (CSV 형식)
- `--bg-threshold`: 배경 임계값 (기본값: 245)

**원본 높이 제공 방법:**

방법 1: 원본 폴더 지정 (권장)
```bash
--original-dir data_carbon/images
```
자동으로 원본 이미지를 찾아 고무 영역 높이를 측정합니다.

방법 2: 공통 높이 지정
```bash
--original-height 1000
```
모든 이미지가 같은 높이일 때 사용합니다.

방법 3: 개별 높이 파일
```bash
--height-file heights.txt
```
파일 형식:
```
image1.png,1024
image2.png,1056
image3.png,998
```

## 동작 원리

### 분할 알고리즘

```python
ratio = 랜덤(0.55~0.65)  # 가우시안 분포, 평균 0.60
crop_h = int(height * ratio)

top = image[0:crop_h]              # 위: 0 ~ ratio
bottom = image[height-crop_h:]     # 아래: (1-ratio) ~ 1
```

예시 (원본 높이 1000px):
- ratio=0.60 선택
- 위: 0~600px (60%)
- 아래: 400~1000px (60%)
- 겹침: 600+600-1000 = 200px (20%)

### 복원 알고리즘

1. **고무 영역 감지**: 배경(밝은 영역)을 제외한 고무 영역 탐지
2. **원본 높이 추정**: 원본 이미지에서 고무 영역의 실제 높이 측정
3. **최적 겹침 계산**: `overlap = top_height + bot_height - original_height`
4. **프로파일 블렌딩**: 밝기 분석 기반 적응적 블렌딩
5. **합성**: S-curve 기반 부드러운 전환으로 자연스러운 복원

## 주요 특징

- **정보 손실 감지**: 위/아래 이미지가 겹치지 않으면 경고 출력
- **자동 높이 추정**: 원본 폴더만 지정하면 자동으로 높이 측정
- **적응적 블렌딩**: 조명 조건에 따라 블렌딩 가중치 자동 조정
- **비례적 리사이즈**: 너비가 다른 이미지 자동 조정
- **정확도 측정**: 복원 결과와 원본 높이 비교

## 제약사항

원본 이미지가 다음 조건을 만족해야 합니다:
- **전체 고무가 완전히 보여야 함**
- 상단/하단이 잘리지 않은 완전한 형태
- 배경이 충분히 밝음 (임계값 245 이상)

잘린 이미지는 복원 정확도가 떨어질 수 있습니다.

## 출력 예시

```
[  1/ 10] Processing: rubber_001
  ✓ 원본 높이 추정: 1024px (from rubber_001.png)
  Input: Top 800×614, Bot 800×618
  Content heights - Top: 612, Bot: 616
  Final overlap: 204 pixels (original_height_based)
  Result: 800×1024
  Accuracy: 0px (0.0%)
  ✓ Saved: rubber_001_restored.png
```

## 문제 해결

**Q: "정보 손실 감지!" 경고가 뜹니다**
A: 위/아래 이미지가 겹치지 않습니다. 원본 이미지가 이미 잘려있거나 분할 비율이 잘못되었을 수 있습니다.

**Q: 복원 높이가 원본과 다릅니다**
A: `--bg-threshold` 값을 조정해보세요. 배경이 어두우면 값을 낮추고, 밝으면 값을 높입니다.

**Q: 원본 높이를 찾을 수 없습니다**
A: 원본 폴더(`--original-dir`)에 같은 이름의 파일이 있는지 확인하세요. 확장자가 달라도 자동으로 찾습니다.

**Q: 블렌딩 경계가 부자연스럽습니다**
A: 겹침 영역이 너무 작을 수 있습니다. `--ratio-min`과 `--ratio-max` 범위를 조정하여 더 큰 겹침을 만드세요.

## 라이센스

MIT License

## 작성자

2025년 고무도장 복원 프로젝트