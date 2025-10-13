#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubber 이미지 처리 파이프라인
1. 밝기 정규화 - Top (data/img1 → out_brightness/img1)
2. 밝기 정규화 - Bot (data/img2 → out_brightness/img2)
3. 알파 마스크 추론 - Top (out_brightness/img1 → pred_alpha/top)
4. 알파를 흰 배경으로 변환 - Top (pred_alpha/top → pred_whitebg/top)
5. 알파 마스크 추론 - Bot (out_brightness/img2 → pred_alpha/bot)
6. 알파를 흰 배경으로 변환 - Bot (pred_alpha/bot → pred_whitebg/bot)
7. 이미지 합성 (pred_whitebg/top,bot → compose_result)
"""

import subprocess, sys
from pathlib import Path

PY = sys.executable

# 스크립트 경로
NORM_BRIGHT = Path("normalize_brightness.py")
INFER       = Path("infer_sample_g.py")
PREP_CROPS  = Path("prep_rubber_crops.py")
COMPOSE     = Path("compose_improved.py")

# 입력/출력 폴더 경로
INPUT_IMG1           = "data/img1"                     # 원본 윗부분
INPUT_IMG2           = "data/img2"                     # 원본 아랫부분
OUT_BRIGHT_TOP       = "out_brightness/img1"           # 1단계: 밝기 정규화 - Top
OUT_BRIGHT_BOT       = "out_brightness/img2"           # 2단계: 밝기 정규화 - Bot
OUT_ALPHA_TOP        = "pred_alpha/top"                # 3단계: Top 알파 마스크 (RGBA)
OUT_ALPHA_BOT        = "pred_alpha/bot"                # 5단계: Bot 알파 마스크 (RGBA)
OUT_WHITEBG_TOP      = "pred_whitebg/top"              # 4단계: Top 흰 배경 (RGB)
OUT_WHITEBG_BOT      = "pred_whitebg/bot"              # 6단계: Bot 흰 배경 (RGB)
OUT_COMPOSE          = "compose_result"                # 7단계: 최종 합성 결과

# 옵션
INFER_COMMON_OPTS  = ["--weights", "runs_seg/best_unet.pth", "--save-rgba"]
PREP_COMMON_OPTS   = ["--bg", "white"]                 # 흰 배경 합성
COMPOSE_OPTS       = []                                # compose 옵션

def run(cmd):
    """명령어 실행"""
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, shell=False)
    if r.returncode != 0:
        print(f"[X] 실패 (코드 {r.returncode}) — 실행 중단")
        sys.exit(r.returncode)

def require(p, kind="경로"):
    """필수 파일/폴더 확인"""
    if not Path(p).exists():
        print(f"[X] {kind} 없음: {p}")
        sys.exit(1)

def main():
    print("="*70)
    print(" " * 15 + "Rubber 이미지 처리 파이프라인")
    print("="*70)
    
    # 스크립트 및 입력 폴더 확인
    for p, kind in [
        (NORM_BRIGHT, "스크립트"),
        (INFER, "스크립트"),
        (PREP_CROPS, "스크립트"),
        (COMPOSE, "스크립트"),
        (INPUT_IMG1, "입력 폴더 (img1)"),
        (INPUT_IMG2, "입력 폴더 (img2)")
    ]:
        require(p, kind)

    # 1) 밝기 정규화 - Top: data/img1 → out_brightness/img1
    print("\n" + "="*70)
    print("[단계 1/7] 밝기 정규화 - Top (윗부분)")
    print("="*70)
    print(f"입력: {INPUT_IMG1}")
    print(f"출력: {OUT_BRIGHT_TOP}")
    run([PY, str(NORM_BRIGHT),
         "--inp", INPUT_IMG1,
         "--out", OUT_BRIGHT_TOP])

    # 2) 밝기 정규화 - Bot: data/img2 → out_brightness/img2
    print("\n" + "="*70)
    print("[단계 2/7] 밝기 정규화 - Bot (아랫부분)")
    print("="*70)
    print(f"입력: {INPUT_IMG2}")
    print(f"출력: {OUT_BRIGHT_BOT}")
    run([PY, str(NORM_BRIGHT),
         "--inp", INPUT_IMG2,
         "--out", OUT_BRIGHT_BOT])

    # 3) Top 알파 마스크 추론: out_brightness/img1 → pred_alpha/top
    print("\n" + "="*70)
    print("[단계 3/7] Top 알파 마스크 추론 (고무 부분 추출)")
    print("="*70)
    print(f"입력: {OUT_BRIGHT_TOP}")
    print(f"출력: {OUT_ALPHA_TOP}")
    run([PY, str(INFER),
         "--inp", OUT_BRIGHT_TOP,
         "--out", OUT_ALPHA_TOP,
         *INFER_COMMON_OPTS])

    # 4) Top 알파를 흰 배경으로 변환: pred_alpha/top → pred_whitebg/top
    print("\n" + "="*70)
    print("[단계 4/7] Top RGBA → 흰 배경 RGB 변환")
    print("="*70)
    print(f"입력: {OUT_ALPHA_TOP}")
    print(f"출력: {OUT_WHITEBG_TOP}")
    run([PY, str(PREP_CROPS),
         "--inp", OUT_ALPHA_TOP,
         "--out", OUT_WHITEBG_TOP,
         *PREP_COMMON_OPTS])

    # 5) Bot 알파 마스크 추론: out_brightness/img2 → pred_alpha/bot
    print("\n" + "="*70)
    print("[단계 5/7] Bot 알파 마스크 추론 (고무 부분 추출)")
    print("="*70)
    print(f"입력: {OUT_BRIGHT_BOT}")
    print(f"출력: {OUT_ALPHA_BOT}")
    run([PY, str(INFER),
         "--inp", OUT_BRIGHT_BOT,
         "--out", OUT_ALPHA_BOT,
         *INFER_COMMON_OPTS])

    # 6) Bot 알파를 흰 배경으로 변환: pred_alpha/bot → pred_whitebg/bot
    print("\n" + "="*70)
    print("[단계 6/7] Bot RGBA → 흰 배경 RGB 변환")
    print("="*70)
    print(f"입력: {OUT_ALPHA_BOT}")
    print(f"출력: {OUT_WHITEBG_BOT}")
    run([PY, str(PREP_CROPS),
         "--inp", OUT_ALPHA_BOT,
         "--out", OUT_WHITEBG_BOT,
         *PREP_COMMON_OPTS])

    # 7) 이미지 합성: pred_whitebg/top + pred_whitebg/bot → compose_result
    print("\n" + "="*70)
    print("[단계 7/7] Top + Bot 이미지 합성 (Stitching)")
    print("="*70)
    print(f"입력 Top: {OUT_WHITEBG_TOP}")
    print(f"입력 Bot: {OUT_WHITEBG_BOT}")
    print(f"출력: {OUT_COMPOSE}")
    run([PY, str(COMPOSE),
         "--top-dir", OUT_WHITEBG_TOP,
         "--bot-dir", OUT_WHITEBG_BOT,
         "--out", OUT_COMPOSE,
         *COMPOSE_OPTS])

    # 완료 메시지
    print("\n" + "="*70)
    print(" " * 25 + "✅ 모든 작업 완료!")
    print("="*70)
    print("\n📁 결과 폴더:")
    print(f"  1. 밝기 정규화 Top    : {OUT_BRIGHT_TOP}")
    print(f"  2. 밝기 정규화 Bot    : {OUT_BRIGHT_BOT}")
    print(f"  3. Top 알파 마스크    : {OUT_ALPHA_TOP} (RGBA)")
    print(f"  4. Top 흰 배경        : {OUT_WHITEBG_TOP} (RGB)")
    print(f"  5. Bot 알파 마스크    : {OUT_ALPHA_BOT} (RGBA)")
    print(f"  6. Bot 흰 배경        : {OUT_WHITEBG_BOT} (RGB)")
    print(f"  7. 최종 합성 결과     : {OUT_COMPOSE}")
    print("="*70)

if __name__ == "__main__":
    main()
