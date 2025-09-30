# flatten_alpha.py
import os, argparse, cv2, numpy as np
from pathlib import Path

def parse_color(s: str):
    s = s.lower().strip()
    presets = {
        'white': (255,255,255),
        'black': (0,0,0),
        'gray': (128,128,128),
        'grey': (128,128,128),
    }
    if s in presets:
        return presets[s]
    if s.startswith('#') and len(s)==7:
        r = int(s[1:3], 16); g = int(s[3:5],16); b = int(s[5:7],16)
        return (b,g,r)  # BGR
    raise ValueError(f'지원하지 않는 색상: {s}')

def auto_bg_bgr(img_bgra: np.ndarray) -> tuple[int,int,int]:
    b,g,r,a = cv2.split(img_bgra)
    # 우선순위1: 알파=0인 배경 픽셀
    bg_mask = (a==0)
    if np.any(bg_mask):
        b_med = int(np.median(b[bg_mask])); g_med = int(np.median(g[bg_mask])); r_med = int(np.median(r[bg_mask]))
        return (b_med, g_med, r_med)
    # 우선순위2: 테두리 픽셀들 (가장자리 평균)
    h,w = a.shape
    border = np.zeros_like(a, dtype=bool)
    border[0,:]=True; border[-1,:]=True; border[:,0]=True; border[:,-1]=True
    b_med = int(np.median(b[border])); g_med = int(np.median(g[border])); r_med = int(np.median(r[border]))
    return (b_med, g_med, r_med)

def main():
    ap = argparse.ArgumentParser(description='Flatten RGBA to RGB by compositing alpha onto a background color')
    ap.add_argument('--inp',  type=str, default='pred_alpha', help='입력 RGBA 폴더')
    ap.add_argument('--out',  type=str, default='data_carbon/images', help='출력 RGB 폴더')
    ap.add_argument('--bg',   type=str, default='white', help="배경색: white/black/gray 또는 #RRGGBB 또는 'auto'")
    ap.add_argument('--tight', action='store_true', help='알파>0 영역으로 타이트 크롭')
    ap.add_argument('--jpg',  action='store_true', help='JPG로 저장(기본은 PNG)')
    ap.add_argument('--quality', type=int, default=95, help='JPG 품질(기본 95)')
    args = ap.parse_args()

    in_dir = Path(args.inp); out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ('.png','.tif','.tiff','.webp')
    files = [p for p in sorted(in_dir.glob('*')) if p.suffix.lower() in exts]
    if not files:
        print(f'입력 파일이 없습니다: {in_dir}'); return

    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # 기대: BGRA
        if img is None or img.ndim!=3 or img.shape[2]!=4:
            print(f'[skip] RGBA 아님: {p.name}')
            continue

        b,g,r,a = cv2.split(img)
        rgb = cv2.merge([b,g,r]).astype(np.float32)
        a_f = (a.astype(np.float32)/255.0)[...,None]

        # 배경색 결정
        if args.bg.lower()=='auto':
            bg_color = auto_bg_bgr(img)           # (b,g,r)
        else:
            bg_color = parse_color(args.bg)       # (b,g,r)
        bg = np.zeros_like(rgb, dtype=np.float32)
        bg[:] = np.array(bg_color, dtype=np.float32)

        # 알파 합성: out = fg*a + bg*(1-a)
        out = (rgb*a_f + bg*(1.0 - a_f)).clip(0,255).astype(np.uint8)

        # 타이트 크롭 (알파 기준)
        if args.tight:
            ys, xs = np.where(a > 0)
            if len(xs) and len(ys):
                x1, x2 = xs.min(), xs.max()+1
                y1, y2 = ys.min(), ys.max()+1
                out = out[y1:y2, x1:x2]

        # 저장 (3채널)
        stem = p.stem
        if args.jpg:
            cv2.imwrite(str(out_dir / f"{stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        else:
            cv2.imwrite(str(out_dir / f"{stem}.png"), out)
        print(f'-> {stem} 저장 완료')

if __name__ == '__main__':
    main()
