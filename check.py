import cv2, sys, numpy as np
from pathlib import Path

p = Path(sys.argv[1])
img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
if img is None:
    print("[X] read fail"); sys.exit(1)

print("shape:", img.shape, "| dtype:", img.dtype)
if img.ndim==2:
    print("gray image. no alpha"); sys.exit(0)

if img.shape[2]==4:
    b,g,r,a = cv2.split(img)
    vals, cnts = np.unique(a, return_counts=True)
    print("alpha unique:", dict(zip(vals.tolist(), cnts.tolist())))
    print("fg pixels(>0):", int((a>0).sum()))
    print("bg pixels(==0):", int((a==0).sum()))
else:
    print("RGB 3ch. no alpha")