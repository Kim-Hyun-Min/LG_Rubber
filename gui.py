import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
# scikit-learn이 설치되어 있다고 가정합니다 (pip install scikit-learn)
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

IMG_DIR = 'out_brightness'
LABEL_DIR = 'labels'
os.makedirs(LABEL_DIR, exist_ok=True)

img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
img_files.sort()

class ImprovedRubberLabelGUI:
    def __init__(self, master):
        self.master = master
        self.master.title('개선된 고무 라벨링 툴')
        self.idx = 0
        self.img = None
        self.original_img = None
        self.tk_img = None
        self.canvas_width = 800
        self.canvas_height = 600
        
        # UI 설정
        self.setup_ui()
        
        # 상태 변수
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.mask = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.current_method = "grabcut"  # "grabcut", "threshold", "color"

        self.load_image()

        # 이벤트 바인딩
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        # 키 바인딩
        master.bind_all('<Key-d>', self.next_img)
        master.bind_all('<Key-a>', self.prev_img)
        master.bind_all('<Key-s>', self.save_mask_only)
        master.bind_all('<Key-o>', self.save_overlay)
        master.bind_all('<Key-r>', self.reset_mask)
        master.bind_all('<Key-1>', lambda e: self.set_method("grabcut"))
        master.bind_all('<Key-2>', lambda e: self.set_method("threshold"))
        master.bind_all('<Key-3>', lambda e: self.set_method("color"))

    def setup_ui(self):
        # 메인 프레임
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 캔버스
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(side=tk.TOP, pady=5)
        
        # 컨트롤 프레임
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 방법 선택 버튼들
        method_frame = tk.Frame(control_frame)
        method_frame.pack(side=tk.LEFT)
        
        tk.Label(method_frame, text="방법:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="grabcut")
        
        tk.Radiobutton(method_frame, text="GrabCut (1)", variable=self.method_var, 
                      value="grabcut", command=self.on_method_change).pack(side=tk.LEFT)
        tk.Radiobutton(method_frame, text="마스크 추가 (2)", variable=self.method_var, 
                      value="threshold", command=self.on_method_change).pack(side=tk.LEFT)
        tk.Radiobutton(method_frame, text="색상 (3)", variable=self.method_var, 
                      value="color", command=self.on_method_change).pack(side=tk.LEFT)
        
        # 파라미터 조정
        param_frame = tk.Frame(control_frame)
        param_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(param_frame, text="임계값:").pack(side=tk.LEFT)
        self.thresh_var = tk.IntVar(value=127)
        self.thresh_scale = tk.Scale(param_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                   variable=self.thresh_var, command=self.on_thresh_change)
        self.thresh_scale.pack(side=tk.LEFT)
        
        # 가장자리 확장 옵션
        tk.Label(param_frame, text="가장자리확장:").pack(side=tk.LEFT, padx=(10,0))
        self.edge_expand_var = tk.BooleanVar(value=True)
        tk.Checkbutton(param_frame, variable=self.edge_expand_var, 
                       text="자동").pack(side=tk.LEFT)
        
        # 버튼들
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        
        tk.Button(button_frame, text="리셋 (R)", command=self.reset_mask).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="저장 (S)", command=self.save_mask_only).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="오버레이만 (O)", command=self.save_overlay).pack(side=tk.LEFT, padx=2)
        
        # 상태 표시
        self.status = tk.Label(main_frame, text='사각형 드래그 → 자동분할 | 1:GrabCut(리셋) 2:마스크추가(누적) 3:색상 | s:저장 r:리셋 a/d:이전/다음', 
                              relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def set_method(self, method):
        self.current_method = method
        self.method_var.set(method)
        
        if method == "threshold":
            self.status.config(text=f'현재 방법: 마스크 추가 (2) | 드래그 영역이 마스크에 누적됩니다.')
        else:
            self.status.config(text=f'현재 방법: {method} | 사각형 드래그하여 분할 영역 지정')

    def on_method_change(self):
        self.current_method = self.method_var.get()
        self.set_method(self.current_method)

    def on_thresh_change(self, val):
        pass 

    def load_image(self):
        if not img_files:
            self.status.config(text='이미지 폴더가 비어있습니다.')
            return
        if self.idx < 0: self.idx = 0
        if self.idx >= len(img_files): self.idx = len(img_files) - 1

        img_path = os.path.join(IMG_DIR, img_files[self.idx])
        bgr = cv2.imread(img_path)
        if bgr is None:
            self.status.config(text=f'로드 실패: {img_files[self.idx]}')
            return

        self.original_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 스케일 계산 및 리사이즈
        orig_h, orig_w = self.original_img.shape[:2]
        self.scale_x = self.canvas_width / orig_w
        self.scale_y = self.canvas_height / orig_h
        
        scale = min(self.scale_x, self.scale_y)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        self.img = cv2.resize(self.original_img, (new_w, new_h))
        self.scale_x = new_w / orig_w
        self.scale_y = new_h / orig_h
        
        self.update_canvas()
        self.reset_state()

        self.status.config(text=f'{img_files[self.idx]} | {self.current_method} 모드 | 사각형 드래그하여 영역 지정')

    def update_canvas(self):
        if self.img is not None:
            img_pil = Image.fromarray(self.img)
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.canvas.delete("all")
            
            # 이미지를 캔버스 중앙에 배치
            img_h, img_w = self.img.shape[:2]
            x_offset = (self.canvas_width - img_w) // 2
            y_offset = (self.canvas_height - img_h) // 2
            
            self.canvas.create_image(x_offset, y_offset, anchor='nw', image=self.tk_img)
            self.img_offset_x = x_offset
            self.img_offset_y = y_offset

            # 마스크가 있으면 결과 오버레이도 다시 표시
            if self.mask is not None:
                self.show_result(update_canvas_only=True)


    def reset_state(self):
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.mask = None

    def reset_mask(self, event=None):
        self.reset_state()
        self.update_canvas()
        self.status.config(text='마스크가 리셋되었습니다.')

    def on_press(self, event):
        self.start_x = event.x - self.img_offset_x
        self.start_y = event.y - self.img_offset_y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, 
            outline='red', width=2
        )

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(
                self.rect, 
                self.start_x + self.img_offset_x, 
                self.start_y + self.img_offset_y,
                event.x, event.y
            )

    def on_release(self, event):
        self.end_x = event.x - self.img_offset_x
        self.end_y = event.y - self.img_offset_y
        if self.rect:
            self.canvas.coords(
                self.rect,
                self.start_x + self.img_offset_x,
                self.start_y + self.img_offset_y,
                self.end_x + self.img_offset_x,
                self.end_y + self.img_offset_y
            )
        self.apply_current_method()

    def apply_current_method(self):
        if None in [self.start_x, self.start_y, self.end_x, self.end_y]:
            return

        # 좌표 정규화
        x1, x2 = sorted([self.start_x, self.end_x])
        y1, y2 = sorted([self.start_y, self.end_y])
        
        # 이미지 경계 체크
        img_h, img_w = self.img.shape[:2]
        x1 = max(0, min(x1, img_w-1))
        x2 = max(0, min(x2, img_w-1))
        y1 = max(0, min(y1, img_h-1))
        y2 = max(0, min(y2, img_h-1))
        
        if x2 <= x1 or y2 <= y1:
            self.status.config(text='유효하지 않은 영역입니다.')
            return

        if self.current_method == "grabcut":
            self.apply_grabcut(x1, y1, x2, y2)
        elif self.current_method == "threshold":
            self.apply_rectangular_segmentation(x1, y1, x2, y2)
        elif self.current_method == "color":
            self.apply_color_segmentation(x1, y1, x2, y2)

    def apply_grabcut(self, x1, y1, x2, y2):
        """[1번 모드] GrabCut 자동 분할 (마스크 덮어쓰기)"""
        try:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            rect = (x1, y1, x2-x1, y2-y1)
            
            mask = np.zeros(img_bgr.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # 더 많은 반복으로 정확도 향상
            cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
            
            # 마스크 후처리
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            
            # 형태학적 연산으로 정제
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            self.mask = mask2 * 255 # 마스크를 덮어씀 (리셋)
            self.show_result()
            self.status.config(text='GrabCut (1) 완료! 마스크가 리셋되었습니다.')
            
        except Exception as e:
            self.status.config(text=f'GrabCut 실패: {str(e)}')

    def apply_rectangular_segmentation(self, x1, y1, x2, y2):
        """[2번 모드] 수동 사각형 마스크 추가 (드래그 영역만 누적)"""
        try:
            # 0. 기존 마스크 복사 (추가 연산을 위해)
            original_mask = self.mask.copy() if self.mask is not None else np.zeros(self.img.shape[:2], dtype=np.uint8)

            # 1. 드래그한 사각형 영역만 마스크로 생성 (자동 확장 로직 제거)
            new_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            cv2.rectangle(new_mask, (x1, y1), (x2, y2), 255, -1)
            
            # 2. 기존 마스크에 새 영역 추가 (OR 연산)
            self.mask = cv2.bitwise_or(original_mask, new_mask)
            
            self.show_result()
            
            add_msg = "(영역 추가됨)" if np.any(original_mask > 0) else ""
            self.status.config(text=f'✅ 수동 마스크 추가 (2) 완료: 드래그한 사각형이 마스크에 누적되었습니다. {add_msg}')
            
        except Exception as e:
            self.status.config(text=f'마스크 추가 실패: {str(e)}')

    def apply_color_segmentation(self, x1, y1, x2, y2):
        """[3번 모드] 색상 기반 분할"""
        if KMeans is None:
            self.status.config(text='색상 분할 실패: sklearn이 설치되어 있어야 합니다.')
            return
        
        try:
            # 1. 선택 영역의 색상 정보 수집
            roi = self.img[y1:y2, x1:x2]
            roi_pixels = roi.reshape(-1, 3)
            
            # 2. 색상 클러스터링으로 주요 색상 찾기
            try:
                kmeans = KMeans(n_clusters=min(3, len(roi_pixels)//100), random_state=42, n_init=10)
                kmeans.fit(roi_pixels)
                centers = kmeans.cluster_centers_
                brightness = np.sum(centers, axis=1)
                rubber_color = centers[np.argmax(brightness)]
            except:
                rubber_color = np.mean(roi_pixels, axis=0)
            
            # 3. 전체 이미지에서 유사 색상 검출
            color_diff = np.linalg.norm(self.img - rubber_color, axis=2)
            
            # 4. 적응적 임계값 (ROI 내 색상 분산 기반)
            roi_diff = np.linalg.norm(roi - rubber_color, axis=2)
            adaptive_threshold = np.mean(roi_diff) + np.std(roi_diff) * 0.8
            
            # 5. 기본 색상 마스크
            color_mask = (color_diff < adaptive_threshold).astype(np.uint8) * 255
            
            # 6. ROI 기반 시드 영역 생성
            roi_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            roi_color_diff = np.linalg.norm(roi - rubber_color, axis=2)
            roi_threshold = np.mean(roi_color_diff) + np.std(roi_color_diff) * 0.5
            roi_mask[y1:y2, x1:x2] = (roi_color_diff < roi_threshold).astype(np.uint8) * 255
            
            # 7. 연결성 기반 확장
            final_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            seed_points = np.where(roi_mask > 0)
            temp_mask = color_mask.copy()
            
            if len(seed_points[0]) > 0:
                step = max(1, len(seed_points[0]) // 15)
                for i in range(0, len(seed_points[0]), step):
                    seed_y, seed_x = seed_points[0][i], seed_points[1][i]
                    if temp_mask[seed_y, seed_x] > 0:
                        cv2.floodFill(temp_mask, None, (seed_x, seed_y), 255, loDiff=0, upDiff=0, flags=4)
                final_mask = temp_mask
            else:
                final_mask = color_mask
            
            # 8. 형태학적 정제 (이전 로직 유지)
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            
            self.mask = final_mask
            self.show_result()
            self.status.config(text=f'색상 기반 분할 (3) 완료!')
            
        except Exception as e:
            self.status.config(text=f'색상 분할 실패: {str(e)}')

    def show_result(self, update_canvas_only=False):
        """결과 미리보기"""
        if self.mask is None:
            return
        
        # 빨간색 오버레이
        overlay = self.img.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = self.mask  # R 채널에 마스크
        
        # 블렌딩
        result = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        
        # 캔버스 업데이트
        img_pil = Image.fromarray(result.astype(np.uint8))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        
        # 이미지가 없으면 기본 이미지를 다시 그립니다.
        if not update_canvas_only:
             img_pil_base = Image.fromarray(self.img)
             self.tk_img_base = ImageTk.PhotoImage(img_pil_base)
             self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor='nw', image=self.tk_img_base)

        # 오버레이 마스크를 기본 이미지 위에 그립니다.
        self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor='nw', image=self.tk_img)


    def save_mask_only(self, event=None):
        """흑백 마스크 저장 + 오버레이 동시 저장"""
        if self.mask is None:
            self.status.config(text='마스크를 먼저 생성하세요!')
            return
        
        # 원본 크기로 리사이즈
        orig_h, orig_w = self.original_img.shape[:2]
        mask_resized = cv2.resize(self.mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 1. 마스크 저장
        mask_save_path = os.path.join(LABEL_DIR, img_files[self.idx])
        cv2.imwrite(mask_save_path, mask_resized)
        
        # 2. 오버레이 폴더 생성 및 오버레이 저장
        overlay_dir = os.path.join(LABEL_DIR, "overlay")
        os.makedirs(overlay_dir, exist_ok=True)
        
        img_bgr = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2BGR)
        color_mask = np.zeros_like(img_bgr)
        color_mask[:, :, 2] = mask_resized  # R 채널에 마스크
        
        overlay = cv2.addWeighted(img_bgr, 0.7, color_mask, 0.3, 0)
        overlay_save_path = os.path.join(overlay_dir, f"overlay_{img_files[self.idx]}")
        cv2.imwrite(overlay_save_path, overlay)
        
        self.status.config(text=f'✅ 저장완료: 마스크 + 오버레이 → overlay/ 폴더')

    def save_overlay(self, event=None):
        """오버레이만 저장 (별도 기능)"""
        if self.mask is None:
            self.status.config(text='마스크를 먼저 생성하세요!')
            return
        
        # 오버레이 폴더 생성
        overlay_dir = os.path.join(LABEL_DIR, "overlay")
        os.makedirs(overlay_dir, exist_ok=True)
        
        # 원본 크기로 처리
        orig_h, orig_w = self.original_img.shape[:2]
        mask_resized = cv2.resize(self.mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        img_bgr = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2BGR)
        color_mask = np.zeros_like(img_bgr)
        color_mask[:, :, 2] = mask_resized  # R 채널
        
        overlay = cv2.addWeighted(img_bgr, 0.7, color_mask, 0.3, 0)
        overlay_save_path = os.path.join(overlay_dir, f"overlay_{img_files[self.idx]}")
        cv2.imwrite(overlay_save_path, overlay)
        self.status.config(text=f'💡 오버레이만 저장: overlay/overlay_{img_files[self.idx]}')

    def next_img(self, event=None):
        self.idx = min(self.idx + 1, len(img_files) - 1)
        self.load_image()

    def prev_img(self, event=None):
        self.idx = max(self.idx - 1, 0)
        self.load_image()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("900x800")
    app = ImprovedRubberLabelGUI(root)
    root.mainloop()