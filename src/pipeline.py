import cv2
import torch
import numpy as np
import os
import math
from PIL import Image
from src.models.east.model import East
from src.models.crnn.model import CRNN
from src.models.crnn.utils import decode as crnn_decode
from src.dataset.crnn_dataset import ResizeNormalize
from src.config import REC_CHAR_SET, REC_IMAGE_SHAPE

class OCRPipeline:
    def __init__(self, use_gpu=True, det_model_path=None, rec_model_path=None):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Initializing OCR Pipeline (EAST + CRNN) on device: {self.device}")

        # Initialize EAST (Detection)
        self.det_model = East(pretrained=True).to(self.device)
        if det_model_path and os.path.exists(det_model_path):
            self.det_model.load_state_dict(torch.load(det_model_path, map_location=self.device))
            print(f"Loaded EAST model from: {det_model_path}")
        self.det_model.eval()

        # Initialize CRNN (Recognition)
        self.rec_model = CRNN().to(self.device)
        if rec_model_path and os.path.exists(rec_model_path):
            self.rec_model.load_state_dict(torch.load(rec_model_path, map_location=self.device))
            print(f"Loaded CRNN model from: {rec_model_path}")
        self.rec_model.eval()
        
        self.crnn_resize_norm = ResizeNormalize()

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract_crop(self, img, points):
        pts = np.float32(points)
        img_crop_width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        img_crop_height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        
        if img_crop_width == 0 or img_crop_height == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        pts_std = np.float32([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ])
        
        M = cv2.getPerspectiveTransform(pts, pts_std)
        dst = cv2.warpPerspective(
            img, M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        return dst

    def get_east_boxes(self, score, geo, score_thresh=0.9, nms_thresh=0.2):
        # We try to use lanms if available, otherwise fallback to simple NMS
        try:
            import lanms
        except ImportError:
            print("WARNING: lanms not found. Bounding boxes might be redundant.")
            return None

        def is_valid_poly(res, score_shape, scale):
            cnt = 0
            for i in range(res.shape[1]):
                if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                        res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
                    cnt += 1
            return True if cnt <= 1 else False

        def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
            polys = []
            index = []
            valid_pos *= scale
            d = valid_geo[:4, :]
            for i in range(valid_pos.shape[0]):
                x = valid_pos[i, 0]
                y = valid_pos[i, 1]
                y_min = y - d[0, i] * 1.3
                y_max = y + d[1, i] * 1.3
                x_min = x - d[2, i] * 1.1
                x_max = x + d[3, i] * 1.1
                coord = np.array([[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]])
                if is_valid_poly(coord, score_shape, scale):
                    index.append(i)
                    polys.append([coord[0, 0], coord[1, 0], coord[0, 1], coord[1, 1],
                                 coord[0, 2], coord[1, 2], coord[0, 3], coord[1, 3]])
            return np.array(polys), index

        score = score[0, :, :]
        xy_text = np.argwhere(score > score_thresh)
        if xy_text.size == 0: return None
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        valid_pos = xy_text[:, ::-1].copy()
        valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
        polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
        if polys_restored.size == 0: return None
        boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = polys_restored
        boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
        return boxes

    def predict(self, image_path, visualize=False):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: raise ValueError(f"Image not found at {image_path}")
        
        # --- 1. DETECTION (EAST) ---
        h, w = img_bgr.shape[:2]
        rh, rw = h, w
        # Resize to multiples of 32
        rh = rh if rh % 32 == 0 else int(rh / 32) * 32
        rw = rw if rw % 32 == 0 else int(rw / 32) * 32
        img_resized = cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_BILINEAR)
        ratio_h, ratio_w = rh / h, rw / w
        
        # Preprocess
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor_img = img_rgb.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        tensor_img = tensor_img.transpose(2, 0, 1)
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            score, geo = self.det_model(tensor_img)
        
        east_boxes = self.get_east_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
        
        results = []
        if east_boxes is not None:
            # Adjust scales
            east_boxes[:, [0, 2, 4, 6]] /= ratio_w
            east_boxes[:, [1, 3, 5, 7]] /= ratio_h
            
            for box in east_boxes:
                poly = box[:8].reshape(4, 2)
                # --- 2. RECOGNITION (CRNN) ---
                crop = self.extract_crop(img_bgr, poly)
                if crop.size == 0: continue
                
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                crop_tensor = self.crnn_resize_norm(crop_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits = self.rec_model(crop_tensor)
                
                # Decoding
                logits = logits.permute(1, 0, 2) # [B, T, C]
                probs = torch.nn.functional.softmax(logits, dim=2)
                preds = torch.argmax(probs, dim=2).cpu().numpy()[0]
                
                decoded = []
                prev = None
                for p in preds:
                    if p != 0 and p != prev:
                        decoded.append(p)
                    prev = p
                
                try:
                    text = crnn_decode(decoded)
                except Exception:
                    text = "???" # Fallback if decode fails (e.g. missing vocab.json)
                
                results.append((poly, text, box[8]))

        # Visualization
        vis_img = None
        if visualize:
            vis_img = img_bgr.copy()
            for box, text, _ in results:
                box_int = box.astype(np.int32)
                cv2.polylines(vis_img, [box_int], True, (0, 255, 0), 2)
                cv2.putText(vis_img, text, (box_int[0][0], max(0, box_int[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return results, vis_img
