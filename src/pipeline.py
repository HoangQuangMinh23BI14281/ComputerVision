import cv2
import torch
import numpy as np
import os
import math
from src.models.dbnet.model import DBNet
from src.models.svtr.model import SVTRv2
from src.models.svtr.vocab import Vocab
from src.config import REC_CHAR_SET, REC_IMAGE_SHAPE, REC_MEAN, REC_STD, REC_MIN_CROP_WIDTH

class OCRPipeline:
    def __init__(self, use_gpu=True, det_model_path=None, rec_model_path=None):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Initializing Pure PyTorch Pipeline on device: {self.device}")

        # Initialize DBNet (Detection)
        self.det_model = DBNet(pretrained=False).to(self.device)
        if det_model_path and os.path.exists(det_model_path):
            self.det_model.load_state_dict(torch.load(det_model_path, map_location=self.device), strict=False)
            print(f"Loaded DET model from: {det_model_path}")
        else:
            print("WARNING: DET model path not provided or not found. Output will be random.")
        self.det_model.eval()

        # Initialize SVTRv2 (Recognition)
        self.char_set = REC_CHAR_SET
        self.vocab = Vocab(self.char_set)
        
        self.rec_model = SVTRv2(imgH=32, nc=3, nclass=self.vocab.num_classes).to(self.device)
        if rec_model_path and os.path.exists(rec_model_path):
            self.rec_model.load_state_dict(torch.load(rec_model_path, map_location=self.device), strict=False)
            print(f"Loaded REC model from: {rec_model_path}")
        else:
            print("WARNING: REC model path not provided or not found. Output will be random.")
        self.rec_model.eval()

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process_det_output(self, prob_map, scale, original_shape, padded_shape, thresh=0.3, box_thresh=0.3, unclip_ratio=1.5):
        import pyclipper
        from shapely.geometry import Polygon
        
        h, w = padded_shape
        new_h, new_w = int(original_shape[0] * scale), int(original_shape[1] * scale)
        prob_map = prob_map[:new_h, :new_w]
        prob_map = cv2.resize(prob_map, (original_shape[1], original_shape[0]))
        
        segmentation = prob_map > thresh
        contours, _ = cv2.findContours((segmentation * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            
            # calculate box size
            w_b, h_b = rect[1]
            if min(w_b, h_b) < 3:
                continue
                
            mask = np.zeros_like(prob_map, dtype=np.uint8)
            cv2.fillPoly(mask, [box.astype(np.int32)], 1)
            mean_score = prob_map[mask == 1].mean() if mask.sum() > 0 else 0
            
            if mean_score > box_thresh:
                # Unclip (Vatti expanding)
                poly = Polygon(box)
                if poly.length == 0: continue
                distance = poly.area * unclip_ratio / poly.length
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(box.astype(np.int32).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                expanded = offset.Execute(distance)
                
                if len(expanded) == 0:
                    continue
                expanded = np.array(expanded[0])
                
                rect_exp = cv2.minAreaRect(expanded)
                box_exp = cv2.boxPoints(rect_exp)
                box_exp = self.order_points_clockwise(box_exp)
                boxes.append(box_exp)
                
        return boxes

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
        
        if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
            dst = np.rot90(dst)
            
        return dst

    def decode_svtr(self, preds_idx, preds_prob, characters):
        # CTC Decoding (Blank index is assumed to be 0)
        results = []
        for i in range(preds_idx.shape[0]):
            char_list = []
            conf_list = []
            for j in range(preds_idx.shape[1]):
                if preds_idx[i, j] != 0 and (not (j > 0 and preds_idx[i, j - 1] == preds_idx[i, j])):
                    char_idx = preds_idx[i, j].item() - 1 
                    if 0 <= char_idx < len(characters):
                        char_list.append(characters[char_idx])
                        conf_list.append(preds_prob[i, j].item())
            res = ''.join(char_list)
            conf = sum(conf_list) / max(len(conf_list), 1) if len(conf_list) > 0 else 0.0
            results.append((res, conf))
        return results

    def predict(self, image_path, visualize=False):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Image not found at {image_path}")
            
        # --- 1. DETECTION ---
        target_size = (640, 640)
        h, w = img_bgr.shape[:2]
        scale = min(target_size[0]/h, target_size[1]/w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized_img = cv2.resize(img_bgr.copy(), (new_w, new_h))
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        padded_img = cv2.copyMakeBorder(resized_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        tensor_img = rgb_img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor_img = (tensor_img - mean) / std
        tensor_img = tensor_img.transpose(2, 0, 1)
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            preds = self.det_model(tensor_img)
            prob_map = preds[0, 0].cpu().numpy()
            
        boxes = self.process_det_output(prob_map, scale, (h, w), target_size)
        results = []
        
        for original_box in boxes:
            # --- 2. RECOGNITION ---
            crop = self.extract_crop(img_bgr, original_box)
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            c_h, c_w = crop_rgb.shape[:2]
            
            # CRITICAL FIX: Match the exact aspect ratio resize logic from recognition_dataset.py
            t_h = REC_IMAGE_SHAPE[1]
            t_w = REC_IMAGE_SHAPE[2]
            
            # Calculate aspect ratio preserving width
            new_cw = int(c_w * (t_h / c_h))
            new_cw = max(min(new_cw, t_w), REC_MIN_CROP_WIDTH) # Add bounds
            
            crop_resized = cv2.resize(crop_rgb, (new_cw, t_h))
            
            # Pad width to t_w if smaller
            crop_padded = np.zeros((t_h, t_w, 3), dtype=np.uint8)
            crop_padded[:, :new_cw, :] = crop_resized
            
            crop_tensor = crop_padded.astype(np.float32) / 255.0
            
            # CRITICAL FIX: Match the Normalization values used in training (recognition_dataset.py)
            mean_c = np.array(REC_MEAN)
            std_c = np.array(REC_STD)
            
            crop_tensor -= mean_c
            crop_tensor /= std_c
            
            crop_tensor = crop_tensor.transpose(2, 0, 1)
            crop_tensor = torch.from_numpy(crop_tensor).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                rec_preds = self.rec_model(crop_tensor) 
            
            preds_transposed = rec_preds.transpose(0, 1) # [B, T, C]
            preds_idx = preds_transposed.argmax(axis=2)
            preds_prob = preds_transposed.softmax(dim=2).max(axis=2)[0]
            
            decoded_results = self.decode_svtr(preds_idx, preds_prob, self.vocab.character)
            text, conf = decoded_results[0]
            
            if conf > 0.05 and len(text) > 0:
                results.append((original_box, text, conf))

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
