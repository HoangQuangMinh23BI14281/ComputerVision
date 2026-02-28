import os
import json
import torch
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from torch.utils.data import Dataset
import albumentations as A

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class DBNetDataset(Dataset):
    def __init__(self, data_root, label_file, is_train=True, img_size=640, shrink_ratio=0.4, min_text_size=8):
        """
        data_root: thư mục chứa ảnh (vd: data/Stage1train/)
        label_file: file txt chứa định dạng BoxDetection
        """
        self.data_root = data_root
        self.img_size = img_size
        self.shrink_ratio = shrink_ratio
        self.min_text_size = min_text_size
        self.is_train = is_train
        
        self.image_paths = []
        self.polygons_list = []
        self.texts_list = []
        
        # Parse label file
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_path = os.path.join(self.data_root, '..', '..', parts[0])
                    # Fix path since the part[0] is usually data/Stage1train/img.jpg
                    
                    try:
                        polygons_dicts = json.loads(parts[1])
                        # Filter valid polygons
                        polys = []
                        texts = []
                        for pd in polygons_dicts:
                            pts = np.array(pd['points'])
                            if Polygon(pts).area > self.min_text_size:
                                polys.append(pts)
                                texts.append(pd['transcription'])
                                
                        if len(polys) > 0:
                            self.image_paths.append(img_path)
                            self.polygons_list.append(polys)
                            self.texts_list.append(texts)
                    except:
                        pass
        
        if self.is_train:
            self.transforms = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT, fill_value=0)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT, fill_value=0)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.image_paths)

    def generate_shrink_map(self, img_shape, polys):
        h, w = img_shape
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        
        for poly in polys:
            poly = np.array(poly).astype(np.int32)
            polygon = Polygon(poly)
            if polygon.area < self.min_text_size:
                cv2.fillPoly(mask, [poly], 0)
                continue
            
            distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
            subject = [tuple(l) for l in poly]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked = pco.Execute(-distance)
            
            if len(shrinked) == 0:
                cv2.fillPoly(mask, [poly], 0)
                continue
                
            shrink_poly = np.array(shrinked[0]).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt, [shrink_poly], 1)
            
        return gt, mask

    def generate_threshold_map(self, img_shape, polys):
        h, w = img_shape
        thresh_map = np.zeros((h, w), dtype=np.float32)
        thresh_mask = np.zeros((h, w), dtype=np.float32)
        
        for poly in polys:
            poly = np.array(poly).astype(np.int32)
            polygon = Polygon(poly)
            if polygon.area < self.min_text_size:
                continue
                
            distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
            subject = [tuple(l) for l in poly]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            
            expanded = pco.Execute(distance)
            if len(expanded) == 0:
                continue
                
            expanded_poly = np.array(expanded[0]).reshape(-1, 2)
            
            # Simple distance map approximation (usually DBNet uses polygon distances, this is a fast hack)
            cv2.fillPoly(thresh_mask, [expanded_poly.astype(np.int32)], 1)
            cv2.fillPoly(thresh_map, [expanded_poly.astype(np.int32)], 0.7)
            cv2.fillPoly(thresh_map, [poly], 0.3)
            
        return thresh_map, thresh_mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Handle empty image
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        polys = self.polygons_list[idx]
        
        # Prepare for albumentations format
        keypoints = []
        for poly in polys:
            for pt in poly:
                keypoints.append((pt[0], pt[1]))
                
        augmented = self.transforms(image=img, keypoints=keypoints)
        img = augmented['image']
        new_kps = augmented['keypoints']
        
        # reconstruct polys
        new_polys = []
        for i in range(len(polys)):
            poly = []
            for j in range(4): # 4 points per poly
                poly.append(list(new_kps[i*4 + j]))
            new_polys.append(poly)
            
        img_shape = img.shape[:2]
        
        # generate maps
        gt, mask = self.generate_shrink_map(img_shape, new_polys)
        thresh_map, thresh_mask = self.generate_threshold_map(img_shape, new_polys)
        
        # Normalization for model
        img = img.astype(np.float32) / 255.0
        # ImageNet mean / std
        img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img = torch.from_numpy(img).permute(2, 0, 1) # [C, H, W]
        gt = torch.from_numpy(gt)
        mask = torch.from_numpy(mask)
        thresh_map = torch.from_numpy(thresh_map)
        thresh_mask = torch.from_numpy(thresh_mask)
        
        return {
            'image': img,
            'gt': gt,
            'mask': mask,
            'thresh_map': thresh_map,
            'thresh_mask': thresh_mask
        }
