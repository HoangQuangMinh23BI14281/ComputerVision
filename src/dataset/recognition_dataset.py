import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class CRNNDataset(Dataset):
    def __init__(self, data_root, label_file, vocab, is_train=True, imgH=32, imgW=100):
        self.data_root = data_root
        self.vocab = vocab
        self.imgH = imgH
        self.imgW = imgW
        self.is_train = is_train
        
        self.image_paths = []
        self.texts = []
        
        # Parse label file
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_path = os.path.join(self.data_root, '..', '..', parts[0])
                    # Ignore images that do not exist
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.texts.append(parts[1])
        
        if self.is_train:
            self.transforms = A.Compose([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.Blur(blur_limit=3, p=0.3),
                A.GaussNoise(p=0.3),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3)
            ])
        else:
            self.transforms = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        text = self.texts[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # Resize to (imgW, imgH) or keep aspect ratio and pad
        h, w = img.shape[:2]
        
        # Calculate aspect ratio preserving width
        new_w = int(w * (self.imgH / h))
        new_w = max(min(new_w, self.imgW), 8) # Add bounds, at least 8 to survive downsampling
        
        img = cv2.resize(img, (new_w, self.imgH))
        
        # Pad width to imgW if smaller
        pad_img = np.zeros((self.imgH, self.imgW, 3), dtype=np.uint8)
        pad_img[:, :new_w, :] = img
        
        # Normalize
        pad_img = pad_img.astype(np.float32) / 255.0
        pad_img -= np.array([0.485, 0.456, 0.406])
        pad_img /= np.array([0.229, 0.224, 0.225])
        
        tensor_img = torch.from_numpy(pad_img).permute(2, 0, 1) # [C, H, W]
        
        # Encode text
        target, _ = self.vocab.encode([text])
        if len(target) == 0:
            # Handle empty target by retrying
            return self.__getitem__((idx + 1) % len(self))
            
        return {
            'image': tensor_img,
            'target': target.squeeze(0) if target.dim() > 1 else target,
            'target_length': len(target)
        }

def collate_crnn(batch):
    images = []
    targets = []
    target_lengths = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
        target_lengths.append(item['target_length'])
        
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return {
        'images': images,
        'targets': targets,
        'target_lengths': target_lengths
    }
