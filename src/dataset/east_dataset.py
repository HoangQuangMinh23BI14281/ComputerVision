from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os
import torch

from src.models.east.utils import rotate_img, get_score_geo, resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EastDataset(Dataset):
    def __init__(self, image_paths, boxes, scale=0.25, length=512):
        super(EastDataset, self).__init__()
        self.image_paths = image_paths
        self.boxes = boxes
        self.scale = scale
        self.length = length
        self.tranforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, item):
        vertices = np.array(self.boxes[item], dtype=np.float32)

        image = Image.open(self.image_paths[item])
        image = image.convert('RGB')
        # image, vertices = rotate_img(image, vertices)
        image, vertices_resized = resize(image, vertices.reshape(-1, 8), self.length)

        score_map, geo_map = get_score_geo(image, vertices_resized, self.scale, self.length)
        image = self.tranforms(image)

        return image, score_map, geo_map, vertices_resized

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def load_from_txt(txt_path):
        """Loads image paths and boxes from a tab-separated text file."""
        image_paths = []
        boxes = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_paths.append(parts[0])
                    # Parse JSON boxes
                    try:
                        data = json.loads(parts[1])
                        img_boxes = []
                        for item in data:
                            # Flatten points [[x1,y1], [x2,y2], ...] to [x1,y1,x2,y2,...]
                            pts = np.array(item['points']).flatten()
                            img_boxes.append(pts)
                        boxes.append(img_boxes)
                    except Exception:
                        continue
        return image_paths, boxes


def east_collate_fn(batch):
    images = []
    score_maps = []
    geo_maps = []
    vertices = []
    for b in batch:
        images.append(b[0])
        score_maps.append(b[1])
        geo_maps.append(b[2])
        vertices.append(b[3])
    
    return torch.stack(images, 0), torch.stack(score_maps, 0), torch.stack(geo_maps, 0), vertices

if __name__ == '__main__':
    print("EastDataset class defined")

