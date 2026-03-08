import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
from PIL import Image, ImageOps
import numpy as np
import os


from src.models.crnn.utils import encode

class CrnnDataset(Dataset):
    def __init__(self, image_paths, texts, width=280, height=64):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = ResizeNormalize(width, height)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path)
        image = ImageOps.grayscale(image)
        text = self.texts[item]
        
        image = self.transform(image)
        text_encode, _ = encode(text)
        
        return image, torch.tensor(text_encode), text

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def load_from_txt(txt_path):
        """Loads image paths and texts from a tab-separated text file."""
        image_paths = []
        texts = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_paths.append(parts[0])
                    texts.append(parts[1])
        return image_paths, texts


class ResizeNormalize(object):
    def __init__(self, width=280, height=64):
        self.scale_width = width
        self.scale_height = height
        self.transforms = transforms.Compose([
            ToTensor(),
            Normalize(mean=0.5,
                      std=0.5)
        ])

    def __call__(self, image):
        w, h = image.size
        new_height = self.scale_height
        new_width = w * (new_height / h)
        new_width = int(new_width)

        if new_width >= self.scale_width:
            image = image.resize((self.scale_width, self.scale_height))
        else:
            image = image.resize((new_width, new_height))
            image_pad = np.zeros((self.scale_height, self.scale_width))
            image_pad[: new_height, : new_width] = np.array(image)
            image = Image.fromarray(np.uint8(image_pad))

        image = self.transforms(image)
        return image


def collate_fn(batch):
    images = []
    text_encodes = []
    text_lens = []
    raw_texts = []
    for b in batch:
        images.append(b[0])
        text_encodes.append(b[1])
        text_lens.append(len(b[1]))
        raw_texts.append(b[2])

    # Pad text_encodes for batching if needed, but CTCLoss usually wants a flat target
    # with lengths. However, for easier batching, let's keep them as a list or cat them.
    flat_targets = torch.cat(text_encodes)
    
    return torch.stack(images, dim=0), flat_targets, torch.tensor(text_lens), raw_texts


if __name__ == '__main__':
    print("CrnnDataset class defined")
