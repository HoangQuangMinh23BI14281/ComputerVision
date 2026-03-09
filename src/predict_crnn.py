import torch
from PIL import Image, ImageOps
from scipy.special import logsumexp
import math
import numpy as np


from src.models.crnn.utils import decode
from src.dataset.crnn_dataset import ResizeNormalize
from src.models.crnn.model import CRNN


NINF = -1 * float('inf')
DEFAULT_THRESHOLD = 0.01


def _reconstruct(labels, blank=0):
    new_labels = list()

    previous = None

    for char in labels:
        if char != previous:
            new_labels.append(char)
            previous = char

    new_labels = [char for char in new_labels if char != blank]

    return new_labels


def greedy_decode(log_probs, blank=0):
    labels = np.argmax(log_probs, axis=-1)
    labels = _reconstruct(labels, blank)

    labels = decode(labels)

    return labels


def beam_search_decode(log_probs, beam_size, threshold=math.log(DEFAULT_THRESHOLD), blank=0):
    length = log_probs.shape[0]
    n_classes = log_probs.shape[-1]

    beams = [([], 0)]

    for t in range(length):
        new_beams = list()
        for prefix, accumulate_log_prob in beams:
            for c in range(n_classes):
                log_prob = log_probs[t, c]
                if log_prob < threshold:
                    continue

                new_prefix = prefix + [c]
                new_accumulate_log_prob = accumulate_log_prob + log_prob
                new_beams.append((new_prefix, new_accumulate_log_prob))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[: beam_size]

    total_accu_log_prob = {}
    for prefix, accumulate_prob in beams:
        labels = tuple(_reconstruct(prefix))
        total_accu_log_prob[labels] = logsumexp([accumulate_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accumulate_prob) for labels, accumulate_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    labels = decode(labels)

    return labels


import os
import argparse

def predict(image_input, model, device):
    """
    Predicts labels for a single image.
    image_input: can be a file path (str) or a PIL Image object.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = image_input
        
    image = ImageOps.grayscale(image)

    normalize = ResizeNormalize()
    image = normalize(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image) # (T, N, C)
        # Reshape for softmax
        # model returns (T, N, C), beam_search_decode expects (T, C) for single batch
        logits = logits.squeeze(1) # (T, C)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.cpu().numpy()

    pred_label = beam_search_decode(log_probs, beam_size=5)

    return pred_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict text using CRNN with Beam Search')
    parser.add_argument('--image', type=str, default='ocr_dataset/crops/crop_0.jpg', help='Path to image crop')
    parser.add_argument('--weights', type=str, default='weights/crnn/best.pth', help='Path to model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CRNN().to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"WARNING: {args.weights} not found. Using random weights.")
        
    model.eval()

    if os.path.exists(args.image):
        pred_label = predict(args.image, model, device)
        print("\n" + "="*30)
        print(f"Image: {args.image}")
        print(f"Result (Beam Search): '{pred_label}'")
        print("="*30)
    else:
        print(f"Image not found: {args.image}")
