import torch
import cv2
import numpy as np
from PIL import Image
import os
import argparse

from src.models.east.model import East
from src.models.east.utils import get_east_boxes, resize
from src.config import DET_SCORE_THRESH, DET_NMS_THRESH

def predict_east(model, image, device, score_thresh=DET_SCORE_THRESH, nms_thresh=DET_NMS_THRESH):
    """
    Predicts bounding boxes for an image using the EAST model.
    Args:
        model: Loaded EAST model.
        image: PIL Image or image path.
        device: CPU or GPU device.
        score_thresh: Threshold for score map.
        nms_thresh: Threshold for NMS.
    Returns:
        boxes: Detected bounding boxes [N, 9] (x1, y1, x2, y2, x3, y3, x4, y4, score).
    """
    if isinstance(image, str):
        img_pil = Image.open(image).convert('RGB')
    else:
        img_pil = image.convert('RGB')
        
    orig_w, orig_h = img_pil.size
    
    # Resize for inference (multiples of 32)
    # We use the same 'resize' logic as in visualize/train for consistency
    img_resized, _ = resize(img_pil, np.zeros((0, 8)), 512)
    new_w, new_h = img_resized.size
    
    # Calculate ratios
    rat_w = new_w / orig_w
    rat_h = new_h / orig_h
    
    # To Tensor
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        score_map, geo_map = model(img_tensor)
        
    # Get boxes on resized image
    boxes = get_east_boxes(score_map, geo_map, score_thresh=score_thresh, nms_thresh=nms_thresh)
    
    if boxes is not None and len(boxes) > 0:
        # Map boxes back to original size
        boxes[:, [0, 2, 4, 6]] /= rat_w
        boxes[:, [1, 3, 5, 7]] /= rat_h
        return boxes
    
    return np.array([])

def main():
    parser = argparse.ArgumentParser(description="EAST Prediction Script")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, default='weights/east/best.pth', help='Path to model weights')
    parser.add_argument('--output', type=str, default='res_east.png', help='Path to save result image')
    parser.add_argument('--thresh', type=float, default=DET_SCORE_THRESH, help='Detection threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = East().to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Error: Weights {args.weights} not found.")
        return

    # Predict
    boxes = predict_east(model, args.image, device, score_thresh=args.thresh)

    # Visualize results
    img_cv = cv2.imread(args.image)
    if boxes is not None and len(boxes) > 0:
        print(f"Detected {len(boxes)} text regions.")
        for box in boxes:
            pts = box[:8].reshape(4, 2).astype(np.int32)
            cv2.polylines(img_cv, [pts], True, (0, 255, 0), 2)
    else:
        print("No text detected.")

    cv2.imwrite(args.output, img_cv)
    print(f"Saved visualization to {args.output}")

if __name__ == '__main__':
    main()
