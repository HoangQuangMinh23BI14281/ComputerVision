import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

# EAST imports
from src.models.east.model import East
from src.models.east.utils import get_east_boxes, resize
from src.config import DET_SCORE_THRESH, DET_NMS_THRESH

# CRNN imports
from src.models.crnn.model import CRNN
from src.dataset.crnn_dataset import ResizeNormalize
from src.predict_crnn import predict as predict_crnn

def order_points(pts):
    """
    Chuẩn hóa thứ tự 4 điểm: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    Tránh lỗi cắt ảnh bị lộn ngược khiến CRNN không đọc được.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left có tổng x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)] # Bottom-Right có tổng x+y lớn nhất
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right có hiệu y-x nhỏ nhất
    rect[3] = pts[np.argmax(diff)] # Bottom-Left có hiệu y-x lớn nhất
    
    return rect

def get_rotate_crop(image, box, padding=0.05):
    """
    Cắt ảnh và xoay phẳng (Rectify) với padding nhẹ để CRNN đọc chính xác hơn.
    box: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    pts = box.reshape(4, 2).astype(np.float32)
    pts = order_points(pts)
    
    (tl, tr, br, bl) = pts
    
    # Tính toán chiều dài và chiều rộng
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Thêm padding để không bị lẹm viền chữ
    pad_x = int(maxWidth * padding)
    pad_y = int(maxHeight * padding)
    
    dst_pts = np.array([
        [pad_x, pad_y],
        [maxWidth + pad_x - 1, pad_y],
        [maxWidth + pad_x - 1, maxHeight + pad_y - 1],
        [pad_x, maxHeight + pad_y - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (maxWidth + 2*pad_x, maxHeight + 2*pad_y))
    
    return warped

def sort_reading_order(boxes, y_tolerance=10):
    """
    Sắp xếp các bounding box theo thứ tự đọc: 
    Từ trên xuống dưới, cùng 1 dòng thì từ trái qua phải.
    """
    if len(boxes) == 0:
        return boxes
        
    # Tính tâm y (center_y) và min x của từng box
    centers = []
    for i, box in enumerate(boxes):
        pts = box[:8].reshape(4, 2)
        center_y = np.mean(pts[:, 1])
        min_x = np.min(pts[:, 0])
        centers.append({'index': i, 'cy': center_y, 'mx': min_x})
        
    # Sắp xếp sơ bộ theo trục Y
    centers = sorted(centers, key=lambda k: k['cy'])
    
    sorted_indices = []
    current_line = [centers[0]]
    
    # Gom nhóm các box trên cùng 1 dòng và sắp xếp theo trục X
    for i in range(1, len(centers)):
        if abs(centers[i]['cy'] - current_line[-1]['cy']) < y_tolerance:
            current_line.append(centers[i])
        else:
            current_line = sorted(current_line, key=lambda k: k['mx'])
            sorted_indices.extend([item['index'] for item in current_line])
            current_line = [centers[i]]
            
    current_line = sorted(current_line, key=lambda k: k['mx'])
    sorted_indices.extend([item['index'] for item in current_line])
    
    return boxes[sorted_indices]

def predict_east(model, image_pil, device):
    orig_w, orig_h = image_pil.size
    img_resized, _ = resize(image_pil, np.zeros((0, 8)), 512)
    new_w, new_h = img_resized.size
    
    rat_w = new_w / orig_w
    rat_h = new_h / orig_h
    
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        score_map, geo_map = model(img_tensor)
        
    boxes = get_east_boxes(score_map, geo_map, score_thresh=DET_SCORE_THRESH, nms_thresh=DET_NMS_THRESH)
    
    if boxes is not None and len(boxes) > 0:
        boxes[:, [0, 2, 4, 6]] /= rat_w
        boxes[:, [1, 3, 5, 7]] /= rat_h
        
        # Sắp xếp lại thứ tự đọc trước khi trả về
        boxes = sort_reading_order(boxes)
        return boxes
    return []

def draw_ocr_results(image_cv, results):
    """
    results: list of dicts {'box': [8], 'text': str}
    """
    img_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_path = "C:/Windows/Fonts/arial.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 18)
    except:
        font = ImageFont.load_default()

    for res in results:
        box = res['box'].reshape(4, 2).astype(np.int32)
        text = res['text']
        
        draw.polygon([tuple(p) for p in box], outline="green", width=2)
        
        # Căn chỉnh text nhích lên trên một chút để không đè vào viền
        pts = order_points(box.astype(np.float32))
        pos = (int(pts[0][0]), int(pts[0][1]) - 20)
        
        try:
            bbox = draw.textbbox(pos, text, font=font)
            draw.rectangle(bbox, fill="green")
        except:
            pass
            
        draw.text(pos, text, font=font, fill="white")

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Integrated EAST + CRNN OCR Pipeline")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--east_weights', type=str, default='weights/east/best.pth', help='Path to EAST weights')
    parser.add_argument('--crnn_weights', type=str, default='weights/crnn/best.pth', help='Path to CRNN weights')
    parser.add_argument('--output', type=str, default='pipeline_result.png', help='Path to save result image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading EAST model...")
    east_model = East().to(device)
    if os.path.exists(args.east_weights):
        east_model.load_state_dict(torch.load(args.east_weights, map_location=device, weights_only=True))
    else:
        print(f"Error: EAST weights not found at {args.east_weights}")
        return

    print("Loading CRNN model...")
    crnn_model = CRNN().to(device)
    if os.path.exists(args.crnn_weights):
        crnn_model.load_state_dict(torch.load(args.crnn_weights, map_location=device, weights_only=True))
    else:
        print(f"Error: CRNN weights not found at {args.crnn_weights}")
        return
    crnn_model.eval()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    img_pil = Image.open(args.image).convert('RGB')
    img_cv = cv2.imread(args.image)

    print("Detecting text regions...")
    boxes = predict_east(east_model, img_pil, device)

    if len(boxes) == 0:
        print("No text detected.")
        cv2.imwrite(args.output, img_cv)
        return

    print(f"Recognizing {len(boxes)} regions...")
    results = []
    
    # In ra một đoạn phân cách cho đẹp
    print("-" * 40)
    print("KẾT QUẢ ĐỌC CHỮ (READING ORDER):")
    
    for i, box in enumerate(boxes):
        crop = get_rotate_crop(img_cv, box[:8])
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        text = predict_crnn(crop_pil, crnn_model, device)
        # Kết quả in ra terminal giờ đây đã chuẩn như đọc văn bản
        print(f"Line {i+1}: {text}") 
        
        results.append({
            'box': box[:8],
            'text': text
        })
    print("-" * 40)

    print("Visualizing results...")
    final_img = draw_ocr_results(img_cv, results)
    
    cv2.imwrite(args.output, final_img)
    print(f"Pipeline finished. Result saved to {args.output}")

if __name__ == "__main__":
    main()