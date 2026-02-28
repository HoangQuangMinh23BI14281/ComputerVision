import os
import json
import cv2
import glob
from tqdm import tqdm

def process_sroie():
    # Paths from the user's project structure
    train_img_dir = "data/Stage1train"
    train_txt_dir = "data/Stage1train"
    
    test_img_dir = "data/Stage1and2test(picture)"
    test_txt_dir = "data/Stage1and2test(text)"
    
    out_dir = "ocr_dataset"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)

    det_train_txt = os.path.join(out_dir, "det_train.txt")
    rec_train_txt = os.path.join(out_dir, "rec_train.txt")
    
    det_test_txt = os.path.join(out_dir, "det_test.txt")
    rec_test_txt = os.path.join(out_dir, "rec_test.txt")

    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")))
    test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    
    print(f"Found {len(train_imgs)} train images.")
    print(f"Found {len(test_imgs)} test images.")

    global_crop_id = 0

    def process_split(split_imgs, txt_dir, img_dir_rel, det_txt, rec_txt):
        nonlocal global_crop_id
        with open(det_txt, 'w', encoding='utf-8') as f_det, \
             open(rec_txt, 'w', encoding='utf-8') as f_rec:
             
            desc = f"{os.path.basename(det_txt)} & {os.path.basename(rec_txt)}"
            for img_path in tqdm(split_imgs, desc=desc):
                img_id = os.path.basename(img_path).replace('.jpg', '')
                txt_path = os.path.join(txt_dir, f"{img_id}.txt")
                
                if not os.path.exists(txt_path):
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                polygons = []
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f_txt:
                    lines = f_txt.readlines()
                    
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 9:
                        # 8 coordinates
                        coords = [int(x) for x in parts[:8]]
                        text = ','.join(parts[8:])
                        
                        points = [
                            [coords[0], coords[1]],
                            [coords[2], coords[3]],
                            [coords[4], coords[5]],
                            [coords[6], coords[7]]
                        ]
                        
                        polygons.append({
                            "transcription": text,
                            "points": points
                        })
                        
                        # --- CROP PROCESSING FOR CRNN/SVTR ---
                        try:
                            import numpy as np
                            pts = np.array(points, dtype=np.float32)
                            img_crop_width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                            img_crop_height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                            
                            pts_std = np.float32([
                                [0, 0],
                                [img_crop_width, 0],
                                [img_crop_width, img_crop_height],
                                [0, img_crop_height]
                            ])
                            
                            M = cv2.getPerspectiveTransform(pts, pts_std)
                            crop_img = cv2.warpPerspective(
                                img, M, (img_crop_width, img_crop_height),
                                borderMode=cv2.BORDER_REPLICATE,
                                flags=cv2.INTER_CUBIC
                            )
                            
                            if crop_img.shape[0] * 1.0 / crop_img.shape[1] >= 1.5:
                                crop_img = np.rot90(crop_img)

                            if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                                crop_filename = f"crop_{global_crop_id}.jpg"
                                crop_path = os.path.join(out_dir, "crops", crop_filename)
                                cv2.imwrite(crop_path, crop_img)
                                
                                # Relative path for dataloader
                                rel_crop_path = f"{out_dir}/crops/{crop_filename}"
                                f_rec.write(f"{rel_crop_path}\t{text}\n")
                                global_crop_id += 1
                        except Exception as e:
                            pass
                
                # --- DBNet ANNOTATION ---
                # We save relative path of original image
                rel_img_path = f"{img_dir_rel}/{img_id}.jpg"
                f_det.write(f"{rel_img_path}\t{json.dumps(polygons, ensure_ascii=False)}\n")

    print("Generating Train Splits...")
    process_split(train_imgs, train_txt_dir, "data/Stage1train", det_train_txt, rec_train_txt)
    print("Generating Test Splits...")
    process_split(test_imgs, test_txt_dir, "data/Stage1and2test(picture)", det_test_txt, rec_test_txt)

    print("PyTorch Dataset preparation complete!")

if __name__ == '__main__':
    process_sroie()
