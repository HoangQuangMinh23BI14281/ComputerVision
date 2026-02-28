import argparse
import sys
import os
import cv2
import random
from src.pipeline import OCRPipeline
from src.config import DEFAULT_DET_WEIGHTS, DEFAULT_REC_WEIGHTS

def main():
    parser = argparse.ArgumentParser(description="Pure PyTorch DBNet + SVTRv2 Pipeline Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to input image. If not provided, a random test image will be used.")
    parser.add_argument("--det_model", type=str, default=DEFAULT_DET_WEIGHTS, help="Path to DBNet weights")
    parser.add_argument("--rec_model", type=str, default=DEFAULT_REC_WEIGHTS, help="Path to SVTRv2 weights")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--out", type=str, default="inference_result.jpg", help="Output visualization path")
    args = parser.parse_args()

    # If no image provided, pick randomly from test set
    if args.image is None:
        test_labels = "ocr_dataset/det_test.txt"
        if not os.path.exists(test_labels):
            test_labels = "paddle_dataset/det_val.txt" # fallback
            
        with open(test_labels, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("Error: Test dataset is empty.")
            sys.exit(1)
            
        random_line = random.choice(lines)
        img_rel_path = random_line.strip().split('\t')[0]
        # img_rel_path is like data/Stage1train/img.jpg
        # We assume inference.py is at the root
        args.image = img_rel_path
        print(f"Randomly selected test image: {args.image}")
        
    if not os.path.exists(args.image):
        print(f"Error: Could not find image at {args.image}")
        sys.exit(1)

    pipeline = OCRPipeline(
        use_gpu=args.use_gpu, 
        det_model_path=args.det_model, 
        rec_model_path=args.rec_model
    )
    
    print(f"\nProcessing image: {args.image}...")
    try:
        results, vis_img = pipeline.predict(args.image, visualize=True)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

    print(f"\nFound {len(results)} text regions:")
    print("-" * 50)
    for i, (box, text, conf) in enumerate(results):
        print(f"[{i+1}] {text} (conf: {conf:.4f})")
        
    if vis_img is not None:
        cv2.imwrite(args.out, vis_img)
        print("-" * 50)
        print(f"Saved visualization to: {args.out}")

if __name__ == '__main__':
    main()
