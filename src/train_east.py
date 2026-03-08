from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import json
import torch

from src.dataset.east_dataset import EastDataset, east_collate_fn
from src.models.east.loss import EastLoss
from src.models.east.model import East

import os
import time
import pandas as pd
from src.utils.metrics import calculate_east_metrics
from src.models.east.utils import get_east_boxes

def evaluate_east(model, dataloader, device):
    model.eval()
    all_gt_boxes = []
    all_pred_boxes = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, gt_score_map, gt_geo_map, gt_boxes in dataloader:
            images = images.to(device)
            pred_scores, pred_geos = model(images)
            
            for i in range(images.size(0)):
                s = pred_scores[i].cpu().numpy()
                g = pred_geos[i].cpu().numpy()
                boxes = get_east_boxes(s, g)
                all_pred_boxes.append(boxes)
                all_gt_boxes.append(gt_boxes[i])

    metrics = calculate_east_metrics(all_gt_boxes, all_pred_boxes)
    dur = time.time() - start_time
    metrics['fps'] = len(dataloader.dataset) / dur if dur > 0 else 0
    return metrics

def main():
    TRAIN_TXT = 'ocr_dataset/det_train.txt'
    VAL_TXT = 'ocr_dataset/det_val.txt'
    TEST_TXT = 'ocr_dataset/det_test.txt'
    WEIGHTS_DIR = 'weights/east'
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    LOG_PATH = os.path.join(WEIGHTS_DIR, 'log.csv')
    
    BATCH_SIZE = 12
    EPOCHS = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Splits
    X_train, y_train = EastDataset.load_from_txt(TRAIN_TXT)
    X_val, y_val = EastDataset.load_from_txt(VAL_TXT)
    X_test, _ = EastDataset.load_from_txt(TEST_TXT) # Only paths for test for now

    train_dataset = EastDataset(X_train, y_train)
    val_dataset = EastDataset(X_val, y_val)

    # Optimized DataLoader
    num_workers = 4 if os.name != 'nt' else 0 # num_workers > 0 can be tricky on Windows/PowerShell
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=east_collate_fn, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=east_collate_fn, num_workers=num_workers)

    model = East(pretrained=True).to(device)
    if os.path.exists(os.path.join(WEIGHTS_DIR, 'best.pth')):
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'best.pth'), map_location=device))
    
    optimizer = Adam(model.parameters(), lr=5e-5)
    loss_fn = EastLoss().to(device)
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    history = []
    best_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, gt_score, gt_geo, _ in pbar:
            images, gt_score, gt_geo = images.to(device), gt_score.to(device), gt_geo.to(device)
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_score, pred_geo = model(images)
                loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
            
            # Scaled Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Safe margin)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Evaluation
        val_metrics = evaluate_east(model, val_loader, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  FPS: {val_metrics['fps']:.1f}")
        
        log_entry = {
            'epoch': epoch+1, 
            'train_loss': avg_train_loss, 
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1': val_metrics['f1'],
            'iou': val_metrics['iou'],
            'fps': val_metrics['fps']
        }
        history.append(log_entry)
        pd.DataFrame(history).to_csv(LOG_PATH, index=False)
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'best.pth'))
            print("  New Best Model Saved!")
            
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'latest.pth'))

if __name__ == '__main__':
    main()
