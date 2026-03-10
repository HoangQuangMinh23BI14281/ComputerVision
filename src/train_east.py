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
from src.config import DET_SCORE_THRESH, DET_NMS_THRESH

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
                # Use centralized threshold from config.py
                boxes = get_east_boxes(s, g, score_thresh=DET_SCORE_THRESH, nms_thresh=DET_NMS_THRESH)
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
    EPOCHS = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Splits
    X_train, y_train = EastDataset.load_from_txt(TRAIN_TXT)
    X_val, y_val = EastDataset.load_from_txt(VAL_TXT)
    X_test, _ = EastDataset.load_from_txt(TEST_TXT) # Only paths for test for now

    train_dataset = EastDataset(X_train, y_train)
    val_dataset = EastDataset(X_val, y_val)

    # Optimized DataLoader
    num_workers = 4 if os.name != 'nt' else 0 
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

    model = East(weights='DEFAULT').to(device)
    if os.path.exists(os.path.join(WEIGHTS_DIR, 'best.pth')):
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'best.pth'), map_location=device, weights_only=True))
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    # Adding LR Scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = EastLoss().to(device)
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    history = []
    best_f1 = 0

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        
        # --- PHASED TRAINING: Freeze backbone for first 5 epochs if starting fresh ---
        if epoch < 5 and not os.path.exists(os.path.join(WEIGHTS_DIR, 'best.pth')):
            for param in model.extractor.parameters():
                param.requires_grad = False
            print(f"  Epoch {epoch+1}: Backbone frozen to stabilize Merge/Output blocks.")
        else:
            for param in model.extractor.parameters():
                param.requires_grad = True

        epoch_loss = 0
        valid_batches = 0 # Đếm số batch hợp lệ (không bị inf/nan)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, gt_score, gt_geo, _ in pbar:
            images, gt_score, gt_geo = images.to(device), gt_score.to(device), gt_geo.to(device)
            
            optimizer.zero_grad()
            
            # CHỈ để model dự đoán trong môi trường FP16 (Autocast)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred_score, pred_geo = model(images)
            
            # KÉO RA KHỎI AUTOCAST: Ép về Float32 trước khi tính Loss để tránh giới hạn 65504
            pred_score = pred_score.float()
            pred_geo = pred_geo.float()
            loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
            
            # Lưới lọc an toàn: Bỏ qua batch nếu Loss bị inf hoặc nan
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nCảnh báo: Loss bị {loss.item()}, đang bỏ qua batch này...")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            # Lưu lại scale trước khi update để check xem optimizer có bị skip không
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            
        # KHẮC PHỤC WARNING SCHEDULER: Chỉ nhảy Scheduler nếu Optimizer đã được cập nhật
        scale_after = scaler.get_scale()
        skip_lr_sched = (scale_before > scale_after)
        if not skip_lr_sched:
            scheduler.step()
            
        # Tính trung bình dựa trên số batch hợp lệ
        avg_train_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
        
        # Evaluation
        val_metrics = evaluate_east(model, val_loader, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Val - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  FPS: {val_metrics['fps']:.1f}")
        
        log_entry = {
            'epoch': epoch+1, 
            'train_loss': avg_train_loss, 
            'lr': optimizer.param_groups[0]['lr'],
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