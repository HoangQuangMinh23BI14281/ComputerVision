import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from src.models.dbnet.model import DBNet
from src.models.dbnet.loss import DBLoss
from src.dataset.dbnet_dataset import DBNetDataset

# Configuration
DATA_ROOT = "data/Stage1train"
TRAIN_LABEL = "ocr_dataset/det_train.txt"
VAL_LABEL = "ocr_dataset/det_test.txt"
BATCH_SIZE = 8
EPOCHS = 150
LR = 0.001
SAVE_DIR = "weights/dbnet"
IMG_SIZE = 640

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training DBNet on {device}...")
    
    # Initialize log file
    log_file = os.path.join(SAVE_DIR, "det_metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("epoch,train_loss,val_loss,precision,recall,f1,iou,lr\n")

    # Dataset & Dataloader
    train_dataset = DBNetDataset(DATA_ROOT, TRAIN_LABEL, is_train=True, img_size=IMG_SIZE)
    val_dataset = DBNetDataset(DATA_ROOT, VAL_LABEL, is_train=False, img_size=IMG_SIZE)

    torch.backends.cudnn.benchmark = True
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = DBNet(pretrained=True).to(device)
    
    # Kịch bản Resume Train (Load model đang train dở)
    latest_weight = os.path.join(SAVE_DIR, "latest.pth")
    start_epoch = 0
    if os.path.exists(latest_weight):
        print(f"RESUMING TRAINING: Loaded weights from {latest_weight}!")
        model.load_state_dict(torch.load(latest_weight, map_location=device))
        
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        start_epoch = int(last_line.split(',')[0])
                        print(f"Resuming from Epoch {start_epoch}")
            except Exception as e:
                print(f"Could not parse start_epoch from log: {e}")
    else:
        print("Starting training from scratch...")
    
    # Loss & Optimizer
    criterion = DBLoss(alpha=1.0, beta=10.0, ohem_ratio=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6, last_epoch=start_epoch - 1)

    best_loss = -1.0 # F1-Score ranges from 0 to 1, higher is better
    total_start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            batch_gt = {
                'gt': batch['gt'].to(device),
                'mask': batch['mask'].to(device),
                'thresh_map': batch['thresh_map'].to(device),
                'thresh_mask': batch['thresh_mask'].to(device)
            }

            optimizer.zero_grad()
            preds = model(images)
            loss_dict = criterion(preds, batch_gt)
            loss = loss_dict['loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'prob': loss_dict['loss_prob'].item(), 'thresh': loss_dict['loss_thresh'].item()})

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        # Pixel-level metric accumulators
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar_val:
                images = batch['image'].to(device)
                batch_gt = {
                    'gt': batch['gt'].to(device),
                    'mask': batch['mask'].to(device),
                    'thresh_map': batch['thresh_map'].to(device),
                    'thresh_mask': batch['thresh_mask'].to(device)
                }

                preds = model(images)
                loss_dict = criterion(preds, batch_gt)
                loss = loss_dict['loss']
                val_loss += loss.item()
                
                # Calculate pixel-level metrics
                # preds is [B, 3, H, W] -> index 0 is prob map
                prob_map = preds[:, 0, :, :]
                pred_mask = (prob_map > 0.3).float()
                
                # GT mask
                gt_mask = batch_gt['gt'].float()
                # Ignore mask where batch_gt['mask'] == 0
                ignore_mask = batch_gt['mask'].float()
                
                # Logical operations for metric extraction
                tp = ((pred_mask == 1) & (gt_mask == 1) & (ignore_mask == 1)).float().sum()
                fp = ((pred_mask == 1) & (gt_mask == 0) & (ignore_mask == 1)).float().sum()
                fn = ((pred_mask == 0) & (gt_mask == 1) & (ignore_mask == 1)).float().sum()
                
                total_tp += tp.item()
                total_fp += fp.item()
                total_fn += fn.item()

        avg_val_loss = val_loss / len(val_loader)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] Time: {epoch_time:.0f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log to file
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{precision:.6f},{recall:.6f},{f1:.6f},{iou:.6f},{scheduler.get_last_lr()[0]:.6f}\n")

        # Save Best Model Criteria based on F1-Score now (previously loss)
        if f1 > best_loss or (f1 == best_loss and avg_val_loss < best_loss):
            best_loss = f1 # Reusing best_loss for F1-Score
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
            print(f"[*] Saved best model with Val F1: {f1:.4f}")
            
        # Save Latest
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "latest.pth"))

    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")

if __name__ == '__main__':
    train()
