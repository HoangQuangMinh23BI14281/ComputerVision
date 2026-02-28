import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from transformers import get_cosine_schedule_with_warmup

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.svtr.model import SVTRv2
from src.models.svtr.vocab import Vocab, CTCLoss
from src.dataset.recognition_dataset import CRNNDataset, collate_crnn

# Configuration
DATA_ROOT = "data/Stage1train"
TRAIN_LABEL = "ocr_dataset/rec_train.txt"
VAL_LABEL = "ocr_dataset/rec_test.txt"
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
SAVE_DIR = "weights/svtr"
IMG_H = 32
IMG_W = 320

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training SVTRv2 on {device}...")
    
    # Initialize log file
    log_file = os.path.join(SAVE_DIR, "rec_metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("epoch,train_loss,val_loss,val_acc,lr\n")

    # Vocab
    char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    vocab = Vocab(char_set)

    # Dataset & Dataloader
    train_dataset = CRNNDataset(DATA_ROOT, TRAIN_LABEL, vocab=vocab, is_train=True, imgH=IMG_H, imgW=IMG_W)
    val_dataset = CRNNDataset(DATA_ROOT, VAL_LABEL, vocab=vocab, is_train=False, imgH=IMG_H, imgW=IMG_W)

    torch.backends.cudnn.benchmark = True
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_crnn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_crnn)

    # Model
    model = SVTRv2(imgH=IMG_H, nc=3, nclass=vocab.num_classes).to(device)
    
    # Kịch bản Resume Train (Load model đang train dở)
    latest_weight = os.path.join(SAVE_DIR, "latest.pth")
    start_epoch = 0
    if os.path.exists(latest_weight):
        print(f"RESUMING TRAINING: Loaded weights from {latest_weight}!")
        # strict=False is required because we added PositionalEncoding, Dropout, and Pre-Norm
        model.load_state_dict(torch.load(latest_weight, map_location=device), strict=False)
        
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
    criterion = CTCLoss(blank=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Scheduler with Warmup
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * max(1, EPOCHS // 20) # 5% warmup
    
    # Check if resuming
    current_step = 0
    if start_epoch > 0:
        current_step = start_epoch * len(train_loader)
        
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps,
    )
    # Fast forward scheduler if resuming
    if current_step > 0:
        for _ in range(current_step):
            scheduler.step()

    best_loss = float('inf')
    total_start_time = time.time()
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

    best_loss = float('inf')
    total_start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            optimizer.zero_grad()
            
            # Ép chạy Mixed Precision
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                preds = model(images) # [seq_len, batch_size, num_classes]
                from src.models.svtr.vocab import compute_loss
                loss = compute_loss(preds, targets, target_lengths, criterion)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Clip gradient (phải unscale trước khi clip)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Lưu lại scale cũ để kiểm tra xem optimizer có bị skip do NaN không
            old_scale = scaler.get_scale()
            
            # Step scaler and update loop
            scaler.step(optimizer)
            scaler.update()
            
            # Fix root cause of warning: Chỉ step scheduler nếu optimizer THỰC SỰ step
            # Nếu scale chênh lệch (get_scale() < old_scale) nghĩa là đã có lỗi NaN gradients => Optimizer bị skip.
            if scaler.get_scale() == old_scale:
                scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_words = 0
        total_words = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar_val:
                images = batch['images'].to(device)
                targets = batch['targets'].to(device)
                target_lengths = batch['target_lengths'].to(device)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    preds = model(images)
                    preds_size = torch.full((images.size(0),), preds.size(0), dtype=torch.int32, device=device)
                    # For loss, compute_loss helper is recommended if using fp16
                    from src.models.svtr.vocab import compute_loss
                    loss = compute_loss(preds, targets, target_lengths, criterion)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    
                # Accuracy Calculation
                preds_idx = preds.argmax(dim=2).transpose(0, 1).cpu().numpy()
                decoded_preds = vocab.decode(preds_idx)
                
                # Decode targets correctly from standard flattened 1D representation
                targets_cpu = targets.cpu().numpy()
                target_lengths_cpu = target_lengths.cpu().numpy()
                target_list = []
                start = 0
                for length in target_lengths_cpu:
                    target_list.append(targets_cpu[start:start+length])
                    start += length
                decoded_targets = vocab.decode(target_list, raw=True)
                
                for pred, target in zip(decoded_preds, decoded_targets):
                    total_words += 1
                    if pred == target:
                        correct_words += 1
                    
                # Decode the first batch for tracking observation
                if val_loss == loss.item(): # Equivalent to first batch
                    print(f"\n[Val Sample] Target: {decoded_targets[0]} -> Pred: {decoded_preds[0]}")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = correct_words / total_words if total_words > 0 else 0
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] Time: {epoch_time:.0f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log to file
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{avg_val_acc:.6f},{scheduler.get_last_lr()[0]:.6f}\n")

        # Save Best criteria based on Accuracy now (previously it was lowest loss)
        if avg_val_acc > best_loss or (avg_val_acc == best_loss and avg_val_loss < best_loss):
            best_loss = avg_val_acc # Reusing best_loss variable for Accuracy to save rewriting
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
            print(f"[*] Saved best model with Val Acc: {avg_val_acc:.4f}")
            
        # Save Latest
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "latest.pth"))

    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")

if __name__ == '__main__':
    train()
