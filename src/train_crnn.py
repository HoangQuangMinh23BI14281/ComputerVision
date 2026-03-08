import torch
from torch import nn
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


from src.dataset.crnn_dataset import CrnnDataset, collate_fn
from src.models.crnn.model import CRNN
from src.config import REC_CHAR_SET

def train_batch(model, images, text_encodes, text_lens, optimizer, criterion, device, scaler):
    model.train()
    images = images.to(device)
    text_encodes = text_encodes.to(device)
    text_lens = text_lens.to(device)

    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        logits = model(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        batch_size = logits.size(1)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.flatten(text_lens)

        loss = criterion(log_probs, text_encodes, input_lengths, target_lengths)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


import time
import pandas as pd
from src.utils.metrics import calculate_crnn_metrics
from src.utils.label_converter import LabelConverter

def evaluate_crnn(model, dataloader, device, converter):
    model.eval()
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, targets, lens, raw_texts in dataloader:
            images = images.to(device)
            logits = model(images)
            
            # Greedy Decode
            preds = logits.argmax(2).permute(1, 0).cpu().numpy()
            decoded_preds = converter.decode_batch(preds)
            
            all_preds.extend(decoded_preds)
            all_targets.extend(raw_texts)

    inf_time = (time.time() - start_time) / len(dataloader.dataset) * 1000 # ms per image
    metrics = calculate_crnn_metrics(all_preds, all_targets)
    metrics['inf_time'] = inf_time
    return metrics

def main():
    TRAIN_TXT = 'ocr_dataset/rec_train.txt'
    VAL_TXT = 'ocr_dataset/rec_val.txt'
    TEST_TXT = 'ocr_dataset/rec_test.txt'
    WEIGHTS_DIR = 'weights/crnn'
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    LOG_PATH = os.path.join(WEIGHTS_DIR, 'log.csv')
    
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Splits
    X_train, y_train = CrnnDataset.load_from_txt(TRAIN_TXT)
    X_val, y_val = CrnnDataset.load_from_txt(VAL_TXT)
    X_test, y_test = CrnnDataset.load_from_txt(TEST_TXT)

    train_dataset = CrnnDataset(X_train, y_train)
    val_dataset = CrnnDataset(X_val, y_val)
    test_dataset = CrnnDataset(X_test, y_test)

    # Optimized DataLoader
    num_workers = 4 if os.name != 'nt' else 0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=num_workers)

    model = CRNN().to(device)
    if os.path.exists(os.path.join(WEIGHTS_DIR, 'best.pth')):
        print("Loading best weights...")
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'best.pth'), map_location=device, weights_only=True))
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    converter = LabelConverter(REC_CHAR_SET)
    history = []
    best_wacc = 0

    for epoch in range(30):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, text_encodes, text_lens, _ in pbar:
            loss = train_batch(model, images, text_encodes, text_lens, optimizer, criterion, device, scaler)
            epoch_loss += loss
            pbar.set_postfix(loss=f"{loss:.4f}")
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Evaluation
        val_metrics = evaluate_crnn(model, val_loader, device, converter)
        test_metrics = evaluate_crnn(model, test_loader, device, converter)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val  - W-Acc: {val_metrics['w_acc']:.4f}, C-Acc: {val_metrics['c_acc']:.4f}, NED: {val_metrics['ned']:.4f}")
        print(f"  Test - W-Acc: {test_metrics['w_acc']:.4f}, C-Acc: {test_metrics['c_acc']:.4f}, NED: {test_metrics['ned']:.4f}")
        print(f"  Inf Time: {val_metrics['inf_time']:.2f}ms/img")
        
        # Logging
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_w_acc': val_metrics['w_acc'], 'val_c_acc': val_metrics['c_acc'], 'val_ned': val_metrics['ned'],
            'test_w_acc': test_metrics['w_acc'], 'test_c_acc': test_metrics['c_acc'], 'test_ned': test_metrics['ned'],
            'inf_time': val_metrics['inf_time']
        }
        history.append(log_entry)
        pd.DataFrame(history).to_csv(LOG_PATH, index=False)
        
        # Save Best
        if val_metrics['w_acc'] > best_wacc:
            best_wacc = val_metrics['w_acc']
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'best.pth'))
            print("  New Best Model Saved!")
            
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'latest.pth'))


if __name__ == '__main__':
    main()
    pass
