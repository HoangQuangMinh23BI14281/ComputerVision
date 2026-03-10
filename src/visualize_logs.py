import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_crnn(log_path, save_dir):
    if not os.path.exists(log_path):
        print(f"CRNN log not found at {log_path}")
        return
    
    df = pd.read_csv(log_path)
    
    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='royalblue')
    plt.title('CRNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'crnn_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Accuracy Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['val_w_acc'], label='Val Word Acc', color='teal')
    plt.plot(df['epoch'], df['val_c_acc'], label='Val Char Acc', color='orange')
    plt.plot(df['epoch'], df['test_w_acc'], label='Test Word Acc', linestyle='--', color='darkgreen')
    plt.title('CRNN Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'crnn_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_east(log_path, save_dir):
    if not os.path.exists(log_path):
        print(f"EAST log not found at {log_path}")
        return
    
    df = pd.read_csv(log_path)
    
    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='crimson')
    plt.title('EAST Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'east_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Detection Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['precision'], label='Precision', color='forestgreen')
    plt.plot(df['epoch'], df['recall'], label='Recall', color='darkviolet')
    plt.plot(df['epoch'], df['f1'], label='F1-score', color='darkorange', linewidth=2)
    plt.plot(df['epoch'], df['iou'], label='IoU', color='dodgerblue', linestyle='--')
    plt.title('EAST Detection Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'east_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_dir = r"c:\Users\ADMIN\OneDrive\Desktop\ComVis"
    crnn_log = os.path.join(base_dir, "weights", "crnn", "log.csv")
    east_log = os.path.join(base_dir, "weights", "east", "log.csv")
    
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating CRNN plots...")
    plot_crnn(crnn_log, plots_dir)
    
    print("Generating EAST plots...")
    plot_east(east_log, plots_dir)
    
    print(f"Done! Plots saved to: {plots_dir}")
