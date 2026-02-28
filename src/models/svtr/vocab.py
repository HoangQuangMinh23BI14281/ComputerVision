import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        # Use PyTorch's built-in CTC Loss
        self.criterion = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)

    def forward(self, preds, targets, preds_lengths, targets_lengths):
        """
        preds: Tensor of shape [seq_len, batch_size, num_classes]
        targets: Tensor of flattened labels [sum(targets_lengths)]
        preds_lengths: Tensor of int [batch_size]
        targets_lengths: Tensor of int [batch_size]
        """
        # CTC loss requires log_softmax
        log_probs = nn.functional.log_softmax(preds, dim=2)
        loss = self.criterion(log_probs, targets, preds_lengths, targets_lengths)
        return loss

def compute_loss(model_out, targets, targets_lengths, ctc_loss_func):
    """
    Helper function to compute loss with mixed precision and auto-generated preds_lengths.
    model_out: [seq_len, batch_size, num_classes]
    """
    seq_len, batch_size, _ = model_out.shape
    preds_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long, device=model_out.device)
    
    # Ép cứng về float32 để chống nhiễu NaN
    model_out_fp32 = model_out.float() 
    
    # Tắt autocast trong phạm vi tính loss
    with torch.autocast(device_type=model_out.device.type, enabled=False):
        loss = ctc_loss_func(model_out_fp32, targets, preds_lengths, targets_lengths)
    
    return loss

class Vocab:
    def __init__(self, character_str):
        """
        character_str: A string of all valid characters (e.g. "0123456789abcdefghijklmnopqrstuvwxyz")
        CTC uses the blank token, typically at index 0.
        """
        self.character = list(character_str)
        self.dict = {}
        # blank is 0
        for i, char in enumerate(self.character):
            self.dict[char] = i + 1

    @property
    def num_classes(self):
        # characters + 1 for blank token
        return len(self.character) + 1 

    def encode(self, text_list):
        """
        text_list: list of strings (e.g. ['hello', 'world'])
        Returns:
            targets: flattened tensor of indices
            targets_lengths: tensor containing the length of each sequence
        """
        targets = []
        lengths = []
        for s in text_list:
            encoded_s = []
            for char in s:
                # Skip unknown characters.
                if char in self.dict:
                    encoded_s.append(self.dict[char])
            targets.extend(encoded_s)
            lengths.append(len(encoded_s))
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode(self, preds_idx, raw=False):
        """
        preds_idx: list or tensor of index paths, typically shape (batch, seq_len)
        Returns list of decoded strings.
        """
        texts = []
        for seq in preds_idx:
            char_list = []
            for i in range(len(seq)):
                idx = seq[i]
                if idx != 0 and (not (not raw and i > 0 and seq[i - 1] == idx)):
                    # Add character to string if it's not the blank token (0).
                    # If raw=False, also collapse repeated adjacent characters.
                    if 1 <= idx <= len(self.character):
                        char_list.append(self.character[idx - 1])
            texts.append(''.join(char_list))
        return texts
