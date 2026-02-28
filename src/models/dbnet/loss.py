import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        
        intersection = torch.sum(pred * gt * mask)
        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss

class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss

class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask):
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        
        if positive_count == 0:
            return pred.sum() * 0.0 # handle empty cases safely
            
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        
        # Hard negative mining
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss

class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, ohem_ratio=3):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.l1_loss = MaskL1Loss()
        self.dice_loss = DiceLoss()

    def forward(self, preds, batch):
        '''
        preds: Tensor of shape [N, 3, H, W] containing (probability_map, threshold_map, binary_map)
        batch: Dict from dataloader, typically containing:
            - 'gt': binary text label map [N, H, W]
            - 'mask': ignore mask for texts [N, H, W]
            - 'thresh_map': Ground truth threshold map [N, H, W]
            - 'thresh_mask': valid mask for threshold map [N, H, W]
        '''
        pred_prob = preds[:, 0, :, :]
        gt = batch['gt']
        mask = batch['mask']
        
        # 1. Probability Map Loss (Binary Cross Entropy with OHEM)
        loss_prob = self.bce_loss(pred_prob, gt, mask)
        
        # During eval(), DBNet only returns 1 channel (probability map) to save compute
        if preds.shape[1] == 1:
            return {
                'loss': loss_prob,
                'loss_prob': loss_prob,
                'loss_thresh': torch.tensor(0.0, device=preds.device),
                'loss_binary': torch.tensor(0.0, device=preds.device)
            }

        pred_thresh = preds[:, 1, :, :]
        pred_binary = preds[:, 2, :, :]
        thresh_map = batch['thresh_map']
        thresh_mask = batch['thresh_mask']

        # 2. Threshold Map Loss (L1 Loss inside valid text regions)
        loss_thresh = self.l1_loss(pred_thresh, thresh_map, thresh_mask)

        # 3. Binary Map Loss (Dice Loss comparing approximated binary map to ground truth)
        loss_binary = self.dice_loss(pred_binary, gt, mask)

        # Combined Loss
        loss = loss_prob + self.alpha * loss_binary + self.beta * loss_thresh

        return {
            'loss': loss,
            'loss_prob': loss_prob,
            'loss_thresh': loss_thresh,
            'loss_binary': loss_binary
        }
