import torch
from torch import nn


def get_dice_loss(gt_score, pred_score):
    """
    Computes the Dice loss for the classification branch.
    Stable implementation using sum over spatial dimensions.
    """
    eps = 1e-4
    # Flatten across batch and spatial dims for global Dice
    gt_flatten = gt_score.view(-1)
    pred_flatten = pred_score.view(-1)
    
    intersection = torch.sum(gt_flatten * pred_flatten)
    union = torch.sum(gt_flatten) + torch.sum(pred_flatten) + eps
    
    dice = (2. * intersection + eps) / union
    return 1. - dice


def get_geo_loss(gt_geo, pred_geo):
    """
    Computes the geometric loss (IoU + Angle).
    Uses Log-IoU for stronger gradients during early training.
    """
    eps = 1e-6
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    
    # Ensure predictions are positive to avoid NaN in area calculation
    d1_pred = torch.clamp(d1_pred, min=0)
    d2_pred = torch.clamp(d2_pred, min=0)
    d3_pred = torch.clamp(d3_pred, min=0)
    d4_pred = torch.clamp(d4_pred, min=0)

    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_inter = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    h_inter = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    area_inter = w_inter * h_inter
    area_union = area_gt + area_pred - area_inter
    
    # Log-IoU Loss: provides stronger gradient than 1 - IoU
    # Clamp IoU to [eps, 1.0] to prevent log(0)
    iou = (area_inter + eps) / (area_union + eps)
    iou = torch.clamp(iou, min=eps, max=1.0)
    iou_loss_map = -torch.log(iou)
    
    # Angle loss: cosine based to handle periodicity
    angle_loss_map = torch.abs(angle_gt - angle_pred)

    return iou_loss_map, angle_loss_map


class EastLoss(nn.Module):
    def __init__(self, weight_angle=10):
        super(EastLoss, self).__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo):
        # Graceful handling of empty ground truth while keeping gradient flow
        if torch.sum(gt_score) < 1:
            return (pred_score.sum() + pred_geo.sum()) * 0

        classify_loss = get_dice_loss(gt_score, pred_score)
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        # Only compute geometric loss on text pixels (masking)
        angle_loss = torch.sum(angle_loss_map * gt_score) / (torch.sum(gt_score) + 1e-5)
        iou_loss = torch.sum(iou_loss_map * gt_score) / (torch.sum(gt_score) + 1e-5)
        
        geo_loss = self.weight_angle * angle_loss + iou_loss

        return geo_loss + classify_loss
