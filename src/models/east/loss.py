import torch
from torch import nn

def get_score_loss(gt_score, pred_score):
    """
    Kết hợp OHEM BCE và Dice Loss để chống mất cân bằng class.
    Khắc phục tình trạng F1 = 0.0000
    """
    eps = 1e-4
    
    # 1. Dice Loss
    intersection = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + eps
    dice_loss = 1.0 - (2.0 * intersection + eps) / union
    
    # 2. OHEM BCE Loss
    pred_score_bce = torch.clamp(pred_score, min=1e-7, max=1.0 - 1e-7)
    pos_mask = gt_score > 0.5
    neg_mask = gt_score <= 0.5
    
    pos_loss = -torch.log(pred_score_bce[pos_mask])
    neg_loss = -torch.log(1.0 - pred_score_bce[neg_mask])
    
    pos_count = pos_mask.sum()
    if pos_count > 0:
        neg_count_to_keep = min(int(pos_count * 3), neg_loss.size(0))
        if neg_count_to_keep > 0:
            neg_loss_hard, _ = torch.topk(neg_loss, neg_count_to_keep)
        else:
            neg_loss_hard = torch.tensor(0.0, device=gt_score.device)
        bce_loss = (pos_loss.sum() + neg_loss_hard.sum()) / (pos_count + neg_count_to_keep + eps)
    else:
        neg_count_to_keep = max(1, int(neg_loss.size(0) * 0.1))
        neg_loss_hard, _ = torch.topk(neg_loss, neg_count_to_keep)
        bce_loss = neg_loss_hard.mean()

    return dice_loss + bce_loss

def get_geo_loss(gt_geo, pred_geo):
    eps = 1e-6
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    
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
    
    iou = (area_inter + eps) / (area_union + eps)
    iou = torch.clamp(iou, min=eps, max=1.0)
    iou_loss_map = -torch.log(iou)
    
    # HOÀN TÁC LẠI COSINE ĐỂ TRÁNH LỖI CHU KỲ GÓC (-90 và +90)
    angle_loss_map = 1 - torch.cos(angle_gt - angle_pred)

    return iou_loss_map, angle_loss_map

class EastLoss(nn.Module):
    def __init__(self, weight_angle=10):
        super(EastLoss, self).__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo):
        # LUÔN tính classify loss với OHEM mới
        classify_loss = get_score_loss(gt_score, pred_score)

        if torch.sum(gt_score) < 1e-5:
            # Khắc phục lỗi mất Gradient ảnh trống
            return classify_loss + (pred_geo.sum() * 0.0)

        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map * gt_score) / (torch.sum(gt_score) + 1e-5)
        iou_loss = torch.sum(iou_loss_map * gt_score) / (torch.sum(gt_score) + 1e-5)
        
        geo_loss = self.weight_angle * angle_loss + iou_loss

        return geo_loss + classify_loss