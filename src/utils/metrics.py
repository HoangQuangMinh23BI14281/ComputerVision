import numpy as np
import torch
import time
from shapely.geometry import Polygon

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_ned(s1, s2):
    """Normalized Edit Distance: 1 - EditDistance(s1, s2) / max(len(s1), len(s2))"""
    if not s1 and not s2:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1 - dist / max(len(s1), len(s2), 1)

def calculate_crnn_metrics(preds, targets):
    """
    preds: List of predicted strings
    targets: List of ground truth strings
    returns: dict with 'w_acc', 'c_acc', 'ned'
    """
    word_correct = 0
    char_correct = 0
    total_chars = 0
    total_ned = 0
    
    for p, t in zip(preds, targets):
        if p == t:
            word_correct += 1
        
        total_ned += calculate_ned(p, t)
        
        # Simple char accuracy
        dist = levenshtein_distance(p, t)
        total_chars += max(len(t), 1)
        # char_correct is max(len(t)) - distance (but not less than 0)
        char_correct += max(0, max(len(t), len(p)) - dist)

    n = len(targets) if len(targets) > 0 else 1
    return {
        'w_acc': word_correct / n,
        'c_acc': char_correct / max(total_chars, 1),
        'ned': total_ned / n
    }

def calculate_iou(poly1, poly2):
    """Calculate IoU between two polygons using shapely"""
    try:
        p1 = Polygon(poly1.reshape(4, 2))
        p2 = Polygon(poly2.reshape(4, 2))
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        intersect = p1.intersection(p2).area
        union = p1.area + p2.area - intersect
        return intersect / union if union > 0 else 0.0
    except Exception:
        return 0.0

def calculate_east_metrics(gt_boxes_list, pred_boxes_list, iou_thresh=0.5):
    """
    gt_boxes_list: List of arrays of GT boxes (N, 8)
    pred_boxes_list: List of arrays of Pred boxes (M, 9) - last is score
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0
    iou_count = 0

    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        pred_boxes = np.array(pred_boxes)
        gt_boxes = np.array(gt_boxes)
        
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt: continue
                iou = calculate_iou(gb, pb[:8])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_thresh:
                total_tp += 1
                matched_gt.add(best_gt_idx)
                total_iou += best_iou
                iou_count += 1
            else:
                total_fp += 1
        
        total_fn += (len(gt_boxes) - len(matched_gt))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': avg_iou
    }
