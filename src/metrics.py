#metrics.py
import torch
import torch.nn as nn

def evaluate_depth(pred, target):
    assert pred.shape == target.shape

    valid_mask = target > 0
    pred = pred[valid_mask]
    target = target[valid_mask]

    pred = torch.clamp(pred, min=1e-3)
    target = torch.clamp(target, min=1e-3)

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float()      / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)


    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))
    mae = torch.mean(torch.abs(diff))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


#evaluation metrics for instance segmentation
def compute_iou(pred, target):
    # pred (np.array): Predicted box/mask.
    # target (list of np.array): List of ground truth boxes/masks.'

    iou_values = []
    for gt in target:
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = intersection / union if union > 0 else 0
        iou_values.append(iou)
    return iou_values

def compute_AP(tp, fp, fn):

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    ap = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return ap
    



def evaluate_instance(pred_boxes, pred_classes, pred_masks, target_boxes,target_classes, target_masks, iou_threshold=0.5):

    
    
    n = len(pred_boxes)
    tp_boxes, fp_boxes, fn_boxes = 0, 0, 0
    tp_masks, fp_masks, fn_masks = 0, 0, 0

    for i in range(n):
        pred_box = pred_boxes[i]
        pred_class = pred_classes[i]
        pred_mask = pred_masks[i]

        iou_box  = compute_iou(pred_box, target_boxes)
        iou_mask = compute_iou(pred_mask, target_masks)

        best_iou_box_idx = torch.argmax(iou_box)
        best_iou_mask_idx = torch.argmax(iou_mask)

        if iou_box[best_iou_box_idx] > iou_threshold and pred_class == target_classes[best_iou_box_idx]:
            tp_boxes += 1
        else:
            fp_boxes += 1
        if best_iou_box_idx >= len(target_boxes):
            fn_boxes += 1

        if iou_mask[best_iou_mask_idx] > iou_threshold and pred_class == target_classes[best_iou_mask_idx]:
            tp_masks += 1
        else:
            fp_masks += 1
        if best_iou_mask_idx >= len(target_masks):
            fn_masks += 1

    boxAP = compute_AP(tp_boxes, fp_boxes, fn_boxes)
    maskAP = compute_AP(tp_masks, fp_masks, fn_masks) 



    return {'maskAP': maskAP, 'boxAP': boxAP}





### Example usage
if __name__ == '__main__':
    pred = torch.tensor([[[[1.5, 1.2], [1.3, 1.0]]]], dtype=torch.float32)  # Example predicted depth
    target = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)  # Example target depth
    mask = None  # Example mask (not applied)
    metrics = evaluate_depth(pred, target, mask)
    print(metrics)


    