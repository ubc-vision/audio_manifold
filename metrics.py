import numpy as np
import torch


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=20):
    pred_mask = pred_mask.contiguous().view(-1)
    mask = mask.contiguous().view(-1)
    iou_per_class = []
    for clas in range(0, n_classes):  # loop per pixel class
        true_class = pred_mask == clas
        true_label = mask == clas
        if true_label.long().sum().item() == 0:  # no exist label in this loop
            iou_per_class.append(np.nan)
        else:
            intersect = (
                torch.logical_and(true_class, true_label).sum().float().item()
            )
            union = (
                torch.logical_or(true_class, true_label).sum().float().item()
            )
            iou = (intersect + smooth) / (union + smooth)
            iou_per_class.append(iou)

    out = np.array(iou_per_class)
    return out


seg_metrics = mIoU


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))

    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    sq_rel = (((gt - pred) ** 2) / gt).mean()
    abs_rel = (np.abs(gt - pred) / gt).mean()
    abs_rel_all = np.abs(gt - pred) / gt

    rel_err = np.array(
        [(abs_rel_all < t).mean() for t in np.linspace(0, 0.3, 30)]
    )
    auc = (rel_err * 0.1).sum()

    asd = np.array([auc, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
    out = np.concatenate((asd, rel_err))

    return out


def depth_metrics(gt_disp, pred_disp, eval_stereo=False):
    # depth evaluation taken from: https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
    pred_depth_scale_factor = 1.0

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_depth = 1 / gt_disp
    gt_height, gt_width = gt_depth.shape[:2]

    pred_depth = 1 / pred_disp

    mask = gt_depth > 0
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    pred_depth *= pred_depth_scale_factor
    gt_depth *= pred_depth_scale_factor

    ratio = np.median(gt_depth) / np.median(pred_depth)
    # ratios.append(ratio)
    pred_depth *= ratio

    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    errors = compute_errors(gt_depth, pred_depth)

    # if not disable_median_scaling:
    # ratios = np.array(ratios)
    # med = np.median(ratios)

    return errors
