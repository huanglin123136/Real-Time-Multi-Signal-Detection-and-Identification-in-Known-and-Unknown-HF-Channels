import os
import numpy as np


def iou_g(gt, loc, label, th=0.3):
    gt_label = int(gt[-1])
    gt_loc = gt[:-1]
    loc_min = max(loc[0], gt_loc[0])
    loc_max = min(loc[1], gt_loc[1])
    inter = loc_max - loc_min
    iou = inter / (loc[1] - loc[0] + gt_loc[1] - gt_loc[0] - inter)
    if iou < th or label != gt_label:
        return False
    else:
        return True


def iou_g_2(gt, loc, th=0.3):

    loc_min = max(loc[0], gt[0])
    loc_max = min(loc[1], gt[1])
    inter = loc_max - loc_min
    iou = inter / (loc[1] - loc[0] + gt[1] - gt[0] - inter)
    if iou < th:
        return False
    else:
        return True


def match_sample(gt, locs, classes, scores):
    if len(gt) == 0:
        return len(locs) == 0

    det = [False] * len(gt)
    scores = np.array(scores)
    sort_index = np.argsort(-scores)
    sorted_scores = [scores[x] for x in sort_index]
    sorted_locs = [locs[x] for x in sort_index]
    sorted_classes = [classes[x] for x in sort_index]
    add_ssb = []
    for i in range(len(sort_index)):
        if sorted_classes[i] == 0:
            lmin, lmax = sorted_locs[i]
            lmean = (lmax + lmin) / 2
            add_ssb.append((lmin, lmean))
            add_ssb.append((lmean, lmax))
    tp = 0
    fp = 0
    matched_gt = 0
    for i in range(len(sort_index)):
        pred_loc = sorted_locs[i]
        pred_label = sorted_classes[i]
        match = 0
        for j in range(len(gt)):
            if match == 0 and not det[j] and iou_g(gt[j], pred_loc, pred_label, 0.5):
                det[j] = True
                match = 1
                matched_gt += 1
                tp += 1
        if match == 0:
            if pred_label == 1:
                match_ssb = 0
                for j in range(len(add_ssb)):
                    if not match_ssb and iou_g_2(add_ssb[j], pred_loc, 0.4):
                        match_ssb = 1
                if not match_ssb:
                    fp += 1
            else:
                fp += 1
        if matched_gt == len(gt):
            break
    return fp < 2 and tp >= max(1, len(gt) - 1)

