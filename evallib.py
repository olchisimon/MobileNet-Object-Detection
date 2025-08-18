import numpy as np
from .label_grid import iou
from .annotation import AnnotationRect


def _assign_detections(det_boxes_scores, gt_boxes, min_iou):
    """Returns a list of 3-tuples (det_box, det_score, closest_gt_box)"""

    assert min_iou > 0, "min_iou should be greater than zero"
    det2gt = []
    for det_box, det_score in det_boxes_scores:
        # get all GTs that overlap at least by 0.5 and sort descending by overlap
        best_iou = -1.0
        closest_gt_box = None
        for gt in gt_boxes:
            curr_iou = iou(gt, det_box)
            if curr_iou > best_iou:
                best_iou = curr_iou
                closest_gt_box = gt

        if best_iou >= min_iou:
            det2gt.append((det_box, det_score, closest_gt_box))
        else:
            det2gt.append((det_box, det_score, None))

    return det2gt


def calculate_ap_pr(det_boxes_scores: dict, gt_boxes: dict, min_iou=0.5):
    """
    Calculates average precision for the given detections and ground truths.
    This function also calculates precision and recall values that can be used to plot a PR curve

    @param det_boxes_scores: A dictionary that maps image numbers to a list of tuples,
    containing detected AnnotationRect objects and a score.
    @param gt_boxes: A dictionary that maps image numbers to the list of ground truth AnnotationRect objects.
    """
    """
    for img_idx, detections in det_boxes_scores.items():
        print(f"Bild {img_idx}: {len(detections)} Boxen vorhergesagt")
        for box, score in detections:
            print(f"  {box} mit Score {score:.4f}")
        print(f"Bild {img_idx}: {len(gt_boxes[img_idx])} Boxen gegengesetzt")
        for box in gt_boxes[img_idx]:
            print(f"  {box}")
    """

    for dets in det_boxes_scores.values():
        assert all(
            isinstance(d, tuple) and len(d) == 2 for d in dets
        ), "Your detection boxes must have scores"

    assert (
        len(set(gt_boxes.keys()).intersection(det_boxes_scores.keys())) > 0
    ), "The two dictionaries have no common keys. Maybe you have selected the wrong dataset?"

    gts_total = 0

    dets2gts_flat = []
    for img in gt_boxes:
        gts_total += len(gt_boxes[img])
        det2gts = _assign_detections(
            det_boxes_scores.get(img, []), gt_boxes[img], min_iou
        )
        dets2gts_flat.extend(det2gts)

    # sort by detection confidence
    # p[1] is the score
    dets2gts_flat.sort(key=lambda p: p[1], reverse=True)

    tp = np.zeros(len(dets2gts_flat), dtype=np.float32)
    fp = np.zeros(len(dets2gts_flat), dtype=np.float32)
    fn_cum = np.zeros(len(dets2gts_flat), dtype=np.float32)
    gts_seen = {None}
    for idx, (_det, _det_score, gt) in enumerate(dets2gts_flat):
        gt_prev_seen = gt in gts_seen
        # If detection was assigned to GT that was not previously assigned to anything-> TP
        tp[idx] = gt is not None and not gt_prev_seen
        # If detection was not assigned to GT or detection was assigned to a previously assigned GT -> FP
        fp[idx] = (gt is None) or gt_prev_seen
        # all gts that have not been assigned yet -> FN
        gts_seen.add(gt)
        fn_cum[idx] = gts_total - (len(gts_seen) - 1)

    # dets2gts_flat i sorted by score, therefore we can calculate tp and fp with cumsum
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precision = tp_cum / (tp_cum + fp_cum)
    # print(precision)
    recall = tp_cum / (tp_cum + fn_cum)
    # print(recall)
    average_prevision = np.trapz(precision, recall)
    return average_prevision, precision, recall


def example():
     dboxes = {
         "31": [
             (AnnotationRect(1, 2, 3, 4), 0.234),
             (AnnotationRect(9, 8, 7, 6), 2.431093),
         ],
         "32": [
             (AnnotationRect(1, 2, 3, 4), 0.234),
             (AnnotationRect(9, 8, 7, 6), 2.431093),
         ],
         # ...
     }
     gboxes = {
         "31": [AnnotationRect(0, 3, 2, 3)],
         "32": [AnnotationRect(9, 8, 7, 6), AnnotationRect(1, 2, 40, 40)],
     }
     ap, _, _ = calculate_ap_pr(dboxes, gboxes)
     print("ap is", ap)
