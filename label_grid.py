from typing import Sequence
import numpy as np

from .annotation import AnnotationRect


def iou(rect1: AnnotationRect, rect2: AnnotationRect) -> float:
    max_x1 = max(rect1.x1, rect2.x1)
    max_y1 = max(rect1.y1, rect2.y1)
    min_x2 = min(rect1.x2, rect2.x2)
    min_y2 = min(rect1.y2, rect2.y2)

    if max_x1 >= min_x2 or max_y1 >= min_y2:
        return 0.0
    else:
        intersection_area = (min_x2 - max_x1) * (min_y2 - max_y1)
        union_area = rect1.area() + rect2.area() - intersection_area
        return intersection_area / union_area



def get_label_grid(
    anchor_grid: np.ndarray, gts: Sequence[AnnotationRect], min_iou: float
) -> tuple[np.ndarray, ...]:

    anchor_grid_shape = anchor_grid.shape
    label_grid = np.zeros(anchor_grid_shape[:-1], dtype=bool)
    iou_counter = 0
    for gtb in gts:
        # loop over boxes in anchor grid
        for idx in np.ndindex(anchor_grid_shape[:-1]):
            anchor_box = AnnotationRect.fromarray(anchor_grid[idx])
            iou_val = iou(gtb, anchor_box)
            if iou_val >= min_iou:
                iou_counter += 1
                # print(f"Found match of: {anchor_grid[idx]} and {np.array(gtb)}")
                label_grid[idx] = True
    # print(label_grid.shape)
    # print("Overlapping boxes: ", iou_counter, "")
    return (label_grid,)
