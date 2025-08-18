from typing import Sequence
import numpy as np


def get_anchor_grid(
    num_rows: int,
    num_cols: int,
    scale_factor: float, # ratio between image size and feature map size
    anchor_widths: Sequence[float], # all possible box widths
    aspect_ratios: Sequence[float], # all possible box aspect ratios height / width
) -> np.ndarray:
    # assign numpy array so i dont get the size wrong
    array = np.zeros((len(anchor_widths), len(aspect_ratios), num_rows, num_cols, 4))
    # print(array.shape)
    for i, width in enumerate(anchor_widths):
        for j, aspect_ratio in enumerate(aspect_ratios):
            for row in range(num_rows):
                center_y = (row + 0.5) * scale_factor
                for col in range(num_cols):
                    height = width * aspect_ratio
                    center_x = (col + 0.5) * scale_factor
                    array[i, j, row, col] = [
                        center_x - width / 2,
                        center_y - height / 2,
                        center_x + width / 2,
                        center_y + height / 2,
                    ]

    return array