LEARNING_RATE = 1e-4
MIN_IOU = 0.4           # minimum overlap to count as positive
THRESHOLD_NMS = 0.4     # minimum overlap to be removed by nms

IMG_SIZE = 448
EPOCHS = 40

NEGATIVE_MINING_RATIO = 3

WEIGHT_BBR_LOSS = 1.0
MATCH_THRESHOLD_BBR = 0.6

# ANCHOR_WIDTHS = [4, 8, 16, 32, 64, 128, 256] old
# ASPECT_RATIOS = [0.5, 1.0, 1.5, 2.0, 3.0] old
ANCHOR_WIDTHS = [8, 16, 24, 32, 48, 64, 96, 128, 192]
ASPECT_RATIOS = [0.5, 0.75, 1.0, 1.33, 2.0]

# ANCHOR_WIDTHS = [16, 32, 64] # fpn
# ASPECT_RATIOS = [0.5, 1.0, 2.0] # fpn