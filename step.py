from datetime import datetime
import os
import random
import torch
import torch.optim as optim

from .bbr import get_bbr_loss
from .model import MmpNet
from .parameters import MATCH_THRESHOLD_BBR, NEGATIVE_MINING_RATIO
from torch.utils.tensorboard import SummaryWriter


def step(
        model: MmpNet,
        criterion,
        optimizer: optim.Optimizer,
        img_batch: torch.Tensor,
        lbl_batch: torch.Tensor,
        anchor_grid: torch.Tensor,
        annotation_batch: list[torch.Tensor,],  # contains the actual annotations
        weight_bbr_loss: float = 0.5,
        sampling: str = "random"
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """
    optimizer.zero_grad()
    anchor_output, bbr_output = model(img_batch)

    unfiltered = criterion(anchor_output, lbl_batch)

    rand = random.random()
    if rand > 0:
        mask = get_random_sampling_mask(lbl_batch, NEGATIVE_MINING_RATIO)
    else:
        mask = get_hard_negative_mining_mask(lbl_batch, unfiltered, NEGATIVE_MINING_RATIO)


    filtered = unfiltered[mask == 1]  # when using mask
    anchor_loss = filtered.mean()  # when using mask

    gt_boxes_per_anchor, valid_matches = match_anchors_to_annotations(anchor_grid, lbl_batch, annotation_batch, MATCH_THRESHOLD_BBR)

    positive_mask = (lbl_batch == 1)
    bbr_mask = positive_mask & valid_matches

    anchor_grid_batched = anchor_grid.unsqueeze(0).expand(lbl_batch.shape[0], -1, -1, -1, -1, -1)
    anchor_boxes = anchor_grid_batched[bbr_mask]
    adjustments = bbr_output[bbr_mask]
    groundtruth_boxes = gt_boxes_per_anchor[bbr_mask]

    if bbr_mask.any():
        valid_gt_mask = (groundtruth_boxes.sum(dim=1) != 0)
        if valid_gt_mask.any():
            bbr_loss = get_bbr_loss(
                anchor_boxes[valid_gt_mask],
                adjustments[valid_gt_mask],
                groundtruth_boxes[valid_gt_mask]
            )
        else:
            bbr_loss = torch.tensor(0.0, device=img_batch.device)
            print("Warning: All ground truth boxes are (0,0,0,0), skipping BBR loss")
    else:
        bbr_loss = torch.tensor(0.0, device=img_batch.device)

    # print(f"Anchor Loss: {anchor_loss.item():.4f}, BBR Loss: {bbr_loss.item():.4f}")
    loss = anchor_loss + weight_bbr_loss * bbr_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def match_anchors_to_annotations(
        anchor_grid: torch.Tensor,
        lbl_batch: torch.Tensor,
        annotations: list[torch.Tensor,],
        iou_threshold: float = 0.6
):
    # anchor_grid shape (anchor_widths, anchors_ratios, rows, cols, 4)
    # lbl_batch shape   (batch_size, anchor_widths, anchors_ratios, rows, cols)
    batch_size = lbl_batch.shape[0]
    gt_boxes_per_anchor = torch.zeros(lbl_batch.shape + (4,), device=lbl_batch.device, dtype=torch.float32)

    valid_matches = torch.zeros_like(lbl_batch, dtype=torch.bool, device=lbl_batch.device)

    for b in range(batch_size):
        gt_boxes = annotations[b].to(lbl_batch.device)
        mask = lbl_batch[b] == 1

        # continue if no predictions where made or now persons are in picture
        if gt_boxes.numel() == 0 or mask.sum() == 0:
            continue

        positive_boxes_grid = anchor_grid[mask]

        ious = iou_tensors(positive_boxes_grid, gt_boxes)
        max_iou, max_indices = ious.max(dim=1)

        threshold_mask = max_iou > iou_threshold

        if threshold_mask.any():
            matched_gt_boxes = gt_boxes[max_indices[threshold_mask]]

            temp_mask = torch.zeros_like(mask, dtype=torch.bool)
            positive_indices = mask.nonzero(as_tuple=True)
            valid_indices = tuple(idx[threshold_mask] for idx in positive_indices)
            temp_mask[valid_indices] = True

            gt_boxes_per_anchor[b][temp_mask] = matched_gt_boxes
            valid_matches[b][temp_mask] = True

    return gt_boxes_per_anchor, valid_matches


def iou_tensors(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes the intersection over union (IOU) as a tensor."""

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])


    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)
    intersection = width_height[:, :, 0] * width_height[:, :, 1]

    union = area1[:, None] + area2 - intersection

    iou = intersection / union.clamp(min=1e-8)
    return iou

def get_tensorboard_writer(model_name, log_path):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(log_path, f"{model_name}_{current_time}")
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(log_path, f"{model_name}_{current_time}"))
    return tensorboard_writer, path

def get_random_sampling_mask(labels: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    """
    @param labels: The label tensor that is returned by your data loader.
    The values are either 0 (negative label) or 1 (positive label).
    @param neg_ratio: The desired negative/positive ratio.
    Hint: after computing the mask, check if the neg_ratio is fulfilled.
    @return: A tensor with the same shape as labels
    """
    mask = torch.zeros_like(labels, dtype=torch.float32) # create mask

    positives = (labels == 1) # bool map for ones
    mask[positives] = 1.0
    negatives = (labels == 0) # bool map for zeros

    # counts
    num_positives = positives.sum().item()
    num_negatives = negatives.sum().item()
    num_negatives_required = int(num_positives * neg_ratio)
    num_negatives_required = min(num_negatives_required, num_negatives)

    negative_indices = negatives.nonzero(as_tuple=True)

    if num_negatives_required > 0:
        permutation = torch.randperm(num_negatives, device=labels.device)
        negatives_selected = permutation[:num_negatives_required]

        selected_indices = tuple(idx[negatives_selected] for idx in negative_indices)
        mask[selected_indices] = 1.0

    # fallback if no element was selected - select all elements
    if mask.sum() == 0:
        if num_positives == 0 and num_negatives > 0:

            num_to_select = max(1, min(64, num_negatives // 8))
            negative_indices = negatives.nonzero(as_tuple=True)
            if len(negative_indices[0]) > 0:

                permutation = torch.randperm(len(negative_indices), device=labels.device)
                selected = tuple(idx[permutation[:num_to_select]] for idx in negative_indices)
                mask[selected] = 1.0
                print(f"Fallback: No positives found, selected {num_to_select} negatives")

        elif num_positives == 0 and num_negatives == 0:
            mask_flat = mask.view(-1)
            mask[:min(32, len(mask_flat))] = 1.0
            print("Fallback: No labels found, using minimal sample")

    return mask

def get_hard_negative_mining_mask(labels: torch.Tensor, losses: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    """
    Selektiert alle positiven Anker und für die negativen Anker diejenigen mit dem höchsten Loss.
    """
    mask = torch.zeros_like(labels, dtype=torch.float32)

    positives = labels == 1
    negatives = labels == 0

    mask[positives] = 1.0
    num_positives = positives.sum().item()
    num_negatives = negatives.sum().item()
    num_negatives_required = int(num_positives * neg_ratio)
    num_negatives_required = min(num_negatives_required, num_negatives)

    if num_negatives_required > 0:
        negative_losses = losses[negatives]
        _, indices = torch.topk(negative_losses, num_negatives_required)
        neg_idx_all = negatives.nonzero(as_tuple=True)
        hard_negative_indices = tuple(idx[indices] for idx in neg_idx_all)
        mask[hard_negative_indices] = 1.0

    return mask
