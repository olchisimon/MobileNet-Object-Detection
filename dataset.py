from .annotation import AnnotationRect
from .label_grid import get_label_grid
from .annotation import read_groundtruth_file

from PIL import Image, ImageOps
from PIL.Image import Resampling
from typing import Tuple
import numpy as np
from pathlib import Path
import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torch.utils.data import DataLoader


class MMP_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: str,
        image_size: int,
        anchor_grid: np.ndarray,
        min_iou: float,
        is_test: bool,
        transformation_list: list[str]
    ):
        """
        @param anchor_grid: The anchor grid to be used for every image
        @param min_iou: The minimum IoU that is required for an overlap for the label grid.
        @param is_test: Whether this is the test set (True) or the validation/training set (False)
        """
        self.filenames = []
        self.anchor_grid = anchor_grid
        self.min_iou = min_iou
        self.is_test = is_test
        self.image_size = image_size
        self.transformation_list = transformation_list
        self.transformations = {
            "flip": v2.RandomHorizontalFlip(p=0.5),
            "rotate": v2.RandomRotation(degrees=15),
            "gray": v2.RandomGrayscale(p=0.1),
            "blur": v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            "crop": v2.RandomApply([v2.RandomIoUCrop()], p=0.5),
            "bright": v2.ColorJitter(brightness=0.3),
            "solar": v2.RandomSolarize(threshold=128.0),
            "contrast": v2.ColorJitter(contrast=0.2)
        }


        path = Path(path_to_data)
        for file in path.glob("*.jpg"):
            self.filenames.append(str(file))
        self.filenames.sort()

        # load image annotation only if not in test mode
        self.annotations = {}
        if not self.is_test:
            for file in self.filenames:
                text_file = file.replace(".jpg", ".gt_data.txt")
                image_annotation = read_groundtruth_file(text_file)
                self.annotations[file] = image_annotation

        # safe scaled annotations for future
        self.scaled_annotations = {}

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, list[AnnotationRect], int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        img_path = self.filenames[idx]
        img = Image.open(img_path)
        width, height = img.size
        if width > height:
            padded_img = ImageOps.expand(img, (0, 0, 0, width - height), fill="black")
        else:
            padded_img = ImageOps.expand(img, (0, 0, height - width, 0), fill="black")

        # resize image and rescale AnnotationRect box coordinates
        resized_img = padded_img.resize((self.image_size, self.image_size), resample=Resampling.LANCZOS)
        scale_factor_boxes = self.image_size / max(width, height)

        # get image id
        image_id_text = Path(img_path).stem

        if not self.is_test:
            scaled_annotations = [AnnotationRect(
                box.x1 * scale_factor_boxes,
                box.y1 * scale_factor_boxes,
                box.x2 * scale_factor_boxes,
                box.y2 * scale_factor_boxes,
            ) for box in self.annotations[img_path]]

            boxes = MMP_Dataset.convert_annotations_to_boxes(scaled_annotations)

            target = {
                "boxes": BoundingBoxes(
                    boxes,
                    format=BoundingBoxFormat.XYXY,
                    canvas_size=(self.image_size, self.image_size)
                ),
                "labels": torch.ones((boxes.shape[0],), dtype=torch.int64)
            }

            # transform image
            transform = self.build_transform(self.transformation_list)
            tensor_img, target = transform(resized_img, target)

            # convert tensor back to label annotations
            transformed_annotations = MMP_Dataset.convert_boxes_to_annotations(target["boxes"])
            self.scaled_annotations[int(image_id_text)] = transformed_annotations

            # label grid
            label_grid = get_label_grid(self.anchor_grid,
                                        transformed_annotations,
                                        self.min_iou)[0]

        else:
            # transform image test case
            transform = self.build_transform([])
            transformed_annotations = []
            tensor_img = transform(resized_img)

            # label grid
            label_grid = np.zeros(self.anchor_grid.shape[:-1], dtype=np.int32)

        return tensor_img, torch.from_numpy(label_grid), transformed_annotations, int(image_id_text)

    def __len__(self) -> int:
        return len(self.filenames)

    def build_transform(self, transformations):
        transform_list = []

        for transformation in transformations:
            transform_list.append(self.transformations[transformation])

        transform_list.append(v2.Resize((self.image_size, self.image_size))) # makes sure image is always size x size
        transform_list.append(v2.PILToTensor())
        transform_list.append(v2.ConvertImageDtype(torch.float))
        transform_list.append(v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]))
        return v2.Compose(transform_list)

    @staticmethod
    def convert_annotations_to_boxes(annotations):
        """converts annotations into torchtensor to use """

        # in case there are no people in the picture
        if not annotations:
            return torch.zeros((0, 4), dtype=torch.float32)

        # convert annotations to tensor
        boxes = []
        for annotation in annotations:
            boxes.append([annotation.x1, annotation.y1, annotation.x2, annotation.y2])

        return torch.tensor(boxes, dtype=torch.float32)

    @staticmethod
    def convert_boxes_to_annotations(boxes):
        """converts boxes into annotations to use"""
        annotations = []
        for box in boxes:
            annotations.append(AnnotationRect(box[0].item(), box[1].item(), box[2].item(), box[3].item()))
        return annotations


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length annotation lists
    """
    images, label_grids, annotation_lists, image_ids = zip(*batch)

    # Stack images and label grids normally
    images = torch.stack(images, 0)
    label_grids = torch.stack(label_grids, 0)

    # Keep annotation_lists as list (don't try to stack)
    annotation_lists = list(annotation_lists)

    # Convert image_ids to tensor
    image_ids = list(image_ids)

    return images, label_grids, annotation_lists, image_ids


def get_dataloader(
    path_to_data: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    anchor_grid: np.ndarray,
    is_test: bool,
    min_iou: float,
    transformations: list[str]
) -> DataLoader:
    dataset = MMP_Dataset(path_to_data, image_size, anchor_grid, min_iou, is_test, transformations)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=not is_test,
                      num_workers=num_workers,
                      collate_fn=custom_collate_fn)


