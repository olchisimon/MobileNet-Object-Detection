from typing import List, Sequence, Tuple
from PIL import Image, ImageDraw
import os

from .annotation import AnnotationRect
from .label_grid import iou

def non_maximum_suppression(
    boxes_scores: Sequence[Tuple[AnnotationRect, float]], threshold: float
) -> List[Tuple[AnnotationRect, float]]:
    """Exercise 6.1
    @param boxes_scores: Sequence of tuples of annotations and scores
    @param threshold: Threshold for NMS

    @return: A list of tuples of the remaining boxes after NMS together with their scores
    """
    final_predictions = []
    boxes_scores = sorted(boxes_scores, key=lambda x: x[1], reverse=True)

    while boxes_scores:
        candidate = boxes_scores.pop(0)
        candidate_box = candidate[0]

        keep = True
        for kept_box, _ in final_predictions:
            if  iou(candidate_box, kept_box) > threshold:
                keep = False
                break

        if keep:
            final_predictions.append(candidate)

    return final_predictions

def draw_image(image_path, boxes: List[AnnotationRect]):

    print(f"Drawing image {image_path} with {len(boxes)} boxes")
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for rect in boxes:
        coords = [rect.x1, rect.y1, rect.x2, rect.y2]
        draw.rectangle(coords , outline="#FF0000", width=2)
    return img


def main():
    data_path = r"C:\Users\olchi\Datasets\mmp2-trainval\val"
    with open("mmp/a6/model_output.txt", "r") as f:
        for line in f:
            boxes_scores = []
            data = line.strip().split(" ")
            current_img = data[0]
            while data[0] == current_img:
                rect = AnnotationRect(float(data[1]), float(data[2]),float(data[3]), float(data[4]))
                score = float(data[5])
                boxes_scores.append((rect, score))
                data = f.readline().strip().split(" ")

            nms_output = non_maximum_suppression(boxes_scores, 0.3)
            img_path = os.path.join(data_path, current_img + ".jpg")
            boxes = [box[0] for box in nms_output if box[1] > 0.5]
            img = draw_image(img_path, boxes)
            img.save(f"mmp/a6/images/{current_img}.jpg")

            boxes_scores.clear()
            nms_output.clear()

if __name__ == "__main__":
    main()

