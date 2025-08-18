from typing import List
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


class AnnotationRect:
    """Exercise 3.1"""

    def __init__(self, x1, y1, x2, y2):


        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def area(self):
        return abs((self.x2 - self.x1) * (self.y2 - self.y1))

    def __array__(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @staticmethod
    def fromarray(arr: np.ndarray):
        return AnnotationRect(*arr)

    def __str__(self):
        return f"[{self.x1:.2f}, {self.y1:.2f}, {self.x2:.2f}, {self.y2:.2f}]"
    def __repr__(self):
        return str(self)

def read_groundtruth_file(path: str) -> List[AnnotationRect]:
    """Exercise 3.1b"""
    l = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cords = line.split()
            l.append(AnnotationRect(*map(float, cords)))
    return l

# put your solution for exercise 3.1c wherever you deem it right
if __name__ == "__main__":
    path = Path(r"C:\Users\olchi\OneDrive\Desktop\Multimediaprojekt\Datasets\mmp2-trainval\train")

    max_annotations = 0
    max_annotations_path = None
    max_annotations_coords = []
    for file_path in path.glob("*.gt_data.txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()
            annotations = len(lines)
            if annotations > max_annotations:
                max_annotations = annotations
                max_annotations_path = file_path
                max_annotations_coords = lines

    img_path = str(max_annotations_path).replace(".gt_data.txt", ".jpg")
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    for line in max_annotations_coords:
        rect = line.split()
        coords = [int(float(x)) for x in rect]
        draw.rectangle(coords, outline="#FF0000", width=2)
    image.show()
    image.save("max_annotations.jpg")



