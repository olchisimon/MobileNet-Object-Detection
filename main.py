from typing import List, Tuple
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from .evallib import calculate_ap_pr
from .nms import non_maximum_suppression
from .annotation import AnnotationRect, read_groundtruth_file
from .anchor_grid import get_anchor_grid
from .model import MmpNet
from .step import get_tensorboard_writer, step
from .dataset import get_dataloader
from .bbr import apply_bbr
from .parameters import *


def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray
) -> List[List[Tuple[AnnotationRect, float]]]:
        images = images.to(device)
        # output shape - (B, 2, num_widths, num_ratios, height, width)
        anchor_output, bbr_output = model(images)

        # probs = torch.softmax(anchor_output, dim=1)

        results = []
        for i in range(anchor_output.shape[0]):
            anchor_scores = anchor_output[i]
            bbr_adjustments = bbr_output[i]

            list_per_image = []
            predicted_classes = anchor_scores.argmax(dim=0)
            values_humans = anchor_scores[1]

            for idx in np.ndindex(predicted_classes.shape):
                if predicted_classes[idx] == 1:

                    adjustment = bbr_adjustments[idx]
                    anchor_tensor =  torch.from_numpy(anchor_grid[idx]).float().to(device)
                    rect = apply_bbr(anchor_tensor, adjustment)

                    value = values_humans[idx].item()
                    list_per_image.append((rect, value))

            list_per_image = non_maximum_suppression(list_per_image, THRESHOLD_NMS)
            results.append(list_per_image)

        return results


def evaluate(model: MmpNet, val_loader: DataLoader, device: torch.device, anchor_grid: np.ndarray, file_path: str, image_size: int) -> float:  # feel free to change the arguments
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """

    print("Starting evaluation...")
    model.eval()

    # map batch_inference result to matching image id
    det_boxes_score = {}

    # map groundtruth-boxes to matching image id
    gt_boxes = {}

    with torch.no_grad():
        for images, labels, annotation_lists, image_ids in val_loader:
            batch_results = batch_inference(model, images, device, anchor_grid)

            for i in range(len(batch_results)):
                image_id = image_ids[i]

                # batch_inference results
                boxes_scores = batch_results[i]
                det_boxes_score[image_id] = boxes_scores.copy()

                # get image information
                img_path = file_path + "/" + str(image_id).zfill(8) + ".jpg"
                img = Image.open(img_path)
                width, height = img.size
                sf = image_size / max(width, height)

                # resize boxes back to image size
                for box, value in det_boxes_score[image_id]:
                    rb = AnnotationRect(box.x1 / sf, box.y1 / sf, box.x2 / sf, box.y2 / sf)
                    det_boxes_score[image_id][det_boxes_score[image_id].index((box, value))] = (rb, value)

                # groundtruth-boxes result
                gt_boxes_path = file_path + "/"+ str(image_id).zfill(8) + ".gt_data.txt"
                gt_boxes[image_id] = read_groundtruth_file(gt_boxes_path)

    # DEBUGGING
    #for key in gt_boxes:
    #    print(str(key) + ": ")
    #    print("Detected boxes: ")
    #    print(det_boxes_score[key])
    #    print("Groundtruth boxes: ")
    #    print(gt_boxes[key])
    #    print("--------------------------------")


    ap, precisions, recalls = calculate_ap_pr(det_boxes_score, gt_boxes)
    """
    x = ""
    while True:
        x = input("Gebe eine Bildnummer ein:")
        if x == "exit":
            break
        img = draw_image(file_path + "\\" + x.zfill(8) + ".jpg", [box for box, _ in det_boxes_score[int(x)]])
        for box, value in det_boxes_score[int(x)]:
            print(f"{box.x1}, {box.y1}, {box.x2}, {box.y2}, score: {value:.4f}")
        print("\n")
        print("Groundtruth boxes: ")
        for box in gt_boxes[int(x)]:
            print(f"{box.x1}, {box.y1}, {box.x2}, {box.y2}")

        img.show()
    """
    # plot presicion and recall
    plt.plot(recalls, precisions)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR-Kurve")
    plt.show()

    print(f"AP: {ap:.4f}")
    print("Finished evaluation.")
    return ap


def evaluate_test(model: MmpNet, test_loader: DataLoader, device: torch.device, anchor_grid: np.ndarray, file_path: str,  image_size: int):  # feel free to change the arguments
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.

    You decide which arguments this function should receive
    """
    print("Starting testing...")

    model.eval()

    with torch.no_grad():
        with open("predictions.txt", "w") as f:
            for images, label_grids, annotation_list, images_id in test_loader:
                batch_results = batch_inference(model, images, device, anchor_grid)

                for i in range(len(batch_results)):
                    image_id = images_id[i]
                    img_path = file_path + "/" + str(image_id).zfill(8) + ".jpg"

                    # get scale factor to backsize image
                    img = Image.open(img_path)
                    width, height = img.size
                    sf = image_size / max(width, height)

                    image_id = str(image_id).zfill(8)
                    boxes_scores = batch_results[i]


                    for box, value in boxes_scores:
                        rb = AnnotationRect(box.x1 / sf, box.y1 / sf, box.x2 / sf, box.y2 / sf)
                        f.write(f"{image_id} {int(rb.x1)} {int(rb.y1)} {int(rb.x2)} {int(rb.y2)} {value}\n")

    print("Finished testing.")

def main():
    """Put the surrounding training code here. The code will probably look very similar to last assignment"""

    transformations = ["bright", "contrast"]

    lr=LEARNING_RATE
    min_iou = MIN_IOU
    weight_bbr_loss = WEIGHT_BBR_LOSS
    anchor_widths = ANCHOR_WIDTHS  # put box widths here
    aspect_ratios = ASPECT_RATIOS  # put aspect ratios here
    train_data_path = r"/cfs/home/o/l/olczyksi/data/mmp-public-3.2/train"  # put path to data here
    train_data_path2 = r"/cfs/home/o/l/olczyksi/data/mmp-public-2.0/mmp2-trainval/train"
    val_data_path = r"/cfs/home/o/l/olczyksi/data/mmp-public-3.2/val"
    test_data_path = r"/cfs/home/o/l/olczyksi/data/mmp-public-3.2/test"
    img_size = IMG_SIZE    # put image size here
    scale_factor = 32 # do not change this since
    grid_size = int(img_size / scale_factor)
    batch_size = 32  # put batch size here
    num_workers = 8  # put number of workers here
    epochs = EPOCHS
    sampling = "random"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # get model, criterion, optimizer, anchor_grid, dataloader
    model = MmpNet(len(anchor_widths), len(aspect_ratios)).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")  # reduction="none" when using mask
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    anchor_grid = get_anchor_grid(grid_size, grid_size, scale_factor, anchor_widths, aspect_ratios)
    anchor_grid_tensor = torch.from_numpy(anchor_grid).float().to(device)
    train_dataloader1 = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False, min_iou, transformations)
    train_dataloader2 = get_dataloader(train_data_path2, img_size, batch_size, num_workers, anchor_grid, False, min_iou, transformations)
    val_dataloader = get_dataloader(val_data_path, img_size, batch_size, num_workers, anchor_grid, False, min_iou, [])
    test_dataloader = get_dataloader(test_data_path, img_size, batch_size, num_workers, anchor_grid, True, min_iou,[])
    writer, log_path = get_tensorboard_writer("MmpNet", "runs")

    print("Anchor grid shape: ", anchor_grid.shape)
    print("Initializing ended, Started Training...")

    train_start = time.time()
    old_loss = 1
    counter = 0
    aps = []
    epochs_final = 0
    max_ap = 0
    for epoch in range(epochs):
        print(">>> Epoch ", epoch + 1, ":")
        epoch_start = time.time()
        model.train()

        current_loss = 0

        for img_batch, lbl_batch, annotation_lists, img_id in train_dataloader1:
            img_batch = img_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            # convert annotation_list to tensor
            annotation_batch = []
            for annotation_list in annotation_lists:
                boxes = torch.tensor([[box.x1, box.y1, box.x2, box.y2] for box in annotation_list], dtype=torch.float32)
                annotation_batch.append(boxes)

            loss = step(model,
                        criterion,
                        optimizer,
                        img_batch,
                        lbl_batch.long(),
                        anchor_grid_tensor,
                        annotation_batch,
                        weight_bbr_loss,
                        sampling)
            current_loss += loss
        scheduler.step()
        average = current_loss / len(train_dataloader1)

        epoch_end = time.time()
        print(f"Epoch {epoch + 1}: Average loss: {average:.4f} ({epoch_end - epoch_start:.2f}s)")
        writer.add_scalar("Loss/train", average, epoch +  1)

        # add augmenations after augementations after a diffent epochs to increase training and stop overfitting
        if epoch + 1 == 10:
            print("Adding augmentations...")
            transformations.append("rotate")
            transformations.append("blur")
            transformations.append("gray")
            train_dataloader1 = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False,
                                              min_iou, transformations)
            # train_dataloader2 = get_dataloader(train_data_path2, img_size, batch_size, num_workers, anchor_grid, False, min_iou, transformations)

        if epoch + 1 == 15:
            print("Removing augmentations...")
            transformations.remove("rotate")
            transformations.remove("blur")
            transformations.remove("gray")
            transformations.remove("bright")
            transformations.remove("contrast")
            print("Adding augmentations...")
            transformations.append("crop")
            transformations.append("flip")
            transformations.append("solar")
            train_dataloader1 = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False, min_iou, transformations)
            # train_dataloader2 = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False,min_iou, transformations)

        if epoch + 1 == 20:
            print("Adding augmentations...")
            transformations.append("flip")
            transformations.append("blur")
            transformations.append("bright")
            train_dataloader1 = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False,
                                              min_iou, transformations)
            # train_dataloader2 = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False, min_iou, transformations)

        if epoch + 1 == 25:
            print("Removing augmentations...")
            transformations.remove("crop")
            transformations.remove("flip")
            transformations.remove("solar")
            transformations.remove("bright")
            transformations.remove("contrast")
            print("Adding augmentations...")
            transformations.append("rotate")
            transformations.append("blur")
            transformations.append("gray")
            train_dataloader1 = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False,
                                              min_iou, transformations)

        if (epoch + 1) % 5 == 0 and epoch != 0 or epoch + 1 == 3:
            ap = evaluate(model, val_dataloader, device, anchor_grid, val_data_path, img_size)
            if ap > max_ap:
                max_ap = ap
            aps.append((epoch + 1, ap))
            writer.add_scalar("AP/validation", ap, epoch + 1)

        if old_loss < average:
            counter += 1
            if counter == 15:
                print(f"Loss increased. Stopping training.")
                break
        else:
            counter = 0
        old_loss = average

        epochs_final = epoch


    # evaluation and testing phase
    print(f"Training ended, took {time.time() - train_start:.2f}s")
    writer.close()
    print(f"Access tensorboard with: tensorboard --logdir={log_path}")

    # evaluation phase
    # ap = evaluate(model, val_dataloader, device, anchor_grid, val_data_path, img_size)
    # aps.append((epochs_final + 1, ap))
    # print(f"Validation final AP: {ap}")
    print(f"Max AP: {max_ap}")

    # plot aps
    epochs = [epoch for epoch, _ in aps]
    ap_values = [ap for _, ap in aps]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ap_values, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Epochs')
    plt.ylabel('average precision (AP)')
    plt.title(f'AP Progression max: {max_ap:.4f}')
    plt.grid(True)
    plt.ylim(0.4, 1)
    plt.show()

    # testing phase
    evaluate_test(model, test_dataloader, device, anchor_grid, test_data_path, img_size)

if __name__ == "__main__":
    main()

