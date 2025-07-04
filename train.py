import cv2
from ultralytics import YOLO
import torch
import multiprocessing


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=200,
        patience=30,
        imgsz=640,
        batch=16,
        workers=4,
        lr0=0.001,
        optimizer="SGD",
        device="0",
        label_smoothing=0.1,
        weight_decay=0.0005,
        augment=True,
        close_mosaic=20,
        hsv_h=0.01,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.2,
        scale=0.3,
        box=9.0,
        cls=1.5,
        conf=0.4,
        iou=0.6,
        fliplr=0.5,
        flipud=0.2,

        val=True,
        split=0.2,
        save_json=True,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
