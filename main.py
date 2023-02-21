from PIL import ImageGrab
import numpy as np
import torch
import cv2
import time
import win32file
import io
from cryptography.fernet import Fernet

from utils.augmentations import letterbox
from utils.general import (non_max_suppression, scale_coords)
from models.experimental import attempt_load


def run():
    rect = (0, 0, 1920, 1080)
    color = (0, 255, 0)
    mid_screen = (rect[2] - rect[0]) >> 1 + (rect[3] - rect[1]) >> 1
    aims = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = device != 'cpu'
    conf_thresh = 0.4
    iou_thresh = 0.05
    model = attempt_load("valorant.pt", device=device, inplace=True, fuse=True)
    stride = max(int(model.stride.max()), 32)
    model.half()
    while True:
        start_time = time.time()
        im = ImageGrab.grab(bbox=rect)
        img0 = np.array(im)
        img = letterbox(img0, stride=stride)[0]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]
        img = img.permute(0, 3, 1, 2)
        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thresh, iou_thresh)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 3)
                    aims.append(((xyxy[0] + xyxy[2]) / 2, (xyxy[3] - xyxy[1]) / 5 + xyxy[1]))
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
            img0 = cv2.resize(img0, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow("display", img0)
            cv2.waitKey(1)
            if len(aims):
                min_dis = 100000
                target_x = 960
                target_y = 540
                for a, b in aims:
                    dis = a + b - mid_screen
                    if dis < min_dis:
                        target_x = a
                        target_y = b
                        min_dis = dis
                print(f"{target_x}, {target_y}")
                aims = []
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")


if __name__ == "__main__":
    run()
