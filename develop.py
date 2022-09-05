from pynput.mouse import Controller
import win32api
from PIL import ImageGrab
import numpy as np
import torch
import cv2

from utils.augmentations import letterbox
from utils.general import (non_max_suppression, scale_coords)
from models.experimental import attempt_load


def run():
    rect = (0, 0, 1920, 1080)
    color = (0, 255, 0)  # RGB  用来画框的颜色
    mid_screen = (rect[2] - rect[0]) >> 1 + (rect[3] - rect[1]) >> 1
    aims = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = device != 'cpu'
    conf_thres = 0.4
    iou_thres = 0.05
    mouse = Controller()  # 鼠标对象

    model = attempt_load('valorant-v12.pt', device=device, inplace=True, fuse=True)

    stride = max(int(model.stride.max()), 32)  # model stride

    model.half()
    while True:
        #if (win32api.GetAsyncKeyState(0x01) & 0x8000) > 0 or (win32api.GetAsyncKeyState(0x02) & 0x8000) > 0:
            im = ImageGrab.grab(bbox=rect)
            img0 = np.array(im)
            img = letterbox(img0, stride=stride)[0]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # 压缩数据维度

            img = img.permute(0, 3, 1, 2)
            pred = model(img, augment=False, visualize=False)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            # Process predictions (based on source code of yolo)
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 3)
                        aims.append(((xyxy[0] + xyxy[2]) / 2, (xyxy[3] - xyxy[1]) / 5 + xyxy[1]))
                cv2.imshow("display", img0)
                cv2.waitKey(1)

                # # 找到最近的目标
                # if len(aims):
                #     min_dis = 100000
                #     target_x = 960
                #     target_y = 540
                #     for a, b in aims:
                #         dis = a + b - mid_screen
                #         if dis < min_dis:
                #             target_x = a
                #             target_y = b
                #             min_dis = dis
                #
                #     aims = []
                #
                #     # 移动鼠标
                #     mouse.position = (target_x, target_y)


if __name__ == "__main__":
    run()
