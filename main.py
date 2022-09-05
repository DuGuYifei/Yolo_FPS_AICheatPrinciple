import win32gui
import win32process
from pynput.mouse import Controller
import win32api
import win32con
from PIL import ImageGrab
import numpy as np
import torch
import cv2

from utils.augmentations import letterbox
from utils.general import (non_max_suppression, scale_coords)
from models.experimental import attempt_load

# pt_path = 'valorant-v12.pt'
pt_path = 'valorant-bot.pt'

# used for detecting the hiding window, if use pynput or pyautogui directly, it can be glossed
hd = []
pid = 39408


def run():
    rect = (0, 0, 1920, 1080)
    mid_screen_x = (rect[2] - rect[0]) >> 1
    mid_screen_y = (rect[3] - rect[1]) >> 1
    mid_screen = mid_screen_x + mid_screen_y
    aims = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = device != 'cpu'
    conf_thres = 0.4
    iou_thres = 0.05

    model = attempt_load(pt_path, device=device, inplace=True, fuse=True)
    stride = max(int(model.stride.max()), 32)  # model stride
    model.half()

    open_helper = False
    mouse = Controller()  # 鼠标对象

    while True:
        # middle button of mouse control open and close
        if (win32api.GetAsyncKeyState(0x04) & 0x8000) > 0:
            open_helper = not open_helper
            if open_helper:
                print("Start")
            else:
                print("End")
            cv2.waitKey(1000)

        if ((win32api.GetAsyncKeyState(0x01) & 0x8000) > 0 or (
                win32api.GetAsyncKeyState(0x02) & 0x8000) > 0) and open_helper:
            im = ImageGrab.grab(bbox=rect)
            img0 = np.array(im)
            img = letterbox(img0, stride=stride)[0]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # img = img.unsqueeze(0) 压缩数据维度

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
                        aims.append(((xyxy[0] + xyxy[2]) / 2, (xyxy[3] - xyxy[1]) / 5 + xyxy[1]))

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

                    # mouse.move = (target_x - mid_screen_x, target_y - mid_screen_y)
                    # pyautogui.moveRel(target_x - mid_screen_x, target_y - mid_screen_y, duration=0.01)
                    temp = win32api.MAKELONG(int(target_x), int(target_y))
                    for hwnd in hd:
                        mouse.position = (mid_screen_x, mid_screen_y)
                        if win32api.SendMessage(hwnd, win32con.WM_MOUSEMOVE, 0, temp):  # 鼠标移动
                            print(hwnd)


# used for detect window of game but not work for valorant. They hide window
def foo(hwnd, mouse):
    ttid, ppid = win32process.GetWindowThreadProcessId(hwnd)
    if ppid == pid:
        hd.append(hwnd)
        print(hwnd)


if __name__ == "__main__":
    win32gui.EnumWindows(foo, 0)
    run()
