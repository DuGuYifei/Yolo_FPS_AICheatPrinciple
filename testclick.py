import win32api
import time
while True:
    if (win32api.GetAsyncKeyState(0x01) & 0x8000) > 0:
        print("左键")
    if (win32api.GetAsyncKeyState(0x02) & 0x8000) > 0:
        print("右键")
    time.sleep(0.01)