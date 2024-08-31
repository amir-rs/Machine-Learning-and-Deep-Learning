import cv2
import sys
import numpy as np
import pyautogui
import ctypes
import os
import datetime
import time
from ultralytics import YOLO

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
counter_correct = 0
counter_wrong = 0

now = datetime.datetime.now()
now = now.second

face_model = YOLO('C:/Users/sepan/Desktop/Lock-Unlock-Laptop-PC-Screen-Using-Face-Recognition-master/yolov8n-face.pt')

cam = cv2.VideoCapture(0)

while True:
    now1 = datetime.datetime.now()
    now1 = now1.second
    if(now1 > now + 8):
        cam.release()
        cv2.destroyAllWindows()
        ctypes.windll.user32.LockWorkStation()
        sys.exit()

    ret, im = cam.read()
    
    results = face_model(im)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            face = im[y1:y2, x1:x2]
            
            # Here you should implement your face recognition logic
            # For example, you can use a pre-trained face recognition model
            # or implement your own logic based on the detected face
            
            # For now, we'll just use a placeholder
            is_recognized = np.random.choice([True, False], p=[0.7, 0.3])
            
            if is_recognized:
                counter_correct += 1
                cv2.putText(im, "Recognized", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                counter_wrong += 1
                cv2.putText(im, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow('Webcam', im)

    if cv2.waitKey(10) & 0xFF == ord('*'):
        break

    if counter_wrong == 3:
        pyautogui.moveTo(48,748)
        pyautogui.click(48,748)
        pyautogui.typewrite("Hello Stranger!!! Whats Up.")
        cam.release()
        cv2.destroyAllWindows()
        ctypes.windll.user32.LockWorkStation()
        sys.exit()

    if counter_correct == 6:
        cam.release()
        cv2.destroyAllWindows()
        sys.exit()

cam.release()
cv2.destroyAllWindows()