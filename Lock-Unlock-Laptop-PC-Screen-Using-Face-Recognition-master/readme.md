# Lock-Unlock Screen Using Face Recognition

This project allows you to automatically lock or unlock your laptop/PC screen based on face recognition. Using a webcam, the system detects your face and performs the desired action (lock/unlock) using a trained YOLO model.

## Features

- **Real-time Face Detection**: Uses YOLO for accurate and fast face recognition.
- **Screen Lock/Unlock**: Automatically locks or unlocks the screen based on the detected face.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/amir-rs/Lock-Unlock-Laptop-PC-Screen-Using-Face-Recognition-master.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the YOLO model and place it in the correct directory:
    - Ensure the model file `yolov8n-face.pt` is placed in the appropriate folder as specified in the code.



![Screenshot](https://github.com/amir-rs/Machine-Learning-and-Deep-Learning/blob/master/Lock-Unlock-Laptop-PC-Screen-Using-Face-Recognition-master/Screenshot%202024-08-31%20211203.png)

## Usage

To start the program, simply run the Python script:

```bash
python lock_unlock_face_recognition.py
Ensure your webcam is connected and working. The program will start detecting faces and perform the lock/unlock actions accordingly.

Dependencies
Python 3.x
OpenCV (cv2)
NumPy
pyautogui
YOLO model
You can install the dependencies using:

bash
pip install opencv-python-headless numpy pyautogui

License
This project is licensed under the MIT License - see the LICENSE file for details.
