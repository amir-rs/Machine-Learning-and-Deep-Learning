# 🔒 Lock-Unlock Apps Using Face Recognition

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-yellow)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)

A sophisticated facial recognition system that automatically manages application access through real-time face detection and authentication.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Configuration](#configuration) • [Contributing](#contributing)

</div>

![Screenshot](https://github.com/amir-rs/Machine-Learning-and-Deep-Learning/blob/master/Lock-Unlock-Laptop-PC-Screen-Using-Face-Recognition-master/Screenshot%202024-08-31%20211203.png)

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🎯 Real-time Face Detection | Uses YOLO v8 for accurate and fast face recognition |
| 🔐 App Management | Automatic locking/unlocking based on face detection |
| 📊 Custom Dataset Creation | Tools to create your own facial recognition dataset |
| 🤖 Model Training | Built-in functionality to train custom models |
| 🎨 Dark/Light Theme | Customizable UI appearance |
| 📱 Multi-platform | Works on Windows, Linux, and macOS |

## 🔄 System Architecture

```mermaid
graph LR
    A[Webcam Input] --> B[YOLO Face Detection]
    B --> C{Face Recognized?}
    C -->|Yes| D[Unlock Apps]
    C -->|No| E[Keep Locked]
    D --> F[Monitor Apps]
    E --> F
    F --> G{App Launch Attempt?}
    G -->|Yes| B
    G -->|No| F
```

## 🛠️ Installation

### Prerequisites

```mermaid
graph TD
    A[Prerequisites] --> B[Python 3.8+]
    A --> C[Webcam]
    A --> D[GPU Optional]
    B --> E[Dependencies]
    E --> F[OpenCV]
    E --> G[NumPy]
    E --> H[PyAutoGUI]
    E --> I[Ultralytics YOLO]
```

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amir-rs/Lock-Unlock-Laptop-PC-Screen-Using-Face-Recognition-master.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Setup:**
   ```bash
   # Download YOLO model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt
   ```

## 📋 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `face_confidence` | 0.7 | Minimum confidence threshold for face detection |
| `recognition_interval` | 8s | Time between recognition attempts |
| `max_wrong_attempts` | 3 | Maximum failed recognition attempts |
| `unlock_threshold` | 6 | Required successful recognitions to unlock |

## 🚀 Usage

### Basic Usage

```bash
python app.py
```

### Program Flow

```mermaid
sequenceDiagram
    participant User
    participant System
    participant Camera
    participant YOLO
    
    User->>System: Launch Application
    System->>Camera: Initialize Webcam
    loop Face Detection
        Camera->>YOLO: Frame
        YOLO->>System: Detection Results
        System->>System: Process Recognition
    end
    alt Face Recognized
        System->>User: Unlock Apps
    else Face Not Recognized
        System->>User: Keep Locked
    end
```

## 🔧 Advanced Configuration

### Model Training Parameters

```yaml
training:
  epochs: 50
  batch_size: 16
  image_size: 640
  learning_rate: 0.01
  momentum: 0.937
  weight_decay: 0.0005
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check USB connection and permissions |
| Model not loading | Verify YOLO model path and file existence |
| High CPU usage | Adjust detection interval in configuration |
| False detections | Increase face_confidence threshold |
