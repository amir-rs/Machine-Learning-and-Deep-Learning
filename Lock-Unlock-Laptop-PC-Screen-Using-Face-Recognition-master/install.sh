#!/bin/bash

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-tk python3-dev
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y gnome-screensaver
sudo apt-get install -y python3-xlib

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download YOLO model
python3 test.py

echo "Installation completed!"