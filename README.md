# Simple ISP Tuning Tool

## Overview
Welcome to my **Simple ISP Tuning Tool**! This project is a product of my deep passion for image processing and ISP (Image Signal Processing) tuning. 
It’s a Python-based application designed to provide an intuitive, real-time platform for tweaking image processing parameters and visualizing their effects. 
Whether you're a student or a professional ( or simply someone who like photography 😇 ) looking to explore ISP pipelines, this tool offers a hands-on way to enhance images and understand the underlying techniques.
The core of this project is an easy to use **Graphical User Interface (GUI)** built with PyQt5, making it easy to load images, adjust settings, and see results instantly. 
From basic adjustments like brightness and blur to advanced feature extraction like ORB keypoints and Canny edges, this tool bridges the gap between theory and practice in image processing.

## Features
- **Interactive GUI:** Load an image and tweak ISP parameters in real-time using sliders, checkboxes, and spin boxes.
- **Real-Time Processing:** See changes instantly with multi-threaded processing to keep the UI responsive.
- **ISP Adjustments:** Includes tools for histogram gain, Gaussian blur, RGB gain, white balance, noise reduction, gamma correction, lens shading, sharpening, tone mapping, and Fourier filtering.
- **Feature Extraction:** Visualize ORB keypoints (with count and strength metrics) and Canny edges (with density metrics) to analyze image structure.
- **Dynamic Zoom:** Click and drag on the image to see a magnified crop of any region, perfect for inspecting keypoints or edges up close.
- **Preset System:** Save and load your custom ISP settings as JSON files.
- **Histogram Visualization:** Compare histograms of original and processed images side-by-side.
- **Frequency Domain Analysis:** Apply Fourier filters and view frequency domain representations.

## Why I Built This
Image processing and ISP tuning are more than just technical skills to me—they’re passions that let me combine creativity and engineering. 
I’ve always been fascinated by how raw sensor data transforms into stunning visuals through careful tuning. 
This project started as a way to further deepen my understanding of ISP pipelines and feature extraction, and it evolved into a portfolio piece to showcase my skills. 
The GUI was a deliberate choice to make the tool accessible and engaging.

## Libraries Used
This project leverages a powerful stack of Python libraries:
- **PyQt5**: For the responsive and polished GUI.
- **OpenCV (cv2)**: The backbone of image processing, handling everything from basic adjustments to feature detection.
- **NumPy**: For efficient array operations and image manipulations.
- **JSON**: For saving and loading presets.
- **Python Standard Library (os, sys)**: For file handling and application management.

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/IdoOk-collab/S-ISP.git
   cd simple-isp-tuning-tool
