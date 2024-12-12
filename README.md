# Project Report

## Team

#### Team Name
6501-Group 9  Wake up!

#### Team Members
- Donggen Li (eaz7wk)
- Guang Wu (uzd5zk)
- Linze Fan (fxp3fw)

## Project Information

#### Project Title:
Driver Drowsiness Dectection System on Nvidia Jetson Nano


#### Project Description
The Driver Drowsiness Detection System on the NVIDIA Jetson Nano aims to provide accurate, real-time alerts to prevent accidents caused by driver fatigue. It will leverage efficient, optimized models to detect key drowsiness indicators—such as prolonged eye closure, blinking frequency, and yawning—while ensuring timely response through auditory or visual warnings. Designed to operate effectively within the constraints of the Jetson Nano, the system will demonstrate smooth performance with minimal lag and power efficiency. Additionally, it will be robust enough to handle real-world conditions like varying lighting, driver positions, and potential occlusions, making it suitable for reliable use in practical driving scenarios.

#### Key Objectives
- Build a real-time drowsiness detection system on Nvidia Jetson Nano platform.
- High detection accuracy and reliability.
- Efficient use of edge computing resources.

#### Technology Stack

System: Linux unbuntu 18.04.1.

Hardware platoform: Nvidia Jetson Nano.

Software tools: JetPack SDK (CUDA, cuDNN, TensorRT), OpenCV, dlib, VS Code.

Language: Python

#### Requirements

A custom PyTorch Dataset class (DrowsinessDataset) is implemented to load images and labels. The preprocessing steps include:
- Conbersion of images to grayscale.
- Resizing images to 128*128 pixels.
- Normalization of pixel values to the range [-1,1]
- The dataset is split into training (80%) and validation (20%) sets
