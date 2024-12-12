# Project Report

## Team

### Team Name
6501-Group 9  Wake up!

### Team Members
- Donggen Li (eaz7wk)
- Guang Wu (uzd5zk)
- Linze Fan (fxp3fw)

## Project Information

### Project Title:
Driver Drowsiness Dectection System on Nvidia Jetson Nano


### Project Description
The goal of this project is to build a drowsiness detection system that identifies whether a person's eyes are open or closed using real-time video feeds. The system utilizes deep learning models (MobileNetV2, ResNet18, and EfficientNet-B0) and is implemented on the Jetson Nano 
platform. The project also demonstrates model training, evaluation, and real-time deployment with a webcam. 

### Key Objectives
- Build a real-time drowsiness detection system on Nvidia Jetson Nano platform.
- High detection accuracy and reliability.
- Efficient use of edge computing resources.

### Technology Stack
System: Linux unbuntu 18.04.1.
Hardware platoform: Nvidia Jetson Nano.
Software tools: JetPack SDK (CUDA, cuDNN, TensorRT), OpenCV, dlib, VS Code.
Language: Python

#### Nvidia Jetson Nano
Nvidia Jetson Nano is a small, powerful tiny computer designed for edge AI applications. It is ideal for edge AI applications like computer vision, robotics and IoT, allowing real-time inference in devices with limited power and space. It is designed to operate with low power consumption which cost enery as little as 5 watts.

#### Data preprocessing
A custom PyTorch Dataset class (DrowsinessDataset) is implemented to load images and labels. The preprocessing steps include:
- Conversion of images to grayscale.
- Resizing images to 128*128 pixels.
- Normalization of pixel values to the range [-1,1]
- The dataset is split into training(80%) and validation(20%) sets.

### Model Architectures
Three models are used: 
- MobileNetV2: A lightweight model with a modified input layer for grayscale images and a custom output layer for binary classification. 
- ResNet18: A deeper architecture trained similarly to MobileNetV2. 
- EfficientNet-B0: A more efficient model for real-time applications.

### Model Training
The training pipeline includes: 
1. Loading the dataset using DataLoader. 
2. Training the model in batches with gradient updates. 
3. Evaluating the model on the validation set at the end of each epoch. 
4. Saving the trained model as .pth files for deployment.

### Real-time Drowsiness Detection
The trained models are deployed for real-time detection using a webcam. The steps are: 
1. Face and Eye Detection: 
  - Uses OpenCV's Haar cascades to detect faces and eyes. 
2. Eye Classification: 
  - Extracts each detected eye and predicts whether it is open or closed using the trained model. 
3. Visualization: 
  - Draws rectangles around detected eyes and labels them as "Open" or "Closed" with color-coded annotations.

## Platform Setup

### Environment Installation
- Use the official Nvidia JetPack to install required drivers, livraries, and tools.
- Install Python, PyTorch, and OpenCV compatible with Jetson Nano's system.
-  
### IDE
Recommended: VSCode. Install it from the official repositroy to ensure compatibility.

### Dependencies
Install necessary Python library (torch, torchvision, opencv-python, etc.).

### How to work
1. Find dataset: Find a dataset of relevant photos of left and right eye opens and closed eyes.
2. Training: Train models using the provided scripts (mobilenet.py, resent18.py, EfficientNet-B0.py). Adjust parameters (eg., epochs, learning rate) in the scripts as needed.
3. Deployment: Use mobilenet_model.py, resnet_model.py, or efficientnet_B0_model,py to run real-time detection. Connect a webcam and ensure it is recognized by the system.

## Conclusion
This project successfully implements a real-time drowsiness detection system on Jetson Nano. The approach is lightweight, scalable, and can be further optimized for deployment in automotive or healthcare settings.  

## Lessons and Optimization
1.Insufficient data augmentation may cause the model to perform poorly for model-specific scenarios (such as light changes or rapid eye opening and closing).
  - Increase the diversity of the training set: 
    (1). Including data for different lighting conditions (angles, blur, etc.).
    (2). Apply data augmentation techniques (rotation, brightness adjustment, blurring, etc.) to improve model robustness
    
2. Judging from the fluctuations in training and validation losses of ResNet18, the model may be overfitting, especially in epoch 5. Model overfitting may cause the training effect to be out of touch with the actual application

## References
- Adrian Rosebrock, PyImageSearch Blog
- https://github.com/akshaybahadur21/Drowsiness_Detection
