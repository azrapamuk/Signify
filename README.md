# Signify - American Sign Language (ASL) Recognition Application

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-blue.svg)](https://pypi.org/project/opencv-python/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-blue.svg)](https://pypi.org/project/mediapipe/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-blue.svg)](https://scikit-learn.org/stable/)

## Description

**Signify** is a Python application that explores the principles of computer vision and machine learning with a practical implementation for sign language recognition. Although currently focused on recognizing **American Sign Language (ASL)**, the application is designed as a foundation for further development in recognizing hand gestures and real-time interactions. 

This application uses advanced techniques for hand movement analysis, offering wide possibilities for implementation in various domains such as **interactive systems**, **gesture interfaces**, and **social interactions**.

## Key Features

- **Real-time sign and gesture recognition:** Uses the camera to analyze hand movements and recognize gestures, enabling a broad range of applications beyond ASL.
- **Python-based:** Developed in Python using popular image processing and machine learning libraries, making it easy to extend the application for new tasks.
- **Using MediaPipe:** For precise hand detection and key point tracking, ensuring high accuracy in recognizing gestures and movements.
- **Machine Learning Model:** A trained model that not only recognizes ASL signs but can also be adapted to different types of gestures and hand signals.

## Instructions

1. **Run `imageCollection.py`:** First, run this script to collect all necessary images for recognizing different classes.
2. **Run `datasetCreation.py`:** Then, run this script to create the dataset that will be used for training the classifier.
3. **Train the classifier with `classifierTraining.py`:** Using the previously collected data, this script trains a model to recognize signs.
4. **Run `finalProgram.py`:** Finally, everything is integrated into one program using `finalProgram.py`, bringing everything together into a functional application.
