# Sign Language and Emotion Recognition System

## Project Overview

This project is a real-time computer vision system designed to detect hand gestures (Sign Language) and recognize facial emotions using deep learning models. The goal of the system is to assist specially-abled children by improving communication through intelligent automation.

The system captures live video from a webcam, detects hands using YOLO-based object detection, and analyzes facial expressions using an emotion recognition model.

---

## Key Features

- Real-time hand detection using YOLO
- Facial emotion recognition from live webcam feed
- Integrated sign and emotion processing pipeline
- Modular Python project structure
- Flask-based web application interface

---

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- DeepFace
- Flask
- PyTorch

---

## Installation and Setup

1. Clone the repository:

   git clone https://github.com/apoorva-6575/sign-language-emotion-recognition.git

2. Navigate into the project folder:

   cd sign-language-emotion-recognition

3. Install required dependencies:

   pip install -r requirements.txt

4. Run the application:

   python main.py

---

## Project Structure

sign-language-emotion-recognition/
│
├── flask_app/
├── emotion_detect.py
├── hand_detect.py
├── main.py
├── requirements.txt
└── README.md

---

## Future Enhancements

- Improve gesture classification accuracy
- Optimize model performance for faster inference
- Add voice output for detected signs
- Deploy the system as a web-based application

---

## Author

Apoorva Kala  
PBL Project – Computer Science Engineering
