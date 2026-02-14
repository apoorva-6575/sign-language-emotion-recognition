# Sign Language and Emotion Recognition System

## Project Overview
This project is a Flask-based web application that performs real-time:

- Person detection using YOLOv8
- Emotion detection using DeepFace
- Gesture region identification for future sign language integration

The system captures webcam input, processes each frame, and streams the result to a web interface.

---

## Technologies Used

- Python
- Flask
- OpenCV
- YOLOv8 (Ultralytics)
- DeepFace
- TensorFlow

---

## Current Implementation (Phase 1)

✔ Real-time webcam streaming using Flask  
✔ Person detection using YOLOv8  
✔ Emotion detection using DeepFace  
✔ Gesture region extraction logic  
✔ Modular architecture (`flask_app/` structure)

This phase establishes the system architecture.

---

## Future Scope (Phase 2)

- MediaPipe hand landmark detection  
- Custom sign language classifier training  
- Dataset creation and model evaluation  
- Accuracy metrics and performance analysis  

---

## How to Run

1. Create virtual environment:
   python -m venv venv

2. Activate:
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run application:
   python flask_app/app.py

5. Open browser:
   http://127.0.0.1:5000

---

## Author
Apoorva Kala
