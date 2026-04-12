import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

# Emotion smoothing variables
prev_emotion = ""
emotion_counter = 0
EMOTION_FRAME_SKIP = 5

def generate_frames():
    global prev_emotion, emotion_counter

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO detection
        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])

            # Only detect person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw person box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # -------- EMOTION DETECTION (IMPROVED) --------
                emotion = prev_emotion

                if emotion_counter % EMOTION_FRAME_SKIP == 0:
                    try:
                        # Crop face/upper body region
                        face = frame[y1:y2, x1:x2]

                        result = DeepFace.analyze(
                            face,
                            actions=['emotion'],
                            enforce_detection=False
                        )

                        emotion = result[0]['dominant_emotion']
                        prev_emotion = emotion

                    except:
                        emotion = prev_emotion

                emotion_counter += 1

                # Display emotion
                cv2.putText(frame, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # -------- GESTURE REGION --------
                upper_body = frame[y1:int((y1 + y2) / 2), x1:x2]

                cv2.rectangle(frame, (x1, y1), (x2, int((y1 + y2) / 2)),
                              (255, 0, 0), 2)

                cv2.putText(frame, "Gesture Region", (x1, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')