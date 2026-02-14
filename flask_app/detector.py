import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np

# Load YOLO model (small and fast)
model = YOLO("yolov8n.pt")

def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # ---------------- YOLO PERSON DETECTION ----------------
        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # Gesture region (upper half of body)
                upper_body = frame[y1:int((y1+y2)/2), x1:x2]

                cv2.putText(frame, "Gesture Region",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0), 2)

        # ---------------- EMOTION DETECTION ----------------
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']

            cv2.putText(frame, emotion,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

        except:
            pass

        # Convert frame to bytes for Flask streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
