import cv2
from deepface import DeepFace
from collections import deque

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Strong smoothing
emotion_history = deque(maxlen=20)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            try:
                result = DeepFace.analyze(
                    face,
                    actions=['emotion'],
                    enforce_detection=True
                )

                emotions = result[0]['emotion']
                dominant = result[0]['dominant_emotion']
                confidence = emotions[dominant]

                # 🔥 Ignore low confidence
                if confidence < 60:
                    continue

                # 🔥 Neutral correction (VERY IMPORTANT)
                if dominant in ['fear', 'sad'] and emotions['neutral'] > 40:
                    dominant = 'neutral'

                emotion_history.append(dominant)

                # 🔥 Strong smoothing (mode of history)
                stable_emotion = max(set(emotion_history), key=emotion_history.count)

                cv2.putText(frame, stable_emotion,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2)

            except:
                pass

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')