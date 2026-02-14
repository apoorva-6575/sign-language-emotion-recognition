from flask import Flask, Response
from detector import generate_frames

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h2>Sign Language and Emotion Recognition System</h2>
    <img src="/video">
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
