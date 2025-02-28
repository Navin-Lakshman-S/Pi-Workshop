import cv2
from flask import Flask, Response

app = Flask(__name__)

# Load the Haar cascade classifier for face detection.
# Make sure 'haarcascade_frontalface_default.xml' is in the same directory.
face_cascade = cv2.CascadeClassifier('haarcascade_alt2.xml')

def generate_frames():
    # Open the default USB webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # print(face_cascade.getFeatureType())
        # Draw rectangles around detected faces.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode the frame in JPEG format.
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in byte format as part of an MJPEG stream.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # The endpoint for streaming video.
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on all network interfaces on port 5000.
    app.run(host='0.0.0.0', port=5000)
