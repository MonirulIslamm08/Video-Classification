import os
import numpy as np
import json
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER ='UCF-101'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Keras model
model = load_model("best_model.keras")

# Load class labels from JSON
with open("class_labels.json", "r") as f:
    classes = json.load(f)

def preprocess_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret or len(frames) == 20:  # Limit to 20 frames
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    if not frames:
        return None

    video = np.array(frames)
    video = video.astype("float32") / 255.0
    video = np.expand_dims(video, axis=0)  # Shape: (1, 20, 224, 224, 3)
    return video

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    if request.method == "POST":
        file = request.files["video"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            video = preprocess_video(filepath)

            if video is None:
                label = "Invalid video or too short"
            else:
                preds = model.predict(video)[0]
                label = classes[int(np.argmax(preds))]

    return render_template("index.html", label=label)

if __name__ == "__main__":
    app.run(debug=True)
