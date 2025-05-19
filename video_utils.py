import cv2
import numpy as np

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        resized = cv2.resize(frame, (64, 64))  # adjust to your model input size
        frames.append(resized)
        success, frame = cap.read()
    cap.release()

    # Example: average frame pixel values
    features = np.mean(frames, axis=0).flatten()
    return features.reshape(1, -1)
