# predict.py
import os
import sys
import json
from collections import deque, Counter

import cv2
import numpy as np
import torch
import mediapipe as mp

# ---------------------------
# Config
# ---------------------------
MODEL_CKPT = "models/checkpoint.pth"        # preferred (contains model_state_dict)
MODEL_STATE = "models/emotion_model.pth"    # fallback plain state_dict
CLASSES_JSON = "models/classes.json"        # saved class order from training

SHOW_TOP3 = True            # show top-3 predictions for debugging
BUFFER_SIZE = 15            # temporal smoothing buffer
CONFIDENCE_THRESHOLD = 0.50 # only add to buffer if model confidence > threshold
ALPHA = 0.3                 # bbox smoothing factor (lower = smoother)

# ---------------------------
# Load labels (must match training order)
# ---------------------------
if os.path.exists(CLASSES_JSON):
    with open(CLASSES_JSON, "r") as f:
        labels = json.load(f)
    print("Loaded labels from classes.json:", labels)
else:
    # fallback: keep a reasonable default but this may be wrong if order differs
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    print("classes.json not found — using fallback labels. Ensure this matches training order.")

num_classes = len(labels)

# ---------------------------
# Model definition (must match training)
# ---------------------------
class EmotionCNN(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            # Block 2
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            # Block 3
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            # Block 4 (extra depth used in training)
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((3, 3))
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 3 * 3, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ---------------------------
# Device and model loading
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EmotionCNN(num_classes=num_classes).to(device)

# Try to load checkpoint or plain state dict
state = None
if os.path.exists(MODEL_CKPT):
    print("Loading checkpoint:", MODEL_CKPT)
    state = torch.load(MODEL_CKPT, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
elif os.path.exists(MODEL_STATE):
    print("Loading state dict:", MODEL_STATE)
    state = torch.load(MODEL_STATE, map_location=device)
else:
    print("No model file found. Expected one of:", MODEL_CKPT, "or", MODEL_STATE)
    sys.exit(1)

# Load with helpful error handling
try:
    model.load_state_dict(state)
    print("Model loaded successfully (strict=True).")
except RuntimeError as e:
    print("RuntimeError while loading state_dict:", e)
    print("Attempting non-strict load to inspect mismatched keys...")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        print("Loaded with strict=False.")
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    except Exception as e2:
        print("Failed to load model even with strict=False:", e2)
        sys.exit(1)

model.eval()

# ---------------------------
# Face detection + smoothing + prediction buffer
# ---------------------------
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

smooth_x = smooth_y = smooth_w = smooth_h = 0.0

pred_buffer = deque(maxlen=BUFFER_SIZE)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

print("Press 'q' to quit. Press 't' to toggle top-3 overlay (currently {})".format(SHOW_TOP3))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Clamp to frame
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            if width < 100 or height < 100:
                continue

            # Smooth bounding box (exponential moving average)
            smooth_x = ALPHA * x + (1 - ALPHA) * smooth_x
            smooth_y = ALPHA * y + (1 - ALPHA) * smooth_y
            smooth_w = ALPHA * width + (1 - ALPHA) * smooth_w
            smooth_h = ALPHA * height + (1 - ALPHA) * smooth_h

            sx = int(smooth_x)
            sy = int(smooth_y)
            sw = int(smooth_w)
            sh = int(smooth_h)

            # Crop face using smoothed coords (safeguard bounds)
            sx = max(0, min(sx, w - 1))
            sy = max(0, min(sy, h - 1))
            sw = max(1, min(sw, w - sx))
            sh = max(1, min(sh, h - sy))

            face = frame[sy:sy + sh, sx:sx + sw]
            if face.size == 0:
                continue

            # Preprocess: grayscale, resize, normalize exactly as training
            face_resized = cv2.resize(face, (48, 48))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_norm = face_gray.astype(np.float32) / 255.0
            face_tensor = torch.tensor(face_norm).unsqueeze(0).unsqueeze(0).to(device)
            face_tensor = (face_tensor - 0.5) / 0.5  # same normalization used during training

            with torch.no_grad():
                output = model(face_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy().flatten()

            top_idx = probs.argsort()[::-1][:3]
            top_probs = probs[top_idx]
            top_labels = [labels[i] for i in top_idx]

            confidence_val = float(top_probs[0])
            pred_label = top_labels[0]

            # Add to buffer only if confident
            if confidence_val > CONFIDENCE_THRESHOLD:
                pred_buffer.append(pred_label)

            # Stable emotion from buffer
            if pred_buffer:
                stable_emotion = Counter(pred_buffer).most_common(1)[0][0]
            else:
                stable_emotion = "..."

            # Draw smoothed box and stable label
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(frame, stable_emotion, (sx, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Confidence text
            conf_text = f"{confidence_val * 100:.0f}%"
            cv2.putText(frame, conf_text, (sx + sw - 60, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # Optional: top-3 debug overlay
            if SHOW_TOP3:
                debug_text = f"{top_labels[0]}:{top_probs[0]:.2f} {top_labels[1]}:{top_probs[1]:.2f} {top_labels[2]}:{top_probs[2]:.2f}"
                cv2.putText(frame, debug_text, (sx, sy + sh + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status
    cv2.putText(frame, "Press q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('t'):
        SHOW_TOP3 = not SHOW_TOP3

cap.release()
cv2.destroyAllWindows()