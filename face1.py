import cv2
import mediapipe as mp
import torch
import torchvision.transforms as T
import json
import os
from PIL import Image

# -------------------------
# Load trained model + classes
# -------------------------
MODEL_DIR = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(MODEL_DIR, "classes.json"), "r") as f:
    classes = json.load(f)

class EmotionCNN(torch.nn.Module):
    def __init__(self, num_classes=len(classes)):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

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

model = EmotionCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "emotion_model.pth"), map_location=device))
model.eval()

transform = T.Compose([
    T.Grayscale(),
    T.Resize((48,48)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

def predict_emotion(face_crop):
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (48,48))
    tensor = transform(Image.fromarray(face_resized)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return classes[pred]

# -------------------------
# Face detection setup
# -------------------------
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

SOURCE = r"E:\EmotionProject\7emotions.mp4"
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

cv2.namedWindow("Face Detection + Emotion", cv2.WINDOW_NORMAL)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            x = max(0, x); y = max(0, y)
            width = min(width, w - x); height = min(height, h - y)

            face_crop = frame[y:y+height, x:x+width]
            if face_crop.size == 0:
                continue

            emotion = predict_emotion(face_crop)
            print("Predicted emotion:", emotion)  # also log to terminal
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y+height+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Detection + Emotion", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
