import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import json
import os
from PIL import Image

# -------------------------
# Load classes
# -------------------------
MODEL_DIR = "models"
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(MODEL_DIR, "classes.json"), "r") as f:
    classes = json.load(f)

# -------------------------
# Model — must match train2.py (ResNet18Emotion)
# -------------------------
class ResNet18Emotion(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        backbone = models.resnet18(weights=None)  # match training
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

model = ResNet18Emotion(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "emotion_model.pth"), map_location=device
))
model.eval()
torch.set_num_threads(os.cpu_count())

# -------------------------
# Transform — matches val_transform in training
# -------------------------
transform = T.Compose([
    T.Resize((48, 48)),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   180, 0  ),
    "fear":     (180, 0,   180),
    "happy":    (0,   220, 220),
    "neutral":  (200, 200, 200),
    "sad":      (220, 120, 0  ),
    "surprise": (0,   200, 255),
}

def predict_emotion(face_crop_bgr):
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(face_rgb)
    tensor   = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs      = torch.softmax(model(tensor), dim=1)
        conf, pred = torch.max(probs, dim=1)
    return classes[pred.item()], conf.item() * 100

# -------------------------
# Face detection
# -------------------------
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.75
)

SOURCE = r"E:\EmotionProject\7emotions.mp4"   # change to 0 for webcam
cap    = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame")
        break

    h, w    = frame.shape[:2]
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x    = max(0, int(bbox.xmin  * w))
            y    = max(0, int(bbox.ymin  * h))
            bw   = min(int(bbox.width  * w), w - x)
            bh   = min(int(bbox.height * h), h - y)

            face_crop = frame[y:y+bh, x:x+bw]
            if face_crop.size == 0:
                continue

            emotion, confidence = predict_emotion(face_crop)
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
            label      = f"{emotion}  {confidence:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            pad = 4
            cv2.rectangle(frame,
                (x, y + bh + pad), (x + tw + pad*2, y + bh + th + pad*3), color, -1)
            cv2.putText(frame, label,
                (x + pad, y + bh + th + pad*2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            print(f"Predicted emotion: {emotion}  ({confidence:.1f}%)")

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
