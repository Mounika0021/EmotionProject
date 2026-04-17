import sys
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

# -------------------------
# Define model (same as training)
# -------------------------
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
        return self.fc(self.conv(x))

# -------------------------
# Load weights
# -------------------------
model = EmotionCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "emotion_model.pth"), map_location=device))
model.eval()

# -------------------------
# Preprocessing
# -------------------------
transform = T.Compose([
    T.Grayscale(),
    T.Resize((48,48)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

def predict_emotion(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return classes[pred]

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    print("Predicted emotion:", predict_emotion(image_path))
