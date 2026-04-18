import sys, os, json, torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

MODEL_DIR = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(MODEL_DIR, "classes.json"), "r") as f:
    classes = json.load(f)
num_classes = len(classes)

# --- Model definitions ---
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3,3))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self,x): return self.fc(self.conv(x))

class ResNet18Emotion(nn.Module):
    def __init__(self,num_classes=7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )
        self.model = backbone
    def forward(self,x): return self.model(x)

# --- Load weights ---
state = torch.load(os.path.join(MODEL_DIR,"emotion_model.pth"), map_location=device)

# Detect architecture
if any(k.startswith("conv.") for k in state.keys()):
    print("Detected EmotionCNN weights")
    model = EmotionCNN(num_classes=num_classes).to(device)
    transform = T.Compose([
        T.Grayscale(),
        T.Resize((48,48)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
else:
    print("Detected ResNet18Emotion weights")
    model = ResNet18Emotion(num_classes=num_classes).to(device)
    transform = T.Compose([
        T.Resize((48,48)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

model.load_state_dict(state, strict=True)
model.eval()

def predict_emotion(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, pred = torch.max(probs, dim=1)
    return classes[pred.item()], conf.item()*100

# --- Entrypoint ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    emotion, confidence = predict_emotion(image_path)
    print(f"Predicted emotion: {emotion} ({confidence:.1f}%)")
