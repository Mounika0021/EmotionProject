import os, sys, json
from collections import deque, Counter
import cv2, torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import mediapipe as mp

MODEL_CKPT   = "models/checkpoint.pth"
MODEL_STATE  = "models/emotion_model.pth"
CLASSES_JSON = "models/classes.json"

SHOW_TOP3 = True
BUFFER_SIZE = 15
CONFIDENCE_THRESHOLD = 0.30
ALPHA = 0.3

# Load labels
if os.path.exists(CLASSES_JSON):
    with open(CLASSES_JSON,"r") as f: labels = json.load(f)
else:
    labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
num_classes = len(labels)

# Define both models
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load state dict
if os.path.exists(MODEL_CKPT):
    state=torch.load(MODEL_CKPT,map_location=device)
    if isinstance(state,dict) and 'model_state_dict' in state:
        state=state['model_state_dict']
elif os.path.exists(MODEL_STATE):
    state=torch.load(MODEL_STATE,map_location=device)
else:
    print("No model file found."); sys.exit(1)

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

# Face detection
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

smooth_x=smooth_y=smooth_w=smooth_h=0.0
pred_buffer=deque(maxlen=BUFFER_SIZE)

SOURCE = 0 if len(sys.argv)<2 else sys.argv[1]
cap=cv2.VideoCapture(SOURCE)
if not cap.isOpened(): print("Cannot open source"); sys.exit(1)

print("Press q to quit. Press t to toggle top-3 overlay.")

while cap.isOpened():
    ret,frame=cap.read()
    if not ret: break
    h,w,_=frame.shape
    results=detector.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    if results.detections:
        for det in results.detections:
            bbox=det.location_data.relative_bounding_box
            x,y=int(bbox.xmin*w),int(bbox.ymin*h)
            bw,bh=int(bbox.width*w),int(bbox.height*h)
            x=max(0,x); y=max(0,y)
            bw=min(bw,w-x); bh=min(bh,h-y)
            if bw<60 or bh<60: continue

            # Smooth bbox
            smooth_x=ALPHA*x+(1-ALPHA)*smooth_x
            smooth_y=ALPHA*y+(1-ALPHA)*smooth_y
            smooth_w=ALPHA*bw+(1-ALPHA)*smooth_w
            smooth_h=ALPHA*bh+(1-ALPHA)*smooth_h
            sx,sy,sw,sh=int(smooth_x),int(smooth_y),int(smooth_w),int(smooth_h)

            face=frame[sy:sy+sh,sx:sx+sw]
            if face.size==0: continue

            pil_img=Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB))
            tensor=transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs=torch.softmax(model(tensor),dim=1).cpu().numpy().flatten()

            top_idx=probs.argsort()[::-1][:3]
            top_probs=probs[top_idx]; top_labels=[labels[i] for i in top_idx]
            conf_val=float(top_probs[0]); pred_label=top_labels[0]

            if conf_val>CONFIDENCE_THRESHOLD: pred_buffer.append(pred_label)
            stable_emotion=Counter(pred_buffer).most_common(1)[0][0] if pred_buffer else "..."

            cv2.rectangle(frame,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
            cv2.putText(frame,f"{stable_emotion} {conf_val*100:.0f}%",(sx,sy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

            if SHOW_TOP3:
                debug=f"{top_labels[0]}:{top_probs[0]:.2f} {top_labels[1]}:{top_probs[1]:.2f} {top_labels[2]}:{top_probs[2]:.2f}"
                cv2.putText(frame,debug,(sx,sy+sh+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    cv2.imshow("Emotion Detection",frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    if key==ord('t'): SHOW_TOP3=not SHOW_TOP3

cap.release(); cv2.destroyAllWindows()
