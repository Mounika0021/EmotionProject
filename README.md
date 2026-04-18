# 🎭 Emotion Recognition Pipeline

A robust, real-time emotion recognition system built with PyTorch that detects faces and classifies them into **7 emotions**: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise` — across webcam streams, video files, and static images.

---

## 🚀 Features

- **ResNet18 backbone** (pretrained on ImageNet) for high accuracy, with optional fallback to a custom CNN
- **Multi-source inference** — works on live webcam, video files, and images
- **Temporal smoothing** — exponential moving average on bounding boxes + prediction buffering to reduce flickering
- **Class imbalance handling** — WeightedRandomSampler + Focal Loss targets hard classes like *disgust*
- **Mixup augmentation** applied post-warm-up for better generalization
- **Top-3 predictions** displayed with confidence scores during live inference

---

## 📁 Project Structure

```
├── train2.py              # Training script (ResNet18 / custom CNN)
├── predict.py             # Live webcam inference (Mediapipe face detection)
├── predict1.py            # Image / video file inference
├── models/
│   ├── emotion_model.pth  # Trained model weights
│   ├── checkpoint.pth     # Training checkpoint
│   └── classes.json       # Class label mapping
├── dataset/
│   ├── train/             # Training images (organized by emotion class)
│   └── val/               # Validation images
└── test_images/           # Sample images for evaluation
```

---

## ⚙️ Setup

### Prerequisites

```bash
pip install torch torchvision mediapipe opencv-python
```

### Clone & Install

```bash
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train2.py
```

**What happens during training:**
- Loads dataset from `dataset/train/` and `dataset/val/`
- Applies augmentation: resize, random crop, horizontal flip, rotation, color jitter, grayscale
- Uses `WeightedRandomSampler` to balance class distribution
- Trains with Focal Loss (or weighted cross-entropy) for hard class handling
- Applies Mixup augmentation after warm-up epochs
- Saves weights to `models/emotion_model.pth` and labels to `models/classes.json`

---

## 🔍 Inference

### 📷 Webcam (Live)

```bash
python predict.py
```

- Uses **Mediapipe** for fast, accurate face detection
- Smooths bounding boxes with exponential moving average
- Buffers predictions across frames to stabilize output
- Displays **Top-3 emotions** with confidence scores in real time

### 🖼️ Image / Video File

```bash
python predict1.py --input path/to/image_or_video
```

- Auto-detects whether weights are from a CNN or ResNet model
- Applies correct preprocessing (grayscale for CNN, ImageNet normalization for ResNet)
- Outputs predicted emotion and confidence score

---

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| Backbone | ResNet18 (pretrained ImageNet) |
| Output classes | 7 emotions |
| Loss function | Focal Loss / Weighted Cross-Entropy |
| Augmentation | Flip, Rotation, Jitter, Mixup |
| Face detector | Mediapipe (webcam), OpenCV (image/video) |

---

## 📊 Emotion Classes

| Label | Description |
|-------|-------------|
| 😠 angry | Anger / frustration |
| 🤢 disgust | Disgust (hard minority class) |
| 😨 fear | Fear / anxiety |
| 😊 happy | Happiness / joy |
| 😐 neutral | No strong emotion |
| 😢 sad | Sadness |
| 😲 surprise | Surprise / shock |

---

## 🔄 Workflow Summary

```
Train (train2.py)
      ↓
  models/emotion_model.pth + classes.json
      ↓
  ┌───────────────┬──────────────────────┐
  ↓               ↓                      ↓
Webcam         Image                  Video
predict.py    predict1.py           predict1.py
```

> **Train once, predict anywhere** — webcam, video, or image — with stable, smoothed emotion outputs.

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) — deep learning framework
- [MediaPipe](https://mediapipe.dev/) — real-time face detection
- [FER / AffectNet datasets](https://paperswithcode.com/datasets?q=emotion) — emotion recognition benchmarks
