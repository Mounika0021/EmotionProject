import os
import json
import inspect
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from multiprocessing import freeze_support

# -------------------------
# Config
# -------------------------
USE_SAMPLER      = True     # WeightedRandomSampler to balance class distribution
USE_FOCAL_LOSS   = True     # Focal loss helps with hard-to-learn classes like 'disgust'
USE_PRETRAINED   = True     # Use ResNet18 pretrained on ImageNet (much better than training from scratch)
NUM_WORKERS      = 2
BATCH_SIZE       = 32
EPOCHS           = 40
LR               = 3e-4
MODEL_DIR        = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Transforms
# -------------------------
# NOTE: If using pretrained ResNet, input must be 3-channel (RGB) and normalized with ImageNet stats.
# If USE_PRETRAINED=False, you can use Grayscale + your own stats.
if USE_PRETRAINED:
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),            # slightly larger crop before random crop
        transforms.RandomCrop(48),              # random crop back to 48x48
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        # If images are grayscale, convert to 3-channel by repeating channels
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.RandomCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# -------------------------
# Model definitions
# -------------------------

class EmotionCNN(nn.Module):
    """
    Custom CNN — use when USE_PRETRAINED=False.
    Slightly deeper than the original with squeeze-excitation-like attention.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ResNet18Emotion(nn.Module):
    """
    ResNet18 pretrained backbone — recommended for best accuracy.
    Input: 3-channel 48x48 (ImageNet-normalized).
    """
    def __init__(self, num_classes=7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace final FC
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


# -------------------------
# Focal Loss
# -------------------------
class FocalLoss(nn.Module):
    """
    Focal loss: down-weights easy examples, focuses on hard ones.
    Great for emotion recognition where some classes are easily confused.
    gamma=2.0 is a good default. Combine with class weights for best results.
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# -------------------------
# Mixup augmentation
# -------------------------
def mixup_data(x, y, alpha=0.2, device='cpu'):
    """Mixup: blend two samples to create a new training example."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------
# Main
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("   Emotion Recognition — Improved Training Script")
    print("=" * 60)
    print(f"\nDevice        : {device}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Pretrained    : {USE_PRETRAINED}")
    print(f"Focal loss    : {USE_FOCAL_LOSS}")
    print(f"Sampler       : {USE_SAMPLER}")

    # ---- Datasets ----
    print("\nLoading datasets...")
    train_dataset = torchvision.datasets.ImageFolder("dataset/train", transform=train_transform)

    val_path = "dataset/test" if os.path.exists("dataset/test") else "dataset/val"
    has_val = os.path.exists(val_path)
    if has_val:
        val_dataset = torchvision.datasets.ImageFolder(val_path, transform=val_transform)

    print(f"Train images  : {len(train_dataset)}")
    if has_val:
        print(f"Val images    : {len(val_dataset)}")
    print(f"Classes       : {train_dataset.classes}")

    counts = Counter(train_dataset.targets)
    class_counts = [counts[i] for i in range(len(train_dataset.classes))]
    print("Class counts  :", dict(zip(train_dataset.classes, class_counts)))

    # ---- DataLoaders ----
    if USE_SAMPLER:
        class_counts_arr = np.array(class_counts, dtype=float)
        class_weights = 1.0 / class_counts_arr
        samples_weights = np.array([class_weights[t] for t in train_dataset.targets])
        sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    if has_val:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ---- Model ----
    num_classes = len(train_dataset.classes)
    if USE_PRETRAINED:
        model = ResNet18Emotion(num_classes=num_classes).to(device)
        print("\nUsing ResNet18 pretrained backbone.")
    else:
        model = EmotionCNN(num_classes=num_classes).to(device)
        print("\nUsing custom CNN.")
    print(f"Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss ----
    # FIX: Always pass class weights to the loss, regardless of sampler.
    # The sampler balances *sampling frequency*; the loss weights handle
    # the gradient scaling — they complement each other.
    class_counts_arr = np.array(class_counts, dtype=float)
    class_weights_for_loss = class_counts_arr.sum() / (num_classes * class_counts_arr)
    class_weights_tensor = torch.tensor(class_weights_for_loss, dtype=torch.float32).to(device)
    print(f"Loss weights  : {dict(zip(train_dataset.classes, class_weights_for_loss.round(2)))}")

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # ALWAYS use weights

    # ---- Optimizer ----
    if USE_PRETRAINED:
        # Lower LR for backbone, higher for head
        backbone_params = [p for n, p in model.named_parameters() if 'model.fc' not in n]
        head_params     = [p for n, p in model.named_parameters() if 'model.fc' in n]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': LR * 0.1},
            {'params': head_params,     'lr': LR}
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler_kwargs = dict(mode='min', factor=0.5, patience=4, min_lr=1e-6)
    if 'verbose' in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
        scheduler_kwargs['verbose'] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    # ---- Training loop ----
    best_val_acc = 0.0
    USE_MIXUP = True  # Mixup helps generalize; disable if val acc seems too low

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Mixup augmentation (optional)
            if USE_MIXUP and epoch >= 5:  # Start mixup after 5 epochs of warm-up
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1:02d} | Batch {batch_idx:04d} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        if has_val:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            # Per-class accuracy tracking
            class_correct = [0] * num_classes
            class_total   = [0] * num_classes

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    for c in range(num_classes):
                        mask = (labels == c)
                        class_correct[c] += (predicted[mask] == labels[mask]).sum().item()
                        class_total[c]   += mask.sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total

            # Print per-class accuracy every 5 epochs to spot imbalance early
            if (epoch + 1) % 5 == 0:
                print("\n  Per-class val accuracy:")
                for i, cls in enumerate(train_dataset.classes):
                    if class_total[i] > 0:
                        acc = 100 * class_correct[i] / class_total[i]
                        bar = "#" * int(acc / 5)
                        print(f"    {cls:10s}: {acc:5.1f}%  {bar}")

            prev_lrs = [g['lr'] for g in optimizer.param_groups]
            scheduler.step(avg_val_loss)
            new_lrs = [g['lr'] for g in optimizer.param_groups]
            if new_lrs != prev_lrs:
                print(f"  LR reduced: {prev_lrs} -> {new_lrs}")

            saved_mark = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'classes': train_dataset.classes,
                    'best_val_acc': best_val_acc
                }
                torch.save(checkpoint, os.path.join(MODEL_DIR, "checkpoint.pth"))
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, "emotion_model.pth"))
                with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
                    json.dump(train_dataset.classes, f)
                saved_mark = "  ✓ SAVED (best)"

            print(f"\nEpoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}%"
                  + saved_mark)
        else:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "emotion_model.pth"))
            with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
                json.dump(train_dataset.classes, f)
            print(f"\nEpoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}%")

        print("-" * 60)

    print("\nTraining complete!")
    if has_val:
        print(f"Best Val Accuracy : {best_val_acc:.1f}%")
    print(f"Model saved to    : {os.path.join(MODEL_DIR, 'emotion_model.pth')}")
    print("=" * 60)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "emotion_model.pth"))
with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
    json.dump(train_dataset.classes, f)


if __name__ == "__main__":
    freeze_support()
    main()