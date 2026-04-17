import os
import json
import inspect
from collections import Counter

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from multiprocessing import freeze_support

# -------------------------
# Config
# -------------------------
USE_SAMPLER = True          # True = use WeightedRandomSampler, False = use shuffle + optional class-weighted loss
USE_FOCAL_LOSS = False      # True = use focal loss (requires class_weights if desired)
NUM_WORKERS = 2            # set to 0 if you still see multiprocessing issues on your system
BATCH_SIZE = 32
EPOCHS = 25
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
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
# Model definition
# -------------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# -------------------------
# Optional focal loss
# -------------------------
class FocalLoss(nn.Module):
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
        else:
            return loss

# -------------------------
# Main training function
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("   Emotion CNN — Final Training Script")
    print("=" * 55)
    print(f"\nUsing device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = torchvision.datasets.ImageFolder("dataset/train", transform=train_transform)

    val_path = "dataset/test"
    if not os.path.exists(val_path):
        val_path = "dataset/val"

    if os.path.exists(val_path):
        val_dataset = torchvision.datasets.ImageFolder(val_path, transform=val_transform)
        has_val = True
    else:
        has_val = False

    print(f"Train images : {len(train_dataset)}")
    if has_val:
        print(f"Val   images : {len(val_dataset)}")
    print(f"Classes      : {train_dataset.classes}")

    # Compute class counts
    counts = Counter(train_dataset.targets)
    print("Class counts:", counts)
    class_counts = [counts[i] for i in range(len(train_dataset.classes))]

    # Create DataLoader (optionally with WeightedRandomSampler)
    if USE_SAMPLER:
        import numpy as np
        class_counts_arr = np.array(class_counts)
        class_weights = 1.0 / class_counts_arr
        samples_weights = np.array([class_weights[t] for t in train_dataset.targets])
        sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    if has_val:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model, optimizer, scheduler
    num_classes = len(train_dataset.classes)
    model = EmotionCNN(num_classes=num_classes).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for loss (useful if not using sampler or as extra weighting)
    import numpy as np
    class_counts_arr = np.array(class_counts)
    class_weights_for_loss = class_counts_arr.sum() / (len(class_counts_arr) * class_counts_arr)
    class_weights_tensor = torch.tensor(class_weights_for_loss, dtype=torch.float32).to(device)

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
    else:
        # If using sampler, you can still use class weights or plain CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor if not USE_SAMPLER else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler_kwargs = dict(mode='min', factor=0.5, patience=3)
    if 'verbose' in inspect.signature(ReduceLROnPlateau).parameters:
        scheduler_kwargs['verbose'] = True
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

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
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total

            prev_lrs = [g['lr'] for g in optimizer.param_groups]
            scheduler.step(avg_val_loss)
            new_lrs = [g['lr'] for g in optimizer.param_groups]
            if new_lrs != prev_lrs:
                print(f"LR reduced: {prev_lrs} -> {new_lrs}")

            # Save best model and checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = avg_val_loss

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'classes': train_dataset.classes,
                    'best_val_acc': best_val_acc
                }
                torch.save(checkpoint, os.path.join(MODEL_DIR, "checkpoint.pth"))
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, "emotion_model.pth"))

                # Save classes.json for inference
                with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
                    json.dump(train_dataset.classes, f)

                saved_mark = "  ✓ SAVED (best)"
            else:
                saved_mark = ""

            print(f"\nEpoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}%"
                  + saved_mark)
        else:
            # No validation — save every epoch
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "emotion_model.pth"))
            with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
                json.dump(train_dataset.classes, f)
            print(f"\nEpoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}%")

        print("-" * 55)

    print("\nTraining complete!")
    if has_val:
        print(f"Best Val Accuracy : {best_val_acc:.1f}%")
        print(f"Best Val Loss     : {best_val_loss:.4f}")
    print("Model saved to    :", os.path.join(MODEL_DIR, "emotion_model.pth"))
    print("=" * 55)

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    # On Windows, freeze_support helps when using multiprocessing
    freeze_support()
    main()