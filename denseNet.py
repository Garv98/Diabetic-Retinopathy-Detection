
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import classification_report, accuracy_score
from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 5
LABEL_FILE = r"C:\Users\garva\Downloads\messidor_cleaned.csv"
DATA_DIR = r"C:\Users\garva\Downloads\messidor_images\messidor-2\images"

class Messidor2Dataset(Dataset):
    def __init__(self, image_paths, labels, augmentations=None):
        self.image_paths = image_paths
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        label = self.labels[idx]
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        return image, label

def load_data(data_dir, label_file):
    data = pd.read_csv(label_file)
    image_paths = [os.path.join(data_dir, img) for img in data['image_id']]
    labels = data['adjudicated_dr_grade'].tolist()
    return image_paths, labels

augmentations = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    RandomBrightnessContrast(p=0.2),
    CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.2),
    ToTensorV2()
])

class DRModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DRModel, self).__init__()
        # Replace with DenseNet or EfficientNet
        self.model = models.densenet121(pretrained=True)  # Or models.efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, total_correct = 0, 0
    for images, labels in tqdm(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():  # Mixed precision training
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)

def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)

def main(data_dir, label_file):
    image_paths, labels = load_data(data_dir, label_file)
    labels = torch.tensor(labels).long()

    train_size = int(0.8 * len(image_paths))
    val_size = len(image_paths) - train_size
    train_paths, val_paths = image_paths[:train_size], image_paths[train_size:]
    train_labels, val_labels = labels[:train_size], labels[val_size:]

    train_dataset = Messidor2Dataset(train_paths, train_labels, augmentations)
    val_dataset = Messidor2Dataset(val_paths, val_labels, augmentations)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = DRModel(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

    torch.save(model.state_dict(), "diabetic_retinopathy_model.pth")

if __name__ == "__main__":
    main(DATA_DIR, LABEL_FILE)
