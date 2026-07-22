"""Reference-only baseline: ResNet50 / VanillaCNN classifier on the raw (uncropped) dataset.

Consolidated from Colab cells (image3.py-image14.py). This path was explored alongside the
YOLOv8s-cls pipeline in src/train.py but was NOT used for the deployed model in
sih2025/best_saved_model/ -- kept here for comparison/reference only.
"""

import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# =========================
# CONFIG
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(PROJECT_ROOT, "sih2025", "data", "indianbovine1")
MODELS_OUT_DIR = os.path.join(PROJECT_ROOT, "sih2025", "resnet_baseline_models")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def read_classes_csv(csv_path):
    df = pd.read_csv(csv_path)
    breed_cols = [c for c in df.columns if c.lower() != "filename"]
    file2label = {}
    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        one_cols = [c for c in breed_cols if int(row[c]) == 1] if not df[breed_cols].isnull().values.any() else []
        if not one_cols:
            one_cols = [breed_cols[int(np.argmax(row[breed_cols].values))]]
        file2label[fname] = one_cols[0]
    return file2label, breed_cols


class CattleDataset(Dataset):
    def __init__(self, images_dir, csv_path, breed_list, transform=None):
        self.images_dir = Path(images_dir)
        self.file2label, _ = read_classes_csv(csv_path)
        self.filenames = sorted(self.file2label.keys())
        self.breed2idx = {b: i for i, b in enumerate(breed_list)}
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(self.images_dir / fname).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.breed2idx[self.file2label[fname]]
        return img, label


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VanillaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return self.classifier(x)


def get_resnet50(num_classes, pretrained=True, freeze_backbone=False):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(512, num_classes))
    if freeze_backbone:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}; Trainable params: {trainable:,}")


def train_model(model, train_loader, val_loader, breed_list, num_epochs=NUM_EPOCHS, lr=LR,
                 out_dir=MODELS_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = running_corrects = n_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - train")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            n_samples += inputs.size(0)
            pbar.set_postfix(loss=running_loss / n_samples, acc=running_corrects / n_samples)

        epoch_train_loss = running_loss / n_samples
        epoch_train_acc = running_corrects / n_samples

        model.eval()
        val_loss = val_corrects = val_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()
                val_samples += inputs.size(0)

        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects / val_samples

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} -> train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc:.4f}, "
              f"val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_acc)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "breed_list": breed_list,
        }
        torch.save(checkpoint, os.path.join(out_dir, f"model_epoch_{epoch+1}.pth"))

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(checkpoint, os.path.join(out_dir, "best_model.pth"))
            print("Saved new best model.")

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate(model, dataloader, breed_list):
    model = model.to(DEVICE)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            preds = model(inputs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=breed_list, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.show()
    print("Overall accuracy:", (np.array(all_preds) == np.array(all_labels)).mean())
    return all_labels, all_preds, cm


def plot_history(history):
    epochs = len(history["train_loss"])
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="train_loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="val_loss")
    plt.legend(); plt.xlabel("Epoch"); plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["train_acc"], label="train_acc")
    plt.plot(range(1, epochs + 1), history["val_acc"], label="val_acc")
    plt.legend(); plt.xlabel("Epoch"); plt.title("Accuracy")
    plt.show()


def main():
    train_dir = os.path.join(ROOT, "train")
    val_dir = os.path.join(ROOT, "valid")
    test_dir = os.path.join(ROOT, "test")

    _, breed_list = read_classes_csv(os.path.join(train_dir, "_classes.csv"))
    num_classes = len(breed_list)
    print("Num classes:", num_classes)

    train_ds = CattleDataset(train_dir, os.path.join(train_dir, "_classes.csv"), breed_list, train_transforms)
    val_ds = CattleDataset(val_dir, os.path.join(val_dir, "_classes.csv"), breed_list, val_transforms)
    test_ds = CattleDataset(test_dir, os.path.join(test_dir, "_classes.csv"), breed_list, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("Sizes:", len(train_ds), len(val_ds), len(test_ds))

    model = get_resnet50(num_classes=num_classes, pretrained=True, freeze_backbone=False)
    count_params(model)

    trained_model, history = train_model(model, train_loader, val_loader, breed_list)
    plot_history(history)

    best_ckpt = torch.load(os.path.join(MODELS_OUT_DIR, "best_model.pth"), map_location=DEVICE)
    eval_model = get_resnet50(num_classes=num_classes, pretrained=False)
    eval_model.load_state_dict(best_ckpt["model_state_dict"])
    evaluate(eval_model, test_loader, breed_list)


if __name__ == "__main__":
    main()
