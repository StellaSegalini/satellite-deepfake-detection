import io
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# --- CONFIGURAZIONE SORGENTI ---
BASE_DIR = "/oblivion/users/ssegalini/tesi_satellitare"
DATA_FMOW_DIR = os.path.join(BASE_DIR, "data/real")
DATA_SPACENET_DIR = os.path.join(BASE_DIR, "data/spacenet_real")
DATA_FAKE_DIR = os.path.join(BASE_DIR, "data/fake")

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definizione dei 5 esperimenti: (quota_fmow, quota_spacenet)
EXPERIMENTS = {
    "fmow_100": (600, 0),
    "spacenet_100": (0, 600),
    "mixed_50_50": (300, 300),
    "mixed_75_25": (450, 150),
    "mixed_25_75": (150, 450)
}

# --- DATA AUGMENTATION ---
class RandomJPEGCompression:
    def __init__(self, quality_min=50, quality_max=90, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(self.quality_min, self.quality_max)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf)
        return img

class SatelliteDatasetFromPaths(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    RandomJPEGCompression(quality_min=50, quality_max=90, p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_valid_files(directory):
    valid_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_paths.append(os.path.join(root, f))
    valid_paths.sort()
    return valid_paths

# Scansione preliminare delle sorgenti
all_fmow = get_valid_files(DATA_FMOW_DIR)
all_spacenet = get_valid_files(DATA_SPACENET_DIR)
all_fake = get_valid_files(DATA_FAKE_DIR)[:600]

assert len(all_fmow) >= 600, f"Mancano campioni fMoW: trovati {len(all_fmow)}"
assert len(all_spacenet) >= 600, f"Mancano campioni SpaceNet: trovati {len(all_spacenet)}"
assert len(all_fake) == 600, f"Mancano campioni Fake: trovati {len(all_fake)}"

# --- ESECUZIONE CICLICA DEI 5 ESPERIMENTI ---
for exp_name, (n_fmow, n_spacenet) in EXPERIMENTS.items():
    print(f"\n{'='*60}")
    print(f" AVVIO TRAINING: {exp_name} (fMoW: {n_fmow}, SpaceNet: {n_spacenet}, Fake: 600)")
    print(f"{'='*60}")

    # Fissaggio seed prima di ogni run per garantire riproducibilità
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    model_save_path = os.path.join(ckpt_dir, "resnet50_best.pth")

    selected_real = all_fmow[:n_fmow] + all_spacenet[:n_spacenet]
    selected_fake = all_fake

    all_paths = selected_real + selected_fake
    all_labels = [0] * len(selected_real) + [1] * len(selected_fake)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=all_labels
    )

    train_loader = DataLoader(
        SatelliteDatasetFromPaths(train_paths, train_labels, train_transforms),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        SatelliteDatasetFromPaths(val_paths, val_labels, val_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_paths)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_paths)
        acc = accuracy_score(all_targets, all_preds)

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), model_save_path)

    print(f"Esperimento '{exp_name}' completato. Miglior Val Accuracy: {best_val_acc * 100:.2f}%")