import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torchvision import models, transforms
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = "/oblivion/users/ssegalini/tesi_satellitare"
DATA_FMOW_DIR = os.path.join(BASE_DIR, "data/real")
DATA_SPACENET_DIR = os.path.join(BASE_DIR, "data/spacenet_real")
DATA_FAKE_UNSEEN_DIR = os.path.join(BASE_DIR, "data/fake_test2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Le quote usate DURANTE il training (fMoW, SpaceNet) su un totale di 600
TRAIN_QUOTAS = {
    "fmow_100": (600, 0),
    "spacenet_100": (0, 600),
    "mixed_50_50": (300, 300),
    "mixed_75_25": (450, 150),
    "mixed_25_75": (150, 450)
}

transform = transforms.Compose([
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

# 1. Carica tutti i file sorgente
all_fmow = get_valid_files(DATA_FMOW_DIR)
all_spacenet = get_valid_files(DATA_SPACENET_DIR)
all_fake_unseen = get_valid_files(DATA_FAKE_UNSEEN_DIR)[:600]

assert len(all_fake_unseen) == 600, f"Servono 600 immagini fake unseen in {DATA_FAKE_UNSEEN_DIR}, trovate {len(all_fake_unseen)}"

summary_results = {}

print("=" * 70)
print(" VALUTAZIONE OUT-OF-DISTRIBUTION SUI CAMPIONI NON UTILIZZATI NEL TRAINING")
print("=" * 70)

for exp_name, (used_fmow, used_spacenet) in TRAIN_QUOTAS.items():
    model_path = os.path.join(BASE_DIR, "checkpoints", exp_name, "resnet50_best.pth")
    report_dir = os.path.join(BASE_DIR, "reports", exp_name)
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"\nModello {exp_name} non trovato in {model_path}, salto.")
        continue

    print(f"\n[Test Unseen] Valutazione modello: {exp_name}")

    # 2. Isola i reali MAI visti durante il training per questo specifico modello
    unseen_real = []
    if used_fmow < 600:
        unseen_real.extend(all_fmow[used_fmow:600])
    if used_spacenet < 600:
        unseen_real.extend(all_spacenet[used_spacenet:600])

    # Se un modello ha usato 600 fMoW e 0 SpaceNet, unseen_real conterrà esattamente i 600 di SpaceNet
    assert len(unseen_real) == 600, f"Errore nel conteggio real per {exp_name}: trovati {len(unseen_real)}"

    test_paths = unseen_real + all_fake_unseen
    test_labels = [0] * len(unseen_real) + [1] * len(all_fake_unseen)

    print(f"   Campioni Test: {len(unseen_real)} Reali unseen + {len(all_fake_unseen)} Fake mai visti = {len(test_paths)} totali")

    # 3. Carica il modello
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 4. Inferenza
    test_preds = []
    with torch.no_grad():
        for p in tqdm(test_paths, desc=f"Inference {exp_name}"):
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            pred = torch.argmax(model(tensor), dim=1).item()
            test_preds.append(pred)

    acc = accuracy_score(test_labels, test_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)
    summary_results[exp_name] = (acc, prec, rec, f1)

    # 5. Salva la matrice di confusione del test unseen
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f"Unseen Held-Out Test ({exp_name})\nAcc: {acc:.2f}%")
    plt.ylabel("Classe Reale")
    plt.xlabel("Classe Predetta")
    cm_path = os.path.join(report_dir, "confusion_matrix_unseen.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

# 6. Tabella di sintesi
print("\n" + "=" * 70)
print("RISULTATI FINALI SUI CAMPIONI NON UTILIZZATI (HELD-OUT / CROSS-DOMAIN)")
print("=" * 70)
print(f"{'Modello':<18} | {'Acc (%)':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<9}")
print("-" * 70)
for exp_name, (acc, prec, rec, f1) in summary_results.items():
    print(f"{exp_name:<18} | {acc:<9.2f} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f}")
print("=" * 70)