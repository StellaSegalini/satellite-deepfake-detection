import io
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from tqdm import tqdm

BASE_DIR = "/oblivion/users/ssegalini/tesi_satellitare"
DATA_FMOW_DIR = os.path.join(BASE_DIR, "data/real")
DATA_SPACENET_DIR = os.path.join(BASE_DIR, "data/spacenet_real")
DATA_FAKE_DIR = os.path.join(BASE_DIR, "data/fake")

VAL_SPLIT = 0.2
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
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

def apply_jpeg_compression(img_pil, quality):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def get_valid_files(directory):
    valid_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_paths.append(os.path.join(root, f))
    valid_paths.sort()
    return valid_paths

all_fmow = get_valid_files(DATA_FMOW_DIR)
all_spacenet = get_valid_files(DATA_SPACENET_DIR)
all_fake = get_valid_files(DATA_FAKE_DIR)[:600]

summary_metrics = {}

for exp_name, (n_fmow, n_spacenet) in EXPERIMENTS.items():
    print(f"\n=======================================================")
    print(f" VALUTAZIONE E GENERAZIONE REPORT: {exp_name}")
    print(f"=======================================================")

    model_path = os.path.join(BASE_DIR, "checkpoints", exp_name, "resnet50_best.pth")
    report_dir = os.path.join(BASE_DIR, "reports", exp_name)
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Checkpoint non trovato per {exp_name} in {model_path}, salto.")
        continue

    # Inizializzazione pulita del modello
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Ricostruzione esatta dei percorsi usati in fase di addestramento
    selected_real = all_fmow[:n_fmow] + all_spacenet[:n_spacenet]
    all_paths = selected_real + all_fake
    all_labels = [0] * len(selected_real) + [1] * len(all_fake)

    # Split con seed fisso: isola le medesime 240 immagini di validation
    _, val_paths, _, val_labels = train_test_split(
        all_paths, all_labels, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=all_labels
    )

    print(f"Campioni di Validazione: {len(val_paths)} (Real: {val_labels.count(0)}, Fake: {val_labels.count(1)})")

    # --- 1. INFERENZA SUL VALIDATION SET & MATRICE DI CONFUSIONE ---
    val_preds = []
    for p in val_paths:
        img = Image.open(p).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.argmax(model(tensor), dim=1).item()
        val_preds.append(pred)

    acc = accuracy_score(val_labels, val_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)
    summary_metrics[exp_name] = (acc, prec, rec, f1)

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f"Confusion Matrix ({exp_name})\nAcc: {acc:.2f}%")
    plt.ylabel("Classe Reale")
    plt.xlabel("Classe Predetta")
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    # --- 2. GRAD-CAM SU IMMAGINE FAKE CASUALE ---
    val_fake_paths = [p for p, l in zip(val_paths, val_labels) if l == 1]
    
    # Resetta l'entropia casuale per scegliere un campione arbitrario e non vincolato al seed
    random.seed()
    chosen_fake_path = random.choice(val_fake_paths)
    filename_sample = os.path.basename(chosen_fake_path)
    print(f"Grad-CAM calcolata sul campione fake estratto: {filename_sample}")

    raw_img = Image.open(chosen_fake_path).convert("RGB").resize((512, 512))
    cam = GradCAM(model, model.layer4).generate_heatmap(transform(raw_img).unsqueeze(0).to(DEVICE), class_idx=1)
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(raw_img), 0.6, heatmap_colored, 0.4, 0)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(raw_img)
    ax[0].axis('off')
    ax[0].set_title(f"Input: {filename_sample}", fontsize=9)

    ax[1].imshow(cam, cmap='jet')
    ax[1].axis('off')
    ax[1].set_title("Grad-CAM Heatmap", fontsize=10)

    ax[2].imshow(overlay)
    ax[2].axis('off')
    ax[2].set_title("Overlay Attivazione", fontsize=10)

    plt.savefig(os.path.join(report_dir, "gradcam.png"), bbox_inches='tight')
    plt.close()

    # --- 3. STRESS TEST ALLA COMPRESSIONE JPEG ---
    qualities = [100, 90, 70, 50, 30, 10]
    q_accs = []
    val_samples = list(zip(val_paths, val_labels))
    
    for q in tqdm(qualities, desc=f"JPEG Test [{exp_name}]"):
        qp, ql = [], []
        for path, label in val_samples:
            raw_img_val = Image.open(path).convert("RGB")
            c_img = apply_jpeg_compression(raw_img_val, q) if q < 100 else raw_img_val
            with torch.no_grad():
                pred = torch.argmax(model(transform(c_img).unsqueeze(0).to(DEVICE)), dim=1).item()
            qp.append(pred)
            ql.append(label)
        q_accs.append(accuracy_score(ql, qp) * 100)

    plt.figure(figsize=(7, 4))
    plt.plot(qualities, q_accs, marker='o', linewidth=2, color='#2b83ba')
    plt.title(f"JPEG Robustness ({exp_name})")
    plt.xlabel("JPEG Quality Factor (100 = No compress)")
    plt.ylabel("Accuracy (%)")
    plt.ylim([0, 105])
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    for idx_q, acc_val in enumerate(q_accs):
        plt.annotate(f"{acc_val:.1f}%", (qualities[idx_q], acc_val + 3), ha='center', fontsize=8)
        
    plt.savefig(os.path.join(report_dir, "jpeg_stress_test.png"), bbox_inches='tight')
    plt.close()

# --- 4. RIEPILOGO FINALE A TERMINALE ---
print("\n" + "=" * 65)
print("RIEPILOGO COMPARATIVO DEI 5 MODELLI (VALIDATION SET 20%)")
print("=" * 65)
print(f"{'Esperimento':<18} | {'Acc (%)':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<9}")
print("-" * 65)
for exp_name, (acc, prec, rec, f1) in summary_metrics.items():
    print(f"{exp_name:<18} | {acc:<9.2f} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f}")
print("=" * 65)