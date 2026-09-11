import sys
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

DEFAULT_MODEL_PATH = "checkpoints/mixed_50_50/resnet50_best.pth"
OUTPUT_DIR = "reports/predictions"

def predict(image_path, model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Errore: Checkpoint non trovato in '{model_path}'.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata in '{image_path}'.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(outputs, dim=1).item()

    labels = {0: "REAL", 1: "FAKE"}
    result_text = f"{labels[pred_class]} ({probs[pred_class]*100:.1f}%)"

    print(f"Modello impiegato : {model_path}")
    print(f"File analizzato   : {image_path}")
    print(f"Predizione        : {labels[pred_class]}")
    print(f"Confidenza        : Real: {probs[0]*100:.2f}% | Fake: {probs[1]*100:.2f}%")

    # Salvataggio visivo
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    annotated = raw_image.copy().resize((512, 512))
    draw = ImageDraw.Draw(annotated)
    
    # Riquadro in alto a sinistra con colore verde (REAL) o rosso (FAKE)
    box_color = (34, 139, 34) if pred_class == 0 else (220, 20, 60)
    draw.rectangle([10, 10, 220, 45], fill=box_color)
    draw.text((20, 18), result_text, fill="white")

    filename = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
    annotated.save(save_path)
    print(f"Immagine salvata  : {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 src/predict_single_image.py <percorso_immagine> [opzionale: <percorso_checkpoint>]")
        sys.exit(1)
    
    ckpt = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL_PATH
    predict(sys.argv[1], ckpt)