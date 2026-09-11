import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

OUTPUT_REAL_DIR = "/oblivion/users/ssegalini/tesi_satellitare/data/real"
os.makedirs(OUTPUT_REAL_DIR, exist_ok=True)

# Mappatura bilanciata delle 5 macro-categorie su classi fMoW
CATEGORY_MAP = {
    "agriculture": [
        "crop_field", "orchard", "pasture", "barn"
    ],
    "coastal": [
        "port", "marina", "coastline", "lighthouse", "shipyard", "pier"
    ],
    "infrastructure": [
        "solar_farm", "wind_turbine", "bridge", "substation", "nuclear_power_plant",
        "power_plant", "oil_or_gas_facility", "storage_tank"
    ],
    "nature": [
        "lake_or_pond", "dam", "archaeological_site", "surface_mine"
    ],
    "urban": [
        "residential_neighbourhood", "single_unit_residential", "multi_unit_residential",
        "commercial_building", "office_building", "stadium"
    ]
}

# 120 per macro-categoria -> 120 * 5 = 600 immagini totali
TARGET_PER_CATEGORY = 120
SHARPNESS_THRESHOLD = 80.0  # Soglia Laplacian per evitare crop sfocati
BUCKET_NAME = "spacenet-dataset"
SPLITS = ["Hosted-Datasets/fmow/fmow-rgb/train/", "Hosted-Datasets/fmow/fmow-rgb/val/"]

print("Connessione al bucket AWS S3 per download di 600 immagini fMoW...")
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def check_and_crop(img):
    w, h = img.size
    if w < 512 or h < 512:
        return False, None
    
    # Center crop a 512x512 senza interpolazioni o upscaling
    left = (w - 512) // 2
    top = (h - 512) // 2
    cropped = img.crop((left, top, left + 512, top + 512))
    
    # Controllo nitidezza tramite varianza laplaciana
    gray = np.array(cropped.convert("L"))
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if variance >= SHARPNESS_THRESHOLD:
        return True, cropped
    return False, None

for macro_cat, sub_classes in CATEGORY_MAP.items():
    existing_files = [f for f in os.listdir(OUTPUT_REAL_DIR) if f.startswith(f"real_{macro_cat}_")]
    saved_in_macro = len(existing_files)
    
    if saved_in_macro >= TARGET_PER_CATEGORY:
        print(f"Categoria '{macro_cat}' già completa ({saved_in_macro}/{TARGET_PER_CATEGORY}). Salto.")
        continue
        
    pbar = tqdm(total=TARGET_PER_CATEGORY, initial=saved_in_macro, desc=f"Download {macro_cat}")
    macro_done = False
    
    for sub_class in sub_classes:
        if macro_done:
            break
            
        for split in SPLITS:
            if macro_done:
                break
                
            prefix = f"{split}{sub_class}/"
            paginator = s3.get_paginator("list_objects_v2")
            
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
                if "Contents" not in page or macro_done:
                    continue
                    
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if (key.endswith('.jpg') or key.endswith('.png')) and not key.endswith('.json'):
                        try:
                            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                            with Image.open(BytesIO(response["Body"].read())) as img:
                                img = img.convert("RGB")
                                is_valid, cropped_img = check_and_crop(img)
                                
                                if is_valid:
                                    save_path = os.path.join(
                                        OUTPUT_REAL_DIR, 
                                        f"real_{macro_cat}_{saved_in_macro+1:04d}.png"
                                    )
                                    cropped_img.save(save_path, "PNG")
                                    saved_in_macro += 1
                                    pbar.update(1)
                                    
                                    if saved_in_macro >= TARGET_PER_CATEGORY:
                                        macro_done = True
                                        break
                        except Exception:
                            continue
                            
    pbar.close()

print(f"\nDownload completato! 600 file fMoW salvati in: {OUTPUT_REAL_DIR}")