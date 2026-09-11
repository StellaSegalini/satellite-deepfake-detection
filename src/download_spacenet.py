import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

DEST_DIR = "/oblivion/users/ssegalini/tesi_satellitare/data/spacenet_real"
TARGET_COUNT = 600
BUCKET_NAME = "spacenet-dataset"

# Prefissi da cui attingere (Vegas e Parigi come riserva)
PREFIXES = [
    "spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/",
    "spacenet/SN2_buildings/train/AOI_3_Paris/PS-RGB/"
]

os.makedirs(DEST_DIR, exist_ok=True)

# Svuota o controlla file già esistenti
existing_files = [f for f in os.listdir(DEST_DIR) if f.lower().endswith(('.tif', '.png', '.jpg'))]
print(f"File già presenti in destinazione: {len(existing_files)}")

s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
paginator = s3.get_paginator("list_objects_v2")

keys_to_download = []
print(f"Scansione bucket s3://{BUCKET_NAME} per raccogliere {TARGET_COUNT} immagini...")

for prefix in PREFIXES:
    if len(keys_to_download) >= TARGET_COUNT:
        break
        
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(('.tif', '.png', '.jpg')):
                    keys_to_download.append(key)
                    if len(keys_to_download) == TARGET_COUNT:
                        break
        if len(keys_to_download) == TARGET_COUNT:
            break

print(f"Selezionate {len(keys_to_download)} immagini da scaricare in: {DEST_DIR}\n")

for key in tqdm(keys_to_download, desc="Scaricamento SpaceNet"):
    filename = os.path.basename(key)
    target_path = os.path.join(DEST_DIR, filename)
    if not os.path.exists(target_path):
        s3.download_file(BUCKET_NAME, key, target_path)

print(f"\nOperazione completata! {len(keys_to_download)} immagini salvate in: {DEST_DIR}")