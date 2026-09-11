import os
import glob
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

DATA_DIR = "/oblivion/users/ssegalini/tesi_satellitare/data/spacenet_real"
TARGET_SIZE = (512, 512)

tif_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))

def stretch_8bit(band):
    p2, p98 = np.percentile(band, (2, 98))
    if p98 > p2:
        band_stretched = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255)
    else:
        band_stretched = band
    return band_stretched.astype(np.uint8)

print(f"Elaborazione con percentile stretching su {len(tif_files)} file...")

for f in tqdm(tif_files):
    png_path = os.path.splitext(f)[0] + ".png"
    try:
        with rasterio.open(f) as src:
            r = stretch_8bit(src.read(1))
            g = stretch_8bit(src.read(2))
            b = stretch_8bit(src.read(3))
            
            rgb_arr = np.dstack((r, g, b))
            
            img = Image.fromarray(rgb_arr).resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            img.save(png_path, "PNG")
            
        os.remove(f)
    except Exception as e:
        print(f"Errore su {f}: {e}")

print("Conversione con contrast stretching completata!")