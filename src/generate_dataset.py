import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tqdm import tqdm

CHECKPOINT_DIR = "/oblivion/users/ssegalini/tesi_satellitare/checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
OUTPUT_BASE_DIR = "/oblivion/users/ssegalini/tesi_satellitare/data/fake"

# 5 macro-categorie coerenti con fMoW (3 prompt ciascuna = 15 prompt)
PROMPT_CATEGORIES = {
    "agriculture": [
        "satellite image of agricultural fields, crop patterns, circular pivot irrigation",
        "high resolution satellite view of rural farmland and small village roads",
        "optical remote sensing image of vineyard fields on rolling hills"
    ],
    "coastal": [
        "high resolution satellite image of a coastal city with harbor and ships",
        "satellite view of a sandy beach coastline with ocean waves and resort buildings",
        "optical satellite imagery of a commercial port with container ships"
    ],
    "infrastructure": [
        "high resolution satellite image of an international airport with runways and airplanes",
        "satellite view of a major highway junction with overpasses and traffic",
        "optical satellite view of a solar farm with solar panel arrays"
    ],
    "nature": [
        "satellite view of a dense forest with a winding river",
        "high resolution satellite image of a mountain range with snow peaks and valleys",
        "optical remote sensing view of a desert landscape with sand dunes"
    ],
    "urban": [
        "high resolution satellite image of a dense residential area with roads and buildings",
        "optical satellite view of an industrial park with warehouses and parking lots",
        "satellite view of a city center with skyscrapers and avenues"
    ]
}

# 15 prompt * 40 immagini = 600 immagini fake totali (120 per macro-categoria)
IMAGES_PER_PROMPT = 40

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    print("Caricamento UNet custom di DiffusionSat...")
    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(CHECKPOINT_DIR, "unet"),
        torch_dtype=torch.float16,
        use_safetensors=False
    )
    
    print(f"Caricamento pipeline completa da: {CHECKPOINT_DIR}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        CHECKPOINT_DIR,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=False,
        local_files_only=True
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    
    print("Inizio generazione batch del dataset Fake (600 campioni)...")
    
    img_counter = 0
    total_expected = sum(len(prompts) * IMAGES_PER_PROMPT for prompts in PROMPT_CATEGORIES.values())
    pbar = tqdm(total=total_expected, desc="Generazione Fake")
    
    for category, prompts in PROMPT_CATEGORIES.items():
        cat_dir = os.path.join(OUTPUT_BASE_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for p_idx, prompt in enumerate(prompts):
            for i in range(IMAGES_PER_PROMPT):    
                generator = torch.Generator("cuda").manual_seed(42 + img_counter)
                
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
                
                file_path = os.path.join(cat_dir, f"{category}_{p_idx}_{i+1:04d}.png")
                image.save(file_path)
                
                img_counter += 1
                pbar.update(1)
                
    pbar.close()
    print(f"\nCompletata generazione di {img_counter} immagini fake in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()