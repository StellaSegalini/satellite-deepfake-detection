import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

checkpoint_dir = "/oblivion/users/ssegalini/tesi_satellitare/checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
unet_dir = os.path.join(checkpoint_dir, "unet")
output_dir = "/oblivion/users/ssegalini/tesi_satellitare/data/fake"
os.makedirs(output_dir, exist_ok=True)

print("Caricamento UNet custom di DiffusionSat...")
unet = UNet2DConditionModel.from_pretrained(
    unet_dir,
    torch_dtype=torch.float16,
    use_safetensors=False
)

print("Assemblaggio della pipeline completa sulla TITAN RTX...")
pipe = StableDiffusionPipeline.from_pretrained(
    checkpoint_dir,
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=False
)
pipe = pipe.to("cuda")

print("Modello caricato con successo sulla GPU!")
print("Generazione dell'immagine satellitare di test in corso...")

prompt = "high resolution satellite image of a coastal city with harbor and ships, optical remote sensing, 10m resolution"

image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

output_path = os.path.join(output_dir, "test_satellite_fake.png")
image.save(output_path)

print(f"Generazione completata! Immagine salvata in: {output_path}")