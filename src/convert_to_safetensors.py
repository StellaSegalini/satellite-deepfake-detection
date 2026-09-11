import os
import torch
from safetensors.torch import save_file

checkpoint_dir = "/oblivion/users/ssegalini/tesi_satellitare/checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"

print(" Conversione dei file .bin in .safetensors in corso...")

for root, dirs, files in os.walk(checkpoint_dir):
    for file in files:
        if file.endswith(".bin") and not file.startswith("optimizer"):
            bin_path = os.path.join(root, file)
            safetensors_path = os.path.splitext(bin_path)[0] + ".safetensors"
            
            print(f" Converto: {file} -> {os.path.basename(safetensors_path)}")
            
            # Carichiamo i pesi torch con weights_only=False in sicurezza
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
            
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            # Salviamo in safetensors
            save_file(state_dict, safetensors_path)
            print(f" Convertito con successo!")

print("\n Conversione completata!")