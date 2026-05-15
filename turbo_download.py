import os
from huggingface_hub import snapshot_download

# Forzar hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

model_id = "mlx-community/gemma-4-26b-a4b-it-4bit"
local_dir = "gemma4_final"

print(f"[*] Iniciando descarga TURBO de {model_id}...")
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        max_workers=16  # Paralelismo agresivo
    )
    print("[+] Descarga completada con éxito.")
except Exception as e:
    print(f"[!] Error en la descarga: {e}")
