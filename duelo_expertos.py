import os
import time
import torch
import gc
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from PIL import Image

def duelo_de_expertos(image_path):
    model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    adapter_v1_path = "lora_v1_200steps"
    adapter_v2_path = "." # El nuevo de 2000 pasos está en la raíz
    
    prompt = "<|user|>\n<|image_pad|>\nRealiza un diagnóstico clínico detallado de esta hoja de caña de azúcar. Indica enfermedad, síntomas visuales observados y recomendación técnica. Atiende como un experto agrónomo.<|end|>\n<|assistant|>\n"

    print("\n" + "="*60)
    print(" ⚔️  DUELO DE EXPERTOS AGRO-IA: 200 vs 2000 PASOS ⚔️")
    print("="*60)
    print(f"[*] Imagen de prueba: {image_path}\n")

    # --- PRUEBA V1 (200 Pasos) ---
    print("[1/2] Evaluando Experto V1 (200 Pasos)...")
    try:
        model_v1, processor_v1 = load(model_path, adapter_path=adapter_v1_path)
        start_time = time.time()
        output_v1 = generate(model_v1, processor_v1, prompt, image=[image_path], max_tokens=300)
        time_v1 = time.time() - start_time
        # Limpiar memoria agresivamente
        del model_v1
        del processor_v1
        gc.collect()
        mx.metal.clear_cache()
        print("[*] Memoria liberada tras V1.")
    except Exception as e:
        output_v1 = f"Error en V1: {e}"
        time_v1 = 0

    print("[2/2] Evaluando Experto V2 (2000 Pasos)...")
    try:
        model_v2, processor_v2 = load(model_path, adapter_path=adapter_v2_path)
        start_time = time.time()
        output_v2 = generate(model_v2, processor_v2, prompt, image=[image_path], max_tokens=300)
        time_v2 = time.time() - start_time
        del model_v2
        del processor_v2
        gc.collect()
        mx.metal.clear_cache()
        print("[*] Memoria liberada tras V2.")
    except Exception as e:
        output_v2 = f"Error en V2: {e}"
        time_v2 = 0

    print("\n" + "="*60)
    print(" 📜 VEREDICTO COMPARATIVO ")
    print("="*60)
    
    print(f"\n--- [EXPERTO V1 - 200 PASOS] ({time_v1:.2f}s) ---")
    print(output_v1)
    
    print(f"\n--- [EXPERTO V2 - 2.000 PASOS] ({time_v2:.2f}s) ---")
    print(output_v2)
    
    print("\n" + "="*60)
    print(" ANÁLISIS: Observa la precisión técnica y el vocabulario. ")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Buscamos una imagen real del dataset para la prueba
    test_img = "forja_lora/images/fusion_mend_1372_rust_93.jpeg" # Imagen confirmada de Roya
    if not os.path.exists(test_img):
        # Intentar buscar alguna imagen en el dataset
        import glob
        imgs = glob.glob("forja_lora/images/*.jpg")
        if imgs:
            test_img = imgs[0]
        else:
            print("No se encontraron imágenes para la prueba.")
            exit()
            
    duelo_de_expertos(test_img)
