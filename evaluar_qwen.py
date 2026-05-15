import json
import os
import sys
import time

import transformers.tokenization_utils_base

# PARCHE DE COMPATIBILIDAD IAG-ARQUI
if not hasattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "vocab"):
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.vocab = property(
        lambda self: self.get_vocab()
    )

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import mlx.core as mx

def clasificar_texto(texto: str) -> str:
    """Extrae la clase predicha del texto generado basándose en palabras clave."""
    t = texto.lower()
    if "roya" in t or "rust" in t or "puccinia" in t:
        return "Rust"
    elif "mosaico" in t or "mosaic" in t or "scmv" in t:
        return "Mosaic"
    elif "red rot" in t or "pudrición roja" in t or "colletotrichum" in t:
        return "Red_Rot"
    elif "yellow" in t or "amarilla" in t or "scylv" in t:
        return "Yellow_Disease"
    elif "sano" in t or "healthy" in t or "óptimo" in t or "sin anomalías" in t:
        return "Healthy"
    else:
        return "Desconocido"

def run_benchmark(max_images=20):
    print("========================================")
    print(" AGRO-IA: BENCHMARK DE PRECISIÓN QWEN-LORA ")
    print("========================================")
    
    valid_file = "forja_lora/valid.jsonl"
    if not os.path.exists(valid_file):
        print(f"Error: No se encuentra {valid_file}")
        return

    with open(valid_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    import random
    random.seed(42)
    random.shuffle(lines) # Mezclar para tener variedad de clases
    
    # Tomar una muestra para no demorar horas
    muestras = lines[:max_images]
    
    print(f"[*] Evaluando una muestra aleatoria de {len(muestras)} imágenes de validación...")
    
    # Cargar modelo
    print("[*] Cargando Qwen2.5-VL y Adaptador LoRA en RAM...")
    model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    model, processor = load(model_path, adapter_path=".")
    config = load_config(model_path)
    
    aciertos = 0
    total = len(muestras)
    matriz_confusion = []

    prompt_texto = "Analiza esta imagen y diagnostica clínicamente el estado fitosanitario de la planta."

    start_time = time.time()

    for idx, line in enumerate(muestras):
        data = json.loads(line)
        img_path = data["images"][0]
        
        # Extraer Ground Truth (Etiqueta Real) del path de la imagen
        # Ej: .../mendeley_dataset/train/Rust/fusion_mend...
        etiqueta_real_cruda = img_path.split("/mendeley_dataset/train/")[1].split("/")[0]
        
        # Normalizar etiqueta real
        etiqueta_real = clasificar_texto(etiqueta_real_cruda)
        if etiqueta_real == "Desconocido":
            # Intentar clasificar el texto original del asistente
            respuesta_real = data["messages"][1]["content"][0]["text"]
            etiqueta_real = clasificar_texto(respuesta_real)

        # Preparar prompt
        prompt = apply_chat_template(processor, config, prompt_texto, num_images=1)
        
        print(f"\n[{idx+1}/{total}] Evaluando imagen: {etiqueta_real_cruda}")
        
        # Inferencia
        try:
            output = generate(
                model, 
                processor, 
                prompt, 
                image=[img_path], 
                max_tokens=200, 
                temperature=0.1,
                verbose=False
            )
            
            # Extraer predicción
            prediccion = clasificar_texto(output)
            
            es_correcto = (prediccion == etiqueta_real)
            if es_correcto:
                aciertos += 1
                print(f"  ✅ Correcto! (Real: {etiqueta_real} | Predicho: {prediccion})")
            else:
                print(f"  ❌ Error! (Real: {etiqueta_real} | Predicho: {prediccion})")
                print(f"  📝 Salida cruda de Qwen: '{output}'")
                
            matriz_confusion.append({"real": etiqueta_real, "predicho": prediccion})
            mx.clear_cache()
            
        except Exception as e:
            print(f"  ⚠️ Error en inferencia: {e}")
            total -= 1

    end_time = time.time()
    
    if total > 0:
        precision = (aciertos / total) * 100
        print("\n========================================")
        print(" RESULTADOS DEL BENCHMARK ")
        print("========================================")
        print(f"Total Evaluadas : {total}")
        print(f"Aciertos        : {aciertos}")
        print(f"PRECISIÓN GLOBAL: {precision:.2f}%")
        print(f"Tiempo Total    : {end_time - start_time:.1f} segundos")
        print("========================================")

if __name__ == "__main__":
    run_benchmark(max_images=5)
