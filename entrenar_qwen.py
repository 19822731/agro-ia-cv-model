import os
import sys
import subprocess

def entrenar_lora(dataset_path: str, salidas_dir: str = "adapters"):
    print("========================================")
    print(" FORJA MULTIMODAL: FINE-TUNING QWEN2.5-VL")
    print(" Apple Silicon (MLX) - LoRA Int4         ")
    print("========================================")
    
    if not os.path.exists(dataset_path):
        print(f"[Error致命] No se encuentra el dataset en: {dataset_path}")
        print("Asegúrate de llenar el archivo con fotos reales antes de entrenar.")
        sys.exit(1)
        
    # Crear directorio de salida si no existe para evitar errores de guardado
    os.makedirs(salidas_dir, exist_ok=True)
        
    print(f"[*] Base de Datos detectada: {dataset_path}")
    print("[*] Iniciando inyección de pesos (LoRA Rank=16) en memoria unificada...")
    
    # Comando optimizado para la versión actual de mlx-vlm (USA MODELO LOCAL PARCHEADO)
    comando = [
        "python", "-m", "mlx_vlm.lora",
        "--model-path", "model_local",
        "--dataset", dataset_path,
        "--batch-size", "1",
        "--steps", "2000",
        "--learning-rate", "1e-5",
        "--output-path", f"{salidas_dir}/adapters.safetensors",
        "--apply-chat-template",
        "--image-resize-shape", "1024", "1024"
    ]
    
    print("\n[Comando Motor MLX]:", " ".join(comando))
    print("\n[!] Precaución: Los ventiladores de la turbina M4 podrían activarse al máximo.\n")
    
    # Ejecutamos el motor de entrenamiento
    proceso = subprocess.run(comando)
    
    if proceso.returncode == 0:
        print("\n========================================")
        print(" ENTRENAMIENTO COMPLETADO EXISTOSAMENTE ")
        print(f"[*] El parche inteligente se ha guardado en: {salidas_dir}/adapters.safetensors")
        print("========================================")
    else:
        print("\n[Error] El entrenamiento fue abortado o hubo una falla de Memoria RAM.")

if __name__ == "__main__":
    # La Forja de élite utiliza el dataset curado en forja_lora/
    # Usamos ruta absoluta para evitar fallos de la librería 'datasets'
    dataset_abs = os.path.abspath("forja_lora")
    entrenar_lora(dataset_abs)
