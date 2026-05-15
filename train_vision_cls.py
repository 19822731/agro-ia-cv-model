import os
import torch
from ultralytics import YOLO

# REGLA CORE IAG-ARQUI: Sistema Inmunológico activado
torch.autograd.set_detect_anomaly(True)

def train_classifier():
    print("Iniciando Módulo Prometheus: Entrenamiento Visual CLASIFICADOR (Cascada Fase 2)")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Sistema] Dispositivo de cómputo detectado: {device.upper()}")

    print("[Módulo Pangea] Inicializando topología CNN Clasificadora (YOLOv8-cls)")
    
    # Cargamos el modelo YOLOv8 experto en clasificación en lugar de detección
    try:
        model = YOLO("yolov8n-cls.pt")  
    except Exception as e:
        print(f"Error cargando el modelo pre-entrenado clasificador: {e}")
        return

    print("[Prometheus] Ejecutando fine-tuning sobre dataset académico Mendeley (6,700 fotos)...")
    
    # YOLO Clasificador absorbe directamente la carpeta raíz que contiene 'train/' y 'val/'
    try:
        results = model.train(
            data="mendeley_dataset",
            epochs=20, 
            imgsz=224, # Resolución clásica ideal para clasificadores
            device=device,
            name="agro_ia_cana_class_v1"
        )
        print("[Sistema] Pesos neuronales de clasificación guardados en: runs/classify/agro_ia_cana_class_v1/weights/best.pt")
    except Exception as e:
        print(f"Aviso de Entrenamiento: {e}")

if __name__ == "__main__":
    train_classifier()
