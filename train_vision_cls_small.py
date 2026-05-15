import os
import torch
from ultralytics import YOLO

torch.autograd.set_detect_anomaly(True)

def train_model():
    print("Iniciando Módulo Prometheus: Entrenamiento Visual Clasificación (SMALL)")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("[Módulo Pangea] Inicializando topología CNN (YOLOv8-CLS Small)")
    model = YOLO("yolov8s-cls.pt")  

    print("[Prometheus] Ejecutando sobre 15,000 imágenes Académicas...")
    
    results = model.train(
        data="mendeley_dataset",
        epochs=25,
        imgsz=224,
        device=device,
        name="agro_ia_cana_class_small"
    )
    print("[Sistema] Pesos Small guardados en: runs/classify/agro_ia_cana_class_small/weights/best.pt")

if __name__ == "__main__":
    train_model()
