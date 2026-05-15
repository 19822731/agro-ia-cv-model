import os
import torch
from ultralytics import YOLO

torch.autograd.set_detect_anomaly(True)

def train_model():
    print("Iniciando Módulo Prometheus: Entrenamiento Visual Detección (SMALL)")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("[Módulo Pangea] Inicializando topología CNN (YOLOv8 Small)")
    model = YOLO("yolov8s.pt")  

    print("[Prometheus] Ejecutando fine-tuning sobre dataset cañero original...")
    
    results = model.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=640,
        device=device,
        name="agro_ia_cana_small"
    )
    print("[Sistema] Pesos Small guardados en: runs/detect/agro_ia_cana_small/weights/best.pt")

if __name__ == "__main__":
    train_model()
