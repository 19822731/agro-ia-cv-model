import os
import torch
from ultralytics import YOLO

# REGLA CORE IAG-ARQUI: Sistema Inmunológico activado
torch.autograd.set_detect_anomaly(True)

def train_model():
    print("Iniciando Módulo Prometheus: Entrenamiento Visual")
    
    # Verificación de aceleración MPS (Metal Performance Shaders para Mac)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Sistema] Dispositivo de cómputo detectado: {device.upper()}")

    # El dataset ya provee su propio data.yaml con 3 clases: ['Bacterial Blight', 'Healthy', 'Red Rot']
    print("[Módulo Pangea] Inicializando topología CNN (YOLOv10 / v8)")
    # Usamos pre-entrenado (transfer learning) para optimizar eficiencia
    try:
        model = YOLO("yolov8n.pt")  
    except Exception as e:
        print(f"Error cargando el modelo pre-entrenado: {e}")
        return

    print("[Prometheus] Ejecutando fine-tuning sobre dataset cañero original...")
    
    # Ejecutamos entrenamiento oficial con las 3 clases reales (Bacterial Blight, Healthy, Red Rot)
    try:
        results = model.train(
            data="dataset/data.yaml",
            epochs=20,  # Ya listo para entrenamiento real
            imgsz=640,
            device=device,
            name="agro_ia_cana_v2"
        )
        print("[Sistema] Pesos neuronales actualizados guardados en: runs/detect/agro_ia_cana_v2/weights/best.pt")
    except Exception as e:
        print(f"Aviso de Entrenamiento (Expected in dev without images): {e}")

if __name__ == "__main__":
    train_model()
