"""
Agro-IA — Gran Torneo de Clasificadores (Benchmarking)
===================================================================================
Enfrenta cara a cara a:
1. EfficientNet-B0 (Base)
2. Swin-V2-Tiny (Actual)
3. Swin-V2-Small (Nuevo)

Compara Precisión (Accuracy) y tiempo de respuesta en Mac M4.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
import gc

DATASET_DIR = os.path.join(os.path.dirname(__file__), "mendeley_dataset")
BATCH_SIZE = 16

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    import mlx.core as mx # Para limpieza de caché unificada si es necesario
else:
    DEVICE = torch.device("cpu")

print(f"[Torneo] Hardware: {DEVICE}")

def cargar_datos(size=224):
    val_transforms = transforms.Compose([
        transforms.Resize((size + 4, size + 4), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dir = os.path.join(DATASET_DIR, "val")
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    return DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False), val_dataset.classes

def evaluar_modelo(model, val_loader, nombre_modelo):
    print(f"[*] Evaluando {nombre_modelo}...")
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    elapsed = time.time() - start_time
    acc = 100.0 * correct / total
    print(f"    -> Acc: {acc:.2f}% | Tiempo: {elapsed:.2f}s")
    return acc, elapsed

def main():
    val_dir = os.path.join(DATASET_DIR, "val")
    if not os.path.exists(val_dir):
        print("Error: Dataset de validación no encontrado.")
        return

    resultados = {}

    # --- 1. EfficientNet-B0 ---
    eff_path = os.path.join(os.path.dirname(__file__), "efficientnet_model", "best_model.pth")
    if os.path.exists(eff_path):
        loader, classes = cargar_datos(224)
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(classes)))
        checkpoint = torch.load(eff_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        resultados["EfficientNet-B0"] = evaluar_modelo(model, loader, "EfficientNet-B0")
        del model
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # --- 2. Swin-V2-Tiny ---
    swin_t_path = os.path.join(os.path.dirname(__file__), "swin_model", "best_model.pth")
    if os.path.exists(swin_t_path):
        loader, classes = cargar_datos(256)
        model = models.swin_v2_t(weights=None)
        num_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(classes)))
        checkpoint = torch.load(swin_t_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        resultados["Swin-V2-Tiny"] = evaluar_modelo(model, loader, "Swin-V2-Tiny")
        del model
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # --- 3. Swin-V2-Small ---
    swin_s_path = os.path.join(os.path.dirname(__file__), "swin_model_small", "best_model.pth")
    if os.path.exists(swin_s_path):
        loader, classes = cargar_datos(256)
        model = models.swin_v2_s(weights=None)
        num_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, len(classes)))
        checkpoint = torch.load(swin_s_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        resultados["Swin-V2-Small"] = evaluar_modelo(model, loader, "Swin-V2-Small")
        del model
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    else:
        print("\n[!] Swin-V2-Small aún no está listo. Espera a que termine el entrenamiento.")

    # --- VEREDICTO ---
    if resultados:
        print("\n" + "="*60)
        print(" 🏆 RESUMEN DEL GRAN TORNEO AGRO-IA 🏆")
        print("="*60)
        for nombre, (acc, t) in resultados.items():
            print(f" {nombre:18} | Precisión: {acc:.2f}% | Latencia: {t/3000:.4f}s/img")
        print("="*60)
        
        ganador = max(resultados, key=lambda x: resultados[x][0])
        print(f"\nEL LÍDER ACTUAL ES: {ganador} con {resultados[ganador][0]:.2f}% de precisión.")

if __name__ == "__main__":
    main()
