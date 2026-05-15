"""
Agro-IA — Entrenamiento EfficientNet-B0 para Clasificación de Enfermedades de Caña
===================================================================================
Reemplaza al clasificador YOLO-cls con un modelo más preciso para datasets pequeños.
Usa transfer learning desde ImageNet + fine-tuning progresivo.

Clases: Healthy, Rust, Mosaic, Red_Rot, Yellow_Disease
Dataset: Mendeley (~12.000 imágenes)
Hardware: Apple M4 (MPS backend)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
import json

# REGLA CORE IAG-ARQUI: Sistema Inmunológico activado
torch.autograd.set_detect_anomaly(True)

# ============ CONFIGURACIÓN ============
DATASET_DIR = os.path.join(os.path.dirname(__file__), "mendeley_dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "efficientnet_model")
NUM_CLASSES = 5  # Healthy, Rust, Mosaic, Red_Rot, Yellow_Disease
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
FREEZE_EPOCHS = 5  # Epochs con backbone congelado antes de descongelar

# Seleccionar el mejor dispositivo disponible
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[Hardware] Apple Silicon M4 (MPS) detectado ✅")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[Hardware] NVIDIA GPU detectada")
else:
    DEVICE = torch.device("cpu")
    print("[Hardware] CPU (entrenamiento lento)")

# ============ DATA AUGMENTATION ============
# Augmentation agresiva para robustecer el modelo ante fotos de campo
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preparar_datos():
    """Carga dataset Mendeley con estructura de carpetas."""
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"No se encuentra el dataset en {train_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    # Mostrar mapeo de clases
    print(f"\n[Dataset] Clases detectadas: {train_dataset.classes}")
    print(f"[Dataset] Train: {len(train_dataset)} imágenes")
    print(f"[Dataset] Val: {len(val_dataset)} imágenes")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, train_dataset.classes

def crear_modelo(num_classes):
    """Crea EfficientNet-B0 con cabeza clasificadora personalizada."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Reemplazar la cabeza clasificadora
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    
    return model.to(DEVICE)

def entrenar(model, train_loader, val_loader, classes):
    """Entrenamiento con fine-tuning progresivo."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint_15min.pth")
    if os.path.exists(checkpoint_path):
        print(f"\n[!] Reanudando desde checkpoint temporal: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint.get("history", history)
        print(f"  -> Retomando en Epoch {start_epoch+1} (Val Acc previa: {best_val_acc:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f" FASE 1: Cabeza clasificadora ({FREEZE_EPOCHS} epochs, backbone congelado)")
    print(f"{'='*60}")
    
    # Congelar backbone
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    last_save_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Descongelar backbone después de FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS:
            print(f"\n{'='*60}")
            print(f" FASE 2: Fine-tuning completo (backbone descongelado)")
            print(f"{'='*60}")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([
                {"params": model.features.parameters(), "lr": FINE_TUNE_LR},
                {"params": model.classifier.parameters(), "lr": LEARNING_RATE / 10},
            ])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # === TRAIN ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start = time.time()
        for images, labels in train_loader:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # --- GUARDADO PERIÓDICO (15 min = 900s) ---
            if time.time() - last_save_time > 900:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "history": history
                }, checkpoint_path)
                print("\n  💾 [Auto-Save] Checkpoint temporal guardado (15 min).")
                last_save_time = time.time()
                
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        elapsed = time.time() - start
        
        # === VAL ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        phase = "🔒" if epoch < FREEZE_EPOCHS else "🔓"
        print(f"  {phase} Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}% | Val Loss: {val_loss:.4f} | "
              f"{elapsed:.1f}s")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
            }, os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  💾 Mejor modelo guardado (Val Acc: {val_acc:.1f}%)")
    
    # Guardar metadata
    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump({
            "classes": classes,
            "best_val_acc": best_val_acc,
            "epochs": NUM_EPOCHS,
            "architecture": "EfficientNet-B0",
            "history": history,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f" ✅ ENTRENAMIENTO COMPLETO")
    print(f" Mejor Val Accuracy: {best_val_acc:.1f}%")
    print(f" Modelo guardado en: {OUTPUT_DIR}/best_model.pth")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_loader, val_loader, classes = preparar_datos()
    model = crear_modelo(len(classes))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modelo] EfficientNet-B0 | Total: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")
    
    entrenar(model, train_loader, val_loader, classes)
