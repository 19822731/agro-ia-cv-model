"""
Agro-IA — Entrenamiento Swin-V2-Tiny para Clasificación de Enfermedades de Caña
===================================================================================
Utiliza Vision Transformers con "Shifted Windows" y "Cosine Attention" (Swin-V2).
Configurado a 25 épocas con Early Stopping para maximizar precisión sin sobreajuste.

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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "swin_model")
NUM_CLASSES = 5
BATCH_SIZE = 16
NUM_EPOCHS = 25  # 25 epochs es ideal, con Early Stopping cortará si no mejora
LEARNING_RATE = 1e-4 # Learning rate menor para transformers
FINE_TUNE_LR = 1e-5
FREEZE_EPOCHS = 3  # Pocas épocas congelado, Swin adapta rápido el head
EARLY_STOP_PATIENCE = 5 # Si en 5 épocas no mejora el Val Acc, cortamos

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
# Swin-V2-Tiny espera resoluciones de 256x256 (a diferencia de los 224x224 del EfficientNet)
train_transforms = transforms.Compose([
    transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preparar_datos():
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"No se encuentra el dataset en {train_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    print(f"\n[Dataset] Clases detectadas: {train_dataset.classes}")
    print(f"[Dataset] Train: {len(train_dataset)} imágenes")
    print(f"[Dataset] Val: {len(val_dataset)} imágenes")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, train_dataset.classes

def crear_modelo(num_classes):
    """Crea Swin-V2-Tiny con cabeza clasificadora personalizada."""
    print("[Sistema] Descargando/Cargando Swin-V2-Tiny (Pre-entrenado)...")
    model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
    
    # Reemplazar la cabeza clasificadora (en Swin se llama model.head)
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    
    return model.to(DEVICE)

def entrenar(model, train_loader, val_loader, classes):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    start_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    print(f"\n{'='*60}")
    print(f" FASE 1: Adaptación de Cabeza ({FREEZE_EPOCHS} epochs, Swin congelado)")
    print(f"{'='*60}")
    
    # Congelar backbone (Swin-V2 layers)
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    
    optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            print(f"\n{'='*60}")
            print(f" FASE 2: Fine-tuning Holístico (Vision Transformer descongelado)")
            print(f"{'='*60}")
            for param in model.parameters():
                param.requires_grad = True
            
            # Tasas de aprendizaje diferenciadas (menor para las capas profundas del transformer)
            optimizer = optim.Adam([
                {"params": [p for n, p in model.named_parameters() if "head" not in n], "lr": FINE_TUNE_LR},
                {"params": model.head.parameters(), "lr": LEARNING_RATE / 5},
            ])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        
        # === TRAIN ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        
        for images, labels in train_loader:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
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
        
        scheduler.step(val_acc) # Usar val_acc para ReduceLROnPlateau (modo 'max')
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        phase = "🔒" if epoch < FREEZE_EPOCHS else "🔓"
        print(f"  {phase} Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}% | Val Loss: {val_loss:.4f} | "
              f"{elapsed:.1f}s")
        
        # MECANISMO ANTI-SOBREAJUSTE: Guardar mejor modelo o contar paciencia
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
            }, os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  💾 Mejor Swin-V2 guardado (Val Acc: {val_acc:.1f}%)")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ Sin mejoras. Paciencia Early Stopping: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 [Protección Anti-Sobreajuste] Activada. El modelo dejó de mejorar y empezó a memorizar.")
                print(f"   Cortando entrenamiento en época {epoch+1}. Nos quedamos con la mejor versión.")
                break
    
    # Guardar metadata
    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump({
            "classes": classes,
            "best_val_acc": best_val_acc,
            "epochs": epoch + 1,
            "architecture": "Swin-V2-Tiny",
            "history": history,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f" ✅ ENTRENAMIENTO DE VISION TRANSFORMER COMPLETO")
    print(f" Mejor Val Accuracy (Swin-V2-T): {best_val_acc:.1f}%")
    print(f" Modelo guardado en: {OUTPUT_DIR}/best_model.pth")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_loader, val_loader, classes = preparar_datos()
    model = crear_modelo(len(classes))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modelo] Swin-V2-Tiny | Total: {total_params/1e6:.1f}M | Trainable (Head): {trainable_params/1e6:.1f}M")
    
    entrenar(model, train_loader, val_loader, classes)
