"""
Agro-IA — Entrenamiento Swin-V2-Small para Clasificación de Enfermedades de Caña
===================================================================================
Utiliza Swin-V2-Small (Vision Transformer) con ~50M de parámetros.
Ideal para capturar matices complejos que la versión Tiny podría omitir.

Configurado a 30 épocas con Early Stopping.
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "swin_model_small")
NUM_CLASSES = 7
BATCH_SIZE = 4  # Reducido de 8 a 4 para liberar RAM unificada
NUM_EPOCHS = 30 
LEARNING_RATE = 1e-4 
FINE_TUNE_LR = 1e-5
FREEZE_EPOCHS = 4 
EARLY_STOP_PATIENCE = 8 # Aumentada la paciencia para el fine-tuning

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[Hardware] Apple Silicon M4 (MPS) detectado ✅")
else:
    DEVICE = torch.device("cpu")
    print("[Hardware] CPU")

# ============ DATA AUGMENTATION ============
train_transforms = transforms.Compose([
    transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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
    """Crea Swin-V2-Small con cabeza clasificadora personalizada."""
    print("[Sistema] Descargando/Cargando Swin-V2-Small (Pre-entrenado)...")
    model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
    
    # Reemplazar la cabeza clasificadora
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.4), # Un poco más de dropout para un modelo más grande
        nn.Linear(num_features, 512), # Aumentado a 512 neuronas intermedias
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    
    # LÓGICA DE REANUDACIÓN (AUTO-RESUME)
    best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        print(f"[*] Detectado checkpoint previo. Reanudando desde: {best_path}")
        checkpoint = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.start_epoch = checkpoint.get("epoch", 0)
        model.best_val_acc = checkpoint.get("best_val_acc", 0.0)
    else:
        model.start_epoch = 0
        model.best_val_acc = 0.0
    
    return model.to(DEVICE)

def entrenar(model, train_loader, val_loader, classes):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = model.best_val_acc
    start_epoch = model.start_epoch
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    print(f"\n{'='*60}")
    print(f" REANUDANDO ENTRENAMIENTO EN ÉPOCA {start_epoch + 1}")
    print(f"{'='*60}")

    # Inicializar fase según la época
    if start_epoch < FREEZE_EPOCHS:
        print(f"[*] Modo: Adaptación de Cabeza (Backbone Congelado)")
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
    else:
        print(f"[*] Modo: Fine-tuning Holístico (Todo Descongelado)")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {"params": [p for n, p in model.named_parameters() if "head" not in n], "lr": FINE_TUNE_LR},
            {"params": model.head.parameters(), "lr": LEARNING_RATE / 5},
        ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Transición de Fase si estamos entrenando
        if epoch == FREEZE_EPOCHS:
            print(f"\n{'='*60}")
            print(f" TRANSICIÓN A FASE 2: Fine-tuning Holístico (Vision Transformer descongelado)")
            print(f"{'='*60}")
            for param in model.parameters():
                param.requires_grad = True
            
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
            # Liberación periódica de caché MPS
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
        
        scheduler.step(val_acc)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        phase = "🔒" if epoch < FREEZE_EPOCHS else "🔓"
        print(f"  {phase} Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}% | Val Loss: {val_loss:.4f} | "
              f"{elapsed:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
            }, os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  💾 Mejor Swin-V2-Small guardado (Val Acc: {val_acc:.1f}%)")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ Sin mejoras. Paciencia Early Stopping: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early Stopping activado en época {epoch+1}.")
                break
    
    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump({
            "classes": classes,
            "best_val_acc": best_val_acc,
            "architecture": "Swin-V2-Small",
            "history": history,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f" ✅ ENTRENAMIENTO COMPLETADO")
    print(f" Mejor Val Accuracy (Swin-V2-S): {best_val_acc:.1f}%")
    print(f" Modelo guardado en: {OUTPUT_DIR}/best_model.pth")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_loader, val_loader, classes = preparar_datos()
    model = crear_modelo(len(classes))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Modelo] Swin-V2-Small | Parámetros: {total_params/1e6:.1f}M")
    
    entrenar(model, train_loader, val_loader, classes)
