import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import time

# Opciones de Entrenamiento:
# 1. Linear Probing (Recomendado): Congelamos DINOv2 y solo entrenamos la última capa lineal.
# 2. Fine-Tuning Completo: Entrenamos todo el modelo (más lento, requiere mucha VRAM).

def train_dinov2_linear_probing(data_dir, batch_size=32, epochs=10):
    print("========================================")
    print(" AGRO-IA: ENTRENAMIENTO DINOv2 (Linear Probing) ")
    print("========================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Dispositivo de entrenamiento: {device}")

    # 1. Preparar los datos (Mendeley Dataset)
    # DINOv2 espera imágenes normalizadas con los valores de ImageNet
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
    
    # --- BALANCEO DE CLASES (Saneamiento de Ruido) ---
    print("[*] Calculando pesos para balanceo de clases...")
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Re-definir train_loader con el sampler (eliminamos shuffle=True porque es incompatible con sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # --- FIN BALANCEO ---

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"[*] Clases detectadas: {class_names} ({num_classes} clases)")

    # 2. Cargar DINOv2 Backbone (Congelado)
    print("[*] Descargando/Cargando DINOv2 (ViT-Small) desde HuggingFace...")
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained('facebook/dinov2-small')
    dinov2 = dinov2.to(device)
    
    # Congelar los pesos del backbone
    for param in dinov2.parameters():
        param.requires_grad = False

    # 3. Definir el clasificador lineal
    embed_dim = 384 # Dimensión de salida de dinov2-small
    classifier = nn.Linear(embed_dim, num_classes).to(device)

    # 4. Optimizador y Función de Pérdida
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] Iniciando entrenamiento por {epochs} épocas...")
    start_time = time.time()

    best_acc = 0.0

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                outputs_dino = dinov2(inputs)
                features = outputs_dino.pooler_output
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # REPORTE DE PASO (Cada 10 batches + el primero)
            batch_idx = total // batch_size
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(f"  > Batch {batch_idx}: Loss {loss.item():.4f} | Acc: {100 * correct / total:.2f}%")
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        # Validación
        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs_dino = dinov2(inputs)
                features = outputs_dino.pooler_output
                outputs = classifier(features)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Época [{epoch+1}/{epochs}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), "dinov2_classifier_best.pth")

    end_time = time.time()
    print("========================================")
    print(f"[*] Entrenamiento completado en {end_time - start_time:.2f} segundos.")
    print(f"[*] Mejor Precisión en Validación: {best_acc:.2f}%")
    print("========================================")

if __name__ == "__main__":
    data_directory = "ultimate_dataset"
    if not os.path.exists(data_directory):
        print(f"Error: Dataset {data_directory} no encontrado.")
    else:
        train_dinov2_linear_probing(data_dir=data_directory, epochs=10)
