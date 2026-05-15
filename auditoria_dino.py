import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoModel
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']

def evaluar_dino_detallado():
    print("========================================")
    print(" AGRO-IA: AUDITORÍA DE PRECISIÓN DINOv2 ")
    print("========================================")
    
    # 1. Preparar Datos de Validación
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder("ultimate_dataset/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. Cargar Modelos
    backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    backbone.eval()
    
    classifier = nn.Linear(384, len(CLASSES)).to(device)
    if os.path.exists("dinov2_classifier_best.pth"):
        classifier.load_state_dict(torch.load("dinov2_classifier_best.pth", map_location=device))
    classifier.eval()
    
    # 3. Evaluar por Clase
    class_correct = list(0. for i in range(len(CLASSES)))
    class_total = list(0. for i in range(len(CLASSES)))
    
    print("[*] Procesando dataset de validación...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = backbone(inputs).pooler_output
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\nREPORTES POR PATOLOGÍA:")
    print("-" * 40)
    for i in range(len(CLASSES)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f" {CLASSES[i]:<20}: {acc:>6.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    total_acc = 100 * sum(class_correct) / sum(class_total)
    print("-" * 40)
    print(f" PRECISIÓN GLOBAL: {total_acc:.2f}%")
    print("========================================")

if __name__ == "__main__":
    evaluar_dino_detallado()
