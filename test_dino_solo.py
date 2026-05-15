import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from transformers import AutoModel
import sys

# 1. CONFIGURACIÓN
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']

def inicializar_dino():
    print("========================================")
    print(" 👁️ DINOv2 SOLO: PRUEBA DE VISIÓN PURA ")
    print("========================================")
    
    backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    backbone.eval()
    
    classifier = nn.Linear(384, len(CLASSES)).to(device)
    if os.path.exists("dinov2_classifier_best.pth"):
        classifier.load_state_dict(torch.load("dinov2_classifier_best.pth", map_location=device))
        print("[+] Pesos de DINOv2 cargados.")
    classifier.eval()
    
    return backbone, classifier

def diagnosticar_solo(image_path, backbone, classifier):
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = backbone(img_t).pooler_output
        logits = classifier(features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        
    disease = CLASSES[pred.item()]
    confidence = conf.item() * 100
    
    return disease, confidence

if __name__ == "__main__":
    backbone, classifier = inicializar_dino()
    
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        test_img = "ultimate_dataset/val/Red_Rot/m1_fusion_mend_991_8.jpg"
        
    if os.path.exists(test_img):
        print(f"[*] Analizando {os.path.basename(test_img)} SIN ayuda de YOLO...")
        disease, conf = diagnosticar_solo(test_img, backbone, classifier)
        print("\n" + "="*50)
        print(f"🚩 DIAGNÓSTICO DINO: {disease}")
        print(f"🎯 CONFIANZA: {conf:.2f}%")
        print("="*50)
    else:
        print(f"Error: No se encontró {test_img}")
