import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from transformers import AutoModel
import sys
import requests
import json

# 1. CONFIGURACIÓN
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']

def inicializar_sistema():
    print("========================================")
    print(" 🟢 VISOR CAÑA: SISTEMA PANGEA (V4) ")
    print("      (Powered by DINOv2 Only)      ")
    print("========================================")
    
    # Cargar DINOv2 (Ojo Experto)
    print("[*] Cargando Ojo Experto (DINOv2)...")
    backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    backbone.eval()
    
    classifier = nn.Linear(384, len(CLASSES)).to(device)
    if os.path.exists("dinov2_classifier_best.pth"):
        classifier.load_state_dict(torch.load("dinov2_classifier_best.pth", map_location=device))
        print("[+] Pesos de DINOv2 cargados (93.15% Acc).")
    classifier.eval()
        
    return backbone, classifier

def diagnosticar(image_path, backbone, classifier):
    img = Image.open(image_path).convert('RGB')
    
    # Clasificación Directa (DINOv2)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = transform(img).unsqueeze(0).to(device)
    
    print(f"[*] Analizando imagen: {os.path.basename(image_path)}")
    with torch.no_grad():
        features = backbone(img_t).pooler_output
        logits = classifier(features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        
    disease = CLASSES[pred.item()]
    confidence = conf.item() * 100
    
    # --- FASE 3: REPORTE CLÍNICO (Gemma 4 vía Ollama) ---
    print(f"[*] Solicitando reporte clínico a Gemma 4 (vía Ollama)...")
    prompt = f"""Eres el motor de IA de 'Visor Caña', un experto mundial en fitopatología de la caña de azúcar.
    
DIAGNÓSTICO DETECTADO: {disease} (Confianza: {confidence:.2f}%)

TAREA: Redacta un reporte clínico profesional en ESPAÑOL. Analiza la gravedad de esta enfermedad y da 3 recomendaciones de manejo agronómico inmediatas según las buenas prácticas internacionales."""

    reporte = "No se pudo generar el reporte."
    try:
        response = requests.post("http://localhost:11434/api/generate", 
                                 json={"model": "gemma4:e4b", "prompt": prompt, "stream": False},
                                 timeout=300)
        if response.status_code == 200:
            reporte = response.json().get("response", "Reporte vacío.")
    except Exception as e:
        print(f"[!] Error conectando con Ollama: {e}")

    return disease, confidence, reporte

if __name__ == "__main__":
    backbone, classifier = inicializar_sistema()
    
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        test_img = "ultimate_dataset/val/Red_Rot/m1_fusion_mend_991_8.jpg"
        
    if os.path.exists(test_img):
        disease, conf, reporte = diagnosticar(test_img, backbone, classifier)
        print("\n" + "="*50)
        print(f"🚩 DIAGNÓSTICO: {disease}")
        print(f"🎯 CONFIANZA: {conf:.2f}%")
        print("-" * 50)
        print(f"📝 REPORTE CLÍNICO GEMMA 4:\n{reporte}")
        print("="*50)
        
        if conf < 80:
            print("⚠️ AVISO: Confianza moderada. Verificar con manual técnico.")
        else:
            print("✅ Diagnóstico sólido. Aplicar protocolo de manejo.")
    else:
        print(f"Error: No se encontró {test_img}")
