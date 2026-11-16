"""
Test Automatizado de Auditoría para Visor Caña Pro (DINOv2 ViT-Small)
AGRO-IA SAS — Validación de Inferencia y Benchmarks Técnicos CFI
"""

import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

CLASSES = [
    'Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 
    'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 
    'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 
    'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 
    'Sett_Rot', 'Smut', 'Yellow_Disease'
]

def test_visor_cana_initialization():
    """Verifica que el modelo DINOv2 y el cabezal lineal de 21 clases se inicialicen."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    backbone.eval()
    
    classifier = nn.Linear(384, len(CLASSES)).to(device)
    weights_path = os.path.join(os.path.dirname(__file__), "dinov2_classifier_best.pth")
    
    assert os.path.exists(weights_path), f"Pesos no encontrados en {weights_path}"
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    classifier.eval()
    
    assert classifier.out_features == 21, "El cabezal debe tener exactamente 21 clases diagnósticas"
    print("✓ Inicialización de modelo y carga de pesos superada.")

def test_visor_cana_inference_latency():
    """Verifica que la inferencia se ejecute en menos de 3.500 ms (criterio CFI)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    classifier = nn.Linear(384, len(CLASSES)).to(device)
    weights_path = os.path.join(os.path.dirname(__file__), "dinov2_classifier_best.pth")
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    backbone.eval()
    classifier.eval()

    # Tensor sintético de prueba
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    start_time = time.time()
    with torch.no_grad():
        features = backbone(dummy_input).pooler_output
        logits = classifier(features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
    
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"✓ Inferencia ejecutada en {elapsed_ms:.1f} ms (Límite CFI: < 3500 ms)")
    assert elapsed_ms < 3500, f"Latencia excedida: {elapsed_ms} ms"
    assert probs.sum().item() > 0.99, "La distribución de probabilidad debe sumar 1.0"

def test_visor_cana_real_image():
    """Verifica el diagnóstico sobre la imagen de muestra temp_crop.jpg."""
    image_path = os.path.join(os.path.dirname(__file__), "temp_crop.jpg")
    if not os.path.exists(image_path):
        print("⚠️ Imagen temp_crop.jpg no encontrada, saltando test de imagen real.")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    classifier = nn.Linear(384, len(CLASSES)).to(device)
    weights_path = os.path.join(os.path.dirname(__file__), "dinov2_classifier_best.pth")
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    backbone.eval()
    classifier.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone(img_t).pooler_output
        logits = classifier(features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = CLASSES[pred.item()]
    confidence_pct = conf.item() * 100

    print(f"✓ Diagnóstico sobre imagen real: {predicted_class} con {confidence_pct:.2f}% de confianza")
    assert confidence_pct > 50.0, "La confianza diagnóstica en imagen clara debe superar el 50%"

if __name__ == "__main__":
    print("==========================================================")
    print(" AGRO-IA — AUDITORÍA AUTOMÁTICA VISOR CAÑA PRO (TRL 6)    ")
    print("==========================================================")
    test_visor_cana_initialization()
    test_visor_cana_inference_latency()
    test_visor_cana_real_image()
    print("==========================================================")
    print(" ✓ TODOS LOS TESTS DE VISOR CAÑA PRO APROBADOS CON ÉXITO ")
    print("==========================================================")
