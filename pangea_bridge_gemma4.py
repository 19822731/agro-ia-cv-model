import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from transformers import AutoModel
from mlx_lm import load, generate
from pypdf import PdfReader
from ultralytics import YOLO
import numpy as np

# 1. Configuración de Clases
CLASSES = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']

# 2. Cargar DINOv2 (Músculo Visual) - 94.69% Accuracy
class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DinoClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Linear(384, num_classes)
        
    def forward(self, x):
        outputs = self.backbone(x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

print("[*] Inicializando DINOv2 Analyzer...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_vision = DinoClassifier(len(CLASSES)).to(device)
model_vision.load_state_dict(torch.load("dinov2_classifier_best.pth", map_location=device), strict=False)
model_vision.eval()

# 2.5 Cargar YOLO (Cazador Lejano - Localizador de anomalías)
print("[*] Despertando al Cazador Elite (YOLO11)...")
try:
    cazador = YOLO("yolo11s.pt")
except:
    cazador = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Cargar Gemma 4 26B (Cerebro Clínico)
print("[*] Preparando Gemma 4 26B MoE (MLX)...")
# El modelo se cargará dinámicamente en la función de diagnóstico

# 4. Función de RAG (Conocimiento Científico Multi-Manual)
def get_manual_context(disease_name):
    context = ""
    manuals_dir = "BIBLIOTECA_TECNICA"
    if not os.path.exists(manuals_dir):
        return context
        
    for manual_file in os.listdir(manuals_dir):
        if manual_file.endswith(".pdf"):
            try:
                manual_path = os.path.join(manuals_dir, manual_file)
                reader = PdfReader(manual_path)
                # Escaneamos las primeras 100 páginas de cada manual buscando la enfermedad
                for i in range(min(len(reader.pages), 100)):
                    text = reader.pages[i].extract_text()
                    disease_keyword = disease_name.replace("_", " ").lower()
                    if disease_keyword in text.lower():
                        context += f"\n--- Fuente: {manual_file} (Pág. {i+1}) ---\n"
                        context += text[:1500] + "...\n"
                        break # Pasamos al siguiente manual para diversificar fuentes
            except Exception as e:
                print(f"[!] Error leyendo manual {manual_file}: {e}")
    return context

# 5. Inferencia Multimodal en Cascada (YOLO + Dino + Gemma 4)
def diagnosticar(image_path):
    print(f"[*] Iniciando Protocolo Pangea para: {image_path}")
    img_pil = Image.open(image_path).convert('RGB')
    
    # PASO 1: CAZADOR (Localización de la Anomalía)
    print("[*] Fase 1: Cazador buscando anomalías visuales...")
    crop_path = "temp_crop.jpg"
    final_img_for_dino = img_pil
    
    if cazador:
        results = cazador.predict(img_pil, conf=0.25, verbose=False)
        if len(results[0].boxes) > 0:
            # Recortamos la zona de mayor confianza
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            final_img_for_dino = img_pil.crop((box[0], box[1], box[2], box[3]))
            print(f"[+] Anomalía localizada. Recorte quirúrgico realizado.")
        else:
            print("[!] No se detectaron cajas claras. Usando imagen completa como fallback.")
    
    # PASO 2: DINO (Identificación del Patógeno)
    print("[*] Fase 2: Ojo Experto (DINOv2) clasificando la muestra...")
    img_t = transform(final_img_for_dino).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model_vision(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    disease = CLASSES[pred.item()]
    confidence = conf.item() * 100
    
    # Paso B: RAG (Búsqueda de literatura técnica)
    manual_info = get_manual_context(disease)
    
    # PASO 3: GEMMA 4 (Razonamiento y Reporte Médico)
    print("[*] Fase 3: Cerebro Clínico analizando evidencia con Gemma 4...")
    
    # 3.2 Cargar Gemma 4 (Modo Texto)
    model_llm, tokenizer = load("gemma4_final")
    
    # 3.3 Construir el Prompt Maestro
    prompt = f"""<|user|>
Eres el motor de IA de 'Visor Caña', un experto mundial en fitopatología de la caña de azúcar.
    
HALLAZGO VISUAL: El Cazador (YOLO11) ha localizado una zona de interés.
DIAGNÓSTICO MATEMÁTICO: DINOv2 ha clasificado la anomalía como '{disease}' con un {confidence:.2f}% de confianza.
    
INFORMACIÓN CIENTÍFICA DE APOYO (RAG):
{manual_info if manual_info else 'Usa tu vasto conocimiento experto en fitopatología.'}
    
TAREA: Redacta un reporte clínico profesional en ESPAÑOL. Analiza los datos y da recomendaciones de manejo agronómico inmediatas.<|end|>
<|assistant|>
"""
    
    reporte = generate(model_llm, tokenizer, prompt, max_tokens=1000)
    
    return disease, confidence, reporte

if __name__ == "__main__":
    # Imagen de prueba (Red Rot)
    test_img = "ultimate_dataset/val/Red_Rot/m1_fusion_mend_991_8.jpg"
    if os.path.exists(test_img):
        disease, conf, reporte = diagnosticar(test_img)
        print("\n" + "="*60)
        print("🟢 VISOR CAÑA - REPORTE DE DIAGNÓSTICO FITOPATOLÓGICO")
        print("="*60)
        print(f"PATOLOGÍA: {disease}")
        print(f"CONFIANZA: {conf:.2f}%")
        print("-" * 60)
        print(reporte)
        print("="*60)
    else:
        print(f"[!] No se encontró la imagen de prueba en {test_img}")
