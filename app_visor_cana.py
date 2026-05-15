import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from transformers import AutoModel
import requests
import json

# 1. CONFIGURACIÓN Y MODELOS (Carga Única)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']

print("[*] Iniciando Motores del Frontend...")

# Cargar DINOv2
backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
backbone.eval()

classifier = nn.Linear(384, len(CLASSES)).to(device)
if os.path.exists("dinov2_classifier_best.pth"):
    classifier.load_state_dict(torch.load("dinov2_classifier_best.pth", map_location=device))
classifier.eval()

def procesar_diagnostico(input_img):
    if input_img is None:
        return "Error", 0, "Por favor suba una imagen."
    
    # Pre-procesamiento
    img = Image.fromarray(input_img).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = transform(img).unsqueeze(0).to(device)
    
    # Fase 2: DINOv2
    with torch.no_grad():
        features = backbone(img_t).pooler_output
        logits = classifier(features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        
    disease = CLASSES[pred.item()]
    confidence = conf.item() * 100
    
    # Fase 3: Gemma 4 via Ollama
    prompt = f"""Eres el motor de IA de 'Visor Caña', un experto mundial en fitopatología de la caña de azúcar.
DIAGNÓSTICO DETECTADO: {disease} (Confianza: {confidence:.2f}%)
TAREA: Redacta un reporte clínico profesional en ESPAÑOL. Analiza la gravedad y da 3 recomendaciones inmediatas."""

    reporte = "El cerebro experto está descansando o no se pudo conectar con Ollama."
    try:
        response = requests.post("http://localhost:11434/api/generate", 
                                 json={"model": "gemma4:e4b", "prompt": prompt, "stream": False},
                                 timeout=300)
        if response.status_code == 200:
            reporte = response.json().get("response", "")
    except Exception as e:
        reporte = f"Error conectando con Gemma 4: {e}"

    return disease, confidence / 100, reporte

# 2. DISEÑO DE LA INTERFAZ (PREMIUM)
custom_css = """
body { background-color: #0b0f0b; color: #e0e0e0; }
.gradio-container { border: 1px solid #1e3a1e !important; border-radius: 15px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important; }
.header-text { text-align: center; color: #4caf50; font-family: 'Inter', sans-serif; }
.report-box { background: #121212 !important; border: 1px solid #2e7d32 !important; border-radius: 10px !important; padding: 15px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🔬 VISOR CAÑA: Estación de Diagnóstico Inteligente
        ### Sistema Pangea V4 - Tríada de Inteligencia Artificial (DINOv2 + Gemma 4)
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Subir Muestra de Hoja", type="numpy")
            btn_diagnosticar = gr.Button("🔍 INICIAR DIAGNÓSTICO", variant="primary")
            
        with gr.Column(scale=1):
            with gr.Group():
                output_disease = gr.Textbox(label="🚩 Enfermedad Detectada", interactive=False)
                output_conf = gr.Label(label="🎯 Confianza del Modelo")
                
    with gr.Row():
        output_report = gr.Markdown(label="📝 Reporte Clínico Gemma 4", elem_classes=["report-box"])
        
    btn_diagnosticar.click(
        fn=procesar_diagnostico,
        inputs=input_image,
        outputs=[output_disease, output_conf, output_report]
    )
    
    gr.Markdown(
        """
        ---
        *Desarrollado por IAG-ARQUI para la industria azucarera global. Hardware: Apple Silicon M4 Accelerated.*
        """
    )

if __name__ == "__main__":
    demo.get_api_info = lambda *args, **kwargs: {"named_endpoints": {}, "unnamed_endpoints": {}}
    demo.launch(server_name="0.0.0.0", share=True, show_api=False)
