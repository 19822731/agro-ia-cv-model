import io
import base64
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# --- CLASIFICADOR (DINOv2) ---
import torch
import torch.nn as tnn
from torchvision import transforms
from transformers import AutoModel

# Opcional: YOLO / DINO (Para la Fase 1 de recorte)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


app = FastAPI(title="Agro-IA SaaS API (Ollama + Visión)")

# --- MIDDLEWARE CORS (Listo para Producción/Marketplace) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== FASE DE CARGA HÍBRIDA =====
print("[Agro-IA SaaS] Inicializando Motor de Inferencia Independiente...")

# 1. Cazador (Fase 1: Recorte / Detección Panorámica)
cazador = None
if YOLO:
    try:
        cazador = YOLO("runs/detect/agro_ia_cana_v23/weights/best.pt") 
        print("[Sistema] Agente Cazador (YOLOv8) Activo. (Listo para migrar a DINO).")
    except Exception:
        print("[Sistema] Cazador no encontrado. Se analizará la imagen completa.")

# 2. Laboratorio (Fase 2: Clasificador Matemático de Precisión con DINOv2)
laboratorio_backbone = None
laboratorio_classifier = None
lab_classes = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']
lab_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
lab_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    laboratorio_backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(lab_device)
    laboratorio_backbone.eval()
    
    laboratorio_classifier = tnn.Linear(384, len(lab_classes)).to(lab_device)
    model_path = os.path.join(os.path.dirname(__file__), "dinov2_classifier_best.pth")
    if os.path.exists(model_path):
        laboratorio_classifier.load_state_dict(torch.load(model_path, map_location=lab_device))
        laboratorio_classifier.eval()
        print(f"[Sistema] Laboratorio Clasificador (DINOv2) Activo ({len(lab_classes)} clases)")
except Exception as e:
    print(f"[Aviso] Laboratorio clasificador no cargado: {e}")


# --- CONEXIÓN OLLAMA (GEMMA / LLAva) ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Usar el modelo específico de Visor Caña
MODELO_OLLAMA = os.getenv("OLLAMA_MODEL", "gemma4:e4b") 

print(f"[Sistema] Científico VLM Conectado a Ollama (Modelo: {MODELO_OLLAMA})")

def razonamiento_vlm(image_path: str, clase_sugerida: str = "Desconocido", contexto_productor: str = "No proporcionado") -> str:
    """Invoca a Ollama para el análisis multimodal."""
    
    # 1. Preparar la imagen en Base64 para Ollama
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        return f"Error leyendo imagen para VLM: {e}"

    # 2. PROMPT DE SENSIBILIDAD MÁXIMA (Anti Falsos Negativos)
    prompt = (
        "Eres el Científico Agrónomo Experto de Agro-IA. Tu prioridad absoluta es la DETECCIÓN TEMPRANA DE ENFERMEDADES. "
        "En agricultura, omitir una enfermedad es catastrófico. No seas complaciente ni asumas que la planta está sana por defecto.\n\n"
        "CONTEXTO CRÍTICO REPORTADO POR EL PRODUCTOR DESDE EL CAMPO:\n"
        f"'{contexto_productor}'\n\n"
        f"El clasificador matemático inicial sugiere posibles indicios de: '{clase_sugerida}'.\n\n"
        "MISION:\n"
        "1. Analiza la imagen con exhaustividad clínica buscando manchas, pústulas, necrosis, estrés hídrico, amarillamientos internervales o deformaciones.\n"
        "2. Si el productor reporta un síntoma en el contexto, ASUME QUE HAY UN PROBLEMA. Tu tarea es IDENTIFICAR LA CAUSA EXPLICANDO QUÉ VES en la imagen que coincida con ese síntoma.\n"
        "3. Sé extremadamente detallado en los signos clínicos encontrados.\n\n"
        "INSTRUCCIONES DE FORMATO:\n"
        "- Usa **negritas** para patologías y anomalías.\n"
        "- Usa listas de puntos si hay múltiples hallazgos.\n"
        "- Sé directo y clínico. Evita introducciones genéricas.\n"
        "Al final, escribe OBLIGATORIAMENTE un veredicto claro: [VEREDICTO: RIESGO] si hay sospecha mínima o enfermedad, o [VEREDICTO: SANO] solo si la hoja es perfecta."
    )
    
    # 3. Petición a la API local de Ollama
    try:
        payload = {
            "model": MODELO_OLLAMA,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1 # Muy analítico y determinista
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        print(f"[Error Ollama] {e}")
        return f"Error de conexión con el motor VLM (Ollama): Verifica que '{MODELO_OLLAMA}' esté corriendo. Detalle: {e}"


# --- MODELOS DE DATOS Pydantic ---
from typing import Optional

class AnalyzeRequest(BaseModel):
    imagenBase64: str
    mimeType: str
    comentario: Optional[str] = None

class ConsultRequest(BaseModel):
    pregunta: str
    reporte_anterior: str
    imagen_path: Optional[str] = None


# --- ENDPOINTS ---
@app.post("/consult")
async def consult_gemma(req: ConsultRequest):
    prompt = (
        "Eres el Científico Agrónomo Experto de Agro-IA. Un productor te hace una consulta de seguimiento sobre un diagnóstico foliar reciente.\n"
        "REPORTE CLÍNICO BASE:\n"
        f"'{req.reporte_anterior}'\n\n"
        "PREGUNTA DEL PRODUCTOR:\n"
        f"'{req.pregunta}'\n\n"
        "Responde de forma empática, profesional y muy práctica. Dale los pasos exactos a seguir agronómicamente."
    )
    payload = {
        "model": MODELO_OLLAMA,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4 # Un poco más creativo para conversar
        }
    }
    
    # Si en el futuro integramos que reenvíen la imagen
    if req.imagen_path and os.path.exists(req.imagen_path):
        try:
            with open(req.imagen_path, "rb") as f:
                payload["images"] = [base64.b64encode(f.read()).decode('utf-8')]
        except Exception:
            pass

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        respuesta = response.json().get('response', '').strip()
        return {"respuesta": respuesta}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error consultando al VLM: {e}")


@app.post("/analyze")
async def analyze_image(req: AnalyzeRequest):
    try:
        image_bytes = base64.b64decode(req.imagenBase64)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == "RGBA":
            img = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 string")
    
    temp_dir = "temp_diagnostics"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "current_inference.jpg")
    img.save(temp_path)

    clase_ganadora = "Desconocido"
    certeza = 0.0
    
    # FASE 1/2: Clasificación Rápida con DINOv2
    if laboratorio_backbone and laboratorio_classifier and lab_classes:
        with torch.no_grad():
            img_tensor = lab_transform(img).unsqueeze(0).to(lab_device)
            features = laboratorio_backbone(img_tensor).pooler_output
            outputs = laboratorio_classifier(features)
            probs = torch.softmax(outputs, dim=1)
            certeza = float(probs.max())
            top1_index = int(probs.argmax())
            clase_ganadora = lab_classes[top1_index]

    # FASE 3: EL CIENTÍFICO (OLLAMA VLM)
    contexto = req.comentario or "No proporcionado"
    reporte_vlm = razonamiento_vlm(temp_path, clase_ganadora, contexto)

    # PROTOCOLO PRECAUTORIO
    repor_vlm_upper = reporte_vlm.upper()
    vlm_dice_riesgo = "VEREDICTO: RIESGO" in repor_vlm_upper or "DIAGNÓSTICO: RIESGO" in repor_vlm_upper
    vlm_dice_sano = "VEREDICTO: SANO" in repor_vlm_upper or "DIAGNÓSTICO: SANO" in repor_vlm_upper
    
    if not vlm_dice_riesgo and not vlm_dice_sano:
        frases_sanas = ["no se observan signos", "no presenta signos", "aspecto saludable", "buen estado"]
        texto_lower = reporte_vlm.lower()
        if sum(1 for f in frases_sanas if f in texto_lower) >= 2:
            vlm_dice_sano = True

    # Limpiar tags
    for tag in ["[VEREDICTO: RIESGO]", "[VEREDICTO: SANO]", "VEREDICTO: RIESGO", "VEREDICTO: SANO"]:
        reporte_vlm = reporte_vlm.replace(tag, "")
    reporte_vlm = reporte_vlm.strip()

    # Detección de Conflicto Clásico vs VLM
    math_dice_enfermo = clase_ganadora.lower() not in ["healthy", "desconocido"]
    es_enfermo = False
    
    if math_dice_enfermo and vlm_dice_sano:
        es_enfermo = True
        reporte_vlm += f"\n\n⚠️ **Nota:** El sistema detectó riesgo de '{clase_ganadora}'. Se activa prevención."
    elif math_dice_enfermo and vlm_dice_riesgo:
        es_enfermo = True
    elif not math_dice_enfermo and vlm_dice_riesgo:
        es_enfermo = True
        clase_ganadora = "Anomalía Foliar (VLM)"
    
    if clase_ganadora.lower() == "desconocido":
        clase_ganadora = "Evaluación VLM"
        
    score_salud = 95
    if es_enfermo:
        score_salud = max(20, 100 - int(certeza * 100)) if certeza > 0 else 40
    else:
        score_salud = max(80, int(certeza * 100)) if certeza > 0 else 90

    lista_issues = []
    if es_enfermo:
        lista_issues.append({
               "tipo": "enfermedad",
               "nombre": clase_ganadora,
               "severidad": "alta",
               "confianza": int(certeza * 100) if certeza > 0 else 85,
               "descripcion": "Identificado en análisis de choque VLM."
        })

    diagnostico = {
        "salud": "en_riesgo" if es_enfermo else "sano",
        "score": score_salud,
        "resumen": f"Agro-IA SaaS - Motor Ollama. Detección: {clase_ganadora}.",
        "reporte_vlm": reporte_vlm,
        "cultivo_detectado": "Caña de azúcar",
        "confianza_analisis": int(certeza * 100) if certeza > 0 else 85,
        "issues": lista_issues,
        "recomendaciones": [
            "Aplicar protocolo de contención sugerido por el VLM.",
            "Contactar con ingeniero agrónomo de zona."
        ] if es_enfermo else [
            "Lote en condiciones óptimas según VLM.",
            "Mantener cronograma estándar."
        ]
    }
    
    return {"diagnostico": diagnostico}


if __name__ == "__main__":
    import uvicorn
    # Servidor listo para desarrollo local y producción
    uvicorn.run("pangea_inference_bridge:app", host="0.0.0.0", port=8001, reload=False)
