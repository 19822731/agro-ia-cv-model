"""
Agro-IA — Pipeline Automático de Mejora de Modelos
===================================================
Ejecuta secuencialmente:
1. Espera a que EfficientNet termine (si está corriendo)
2. Integra EfficientNet en el bridge de Pangea
3. Lanza entrenamiento LoRA de Qwen2.5-VL (1000 pasos)
4. Integra el nuevo LoRA
5. Reinicia Pangea
"""

import os
import sys
import json
import time
import shutil
import subprocess
import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EFFICIENTNET_MODEL = os.path.join(BASE_DIR, "efficientnet_model", "best_model.pth")
EFFICIENTNET_INFO = os.path.join(BASE_DIR, "efficientnet_model", "training_info.json")
BRIDGE_FILE = os.path.join(BASE_DIR, "pangea_inference_bridge.py")
LORA_DATASET = os.path.join(BASE_DIR, "forja_lora")
LORA_OUTPUT = os.path.join(BASE_DIR, "lora_v2")
PYTHON = os.path.join(BASE_DIR, "venv", "bin", "python3")
LOG_FILE = os.path.join(BASE_DIR, "pipeline_log.txt")

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def paso_1_esperar_efficientnet():
    """Espera a que el modelo EfficientNet termine de entrenar."""
    log("=" * 60)
    log("PASO 1: Esperando finalización de EfficientNet-B0...")
    log("=" * 60)
    
    # Verificar si ya terminó
    if os.path.exists(EFFICIENTNET_INFO):
        with open(EFFICIENTNET_INFO) as f:
            info = json.load(f)
        log(f"✅ EfficientNet ya entrenado: Val Acc = {info['best_val_acc']:.1f}%")
        return True
    
    # Esperar a que termine (chequear cada 30 seg)
    log("⏳ EfficientNet aún entrenando. Monitoreando cada 30 segundos...")
    max_wait = 7200  # 2 horas máximo
    waited = 0
    while waited < max_wait:
        time.sleep(30)
        waited += 30
        if os.path.exists(EFFICIENTNET_INFO):
            with open(EFFICIENTNET_INFO) as f:
                info = json.load(f)
            log(f"✅ EfficientNet completado tras {waited}s de espera: Val Acc = {info['best_val_acc']:.1f}%")
            return True
        if waited % 300 == 0:
            log(f"  ... aún esperando ({waited//60} min)")
    
    log("❌ Timeout esperando EfficientNet (2 horas). Abortando.")
    return False

def paso_2_integrar_efficientnet():
    """Modifica el bridge de Pangea para usar EfficientNet como clasificador."""
    log("=" * 60)
    log("PASO 2: Integrando EfficientNet-B0 en el bridge de Pangea...")
    log("=" * 60)
    
    # Leer el bridge actual
    with open(BRIDGE_FILE, "r") as f:
        code = f.read()
    
    # Backup
    backup_path = BRIDGE_FILE + ".backup_pre_efficientnet"
    if not os.path.exists(backup_path):
        shutil.copy2(BRIDGE_FILE, backup_path)
        log(f"  📋 Backup creado: {backup_path}")
    
    # Verificar que no se haya integrado ya
    if "EfficientNet" in code and "efficientnet_model" in code:
        log("  ⚠️ EfficientNet ya integrado en el bridge. Saltando.")
        return True
    
    # Agregar imports de torchvision después de los imports existentes
    import_block = """
# --- CLASIFICADOR EFFICIENTNET-B0 (Reemplaza YOLO-cls) ---
from torchvision import transforms, models
import torch.nn as tnn
"""
    
    # Agregar después de "import numpy as np"
    code = code.replace(
        "import numpy as np",
        "import numpy as np" + import_block
    )
    
    # Reemplazar la carga del laboratorio YOLO-cls con EfficientNet
    old_lab_block = '''# 2. Laboratorio (Fase 2: Analiza de QUÉ ENFERMEDAD sufre el recorte)
try:
    # Fase 2: Laboratorio (Clasificador) - Restaurado al ganador Nano (v12)
    model_clasificador_path = os.getenv("MODEL_CLASIFICADOR_PATH", "runs/classify/agro_ia_cana_class_v12/weights/best.pt")
    laboratorio = YOLO(model_clasificador_path)
    # Cuando termine usaríamos: laboratorio = YOLO("runs/classify/agro_ia_cana_class_v1/weights/best.pt")
    print("[Sistema] Agente Clasificador Académico Activo.")
except Exception as e:
    print(f"Laboratorio no encontrado: {e}")
    laboratorio = None'''
    
    new_lab_block = '''# 2. Laboratorio (Fase 2: EfficientNet-B0 — Clasificador de Alta Precisión)
laboratorio = None
lab_classes = []
lab_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
try:
    efficientnet_path = os.path.join(os.path.dirname(__file__), "efficientnet_model", "best_model.pth")
    if os.path.exists(efficientnet_path):
        checkpoint = torch.load(efficientnet_path, map_location="cpu", weights_only=False)
        lab_classes = checkpoint["classes"]
        laboratorio = models.efficientnet_b0(weights=None)
        num_features = laboratorio.classifier[1].in_features
        laboratorio.classifier = tnn.Sequential(
            tnn.Dropout(p=0.3),
            tnn.Linear(num_features, 256),
            tnn.ReLU(),
            tnn.Dropout(p=0.2),
            tnn.Linear(256, len(lab_classes)),
        )
        laboratorio.load_state_dict(checkpoint["model_state_dict"])
        laboratorio.eval()
        print(f"[Sistema] Agente Clasificador EfficientNet-B0 Activo ({len(lab_classes)} clases: {lab_classes})")
        print(f"[Sistema] Val Accuracy de entrenamiento: {checkpoint.get('best_val_acc', 'N/A')}%")
    else:
        # Fallback a YOLO si EfficientNet no está disponible
        model_clasificador_path = os.getenv("MODEL_CLASIFICADOR_PATH", "runs/classify/agro_ia_cana_class_v12/weights/best.pt")
        laboratorio = YOLO(model_clasificador_path)
        lab_classes = []
        print("[Sistema] Agente Clasificador YOLO Activo (fallback).")
except Exception as e:
    print(f"Laboratorio no encontrado: {e}")
    laboratorio = None'''
    
    code = code.replace(old_lab_block, new_lab_block)
    
    # Reemplazar las llamadas de predicción del laboratorio YOLO con EfficientNet
    old_predict = '''        # ---- FASE 2: LABORATORIO (CLASIFICADOR MACRO) ----
        if laboratorio:
            res_lab = laboratorio.predict(img_recortada, save=False)
            top1_index = res_lab[0].probs.top1
            certeza = float(res_lab[0].probs.top1conf)
            clase_ganadora = res_lab[0].names[top1_index]'''
    
    new_predict = '''        # ---- FASE 2: LABORATORIO (EfficientNet-B0 Clasificador) ----
        if laboratorio and lab_classes:
            with torch.no_grad():
                img_tensor = lab_transform(img_recortada).unsqueeze(0)
                outputs = laboratorio(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                certeza = float(probs.max())
                top1_index = int(probs.argmax())
                clase_ganadora = lab_classes[top1_index]
        elif laboratorio:
            # Fallback YOLO
            res_lab = laboratorio.predict(img_recortada, save=False)
            top1_index = res_lab[0].probs.top1
            certeza = float(res_lab[0].probs.top1conf)
            clase_ganadora = res_lab[0].names[top1_index]'''
    
    code = code.replace(old_predict, new_predict)
    
    # Reemplazar el fallback completo también
    old_fallback = '''        # Fallback de Resistencia: Le tiramos la foto COMPLETA.
        if laboratorio:
            res_lab = laboratorio.predict(img, save=False)
            top1_index = res_lab[0].probs.top1
            certeza = float(res_lab[0].probs.top1conf)
            clase_ganadora = res_lab[0].names[top1_index]'''
    
    new_fallback = '''        # Fallback de Resistencia: Le tiramos la foto COMPLETA.
        if laboratorio and lab_classes:
            with torch.no_grad():
                img_tensor = lab_transform(img).unsqueeze(0)
                outputs = laboratorio(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                certeza = float(probs.max())
                top1_index = int(probs.argmax())
                clase_ganadora = lab_classes[top1_index]
        elif laboratorio:
            res_lab = laboratorio.predict(img, save=False)
            top1_index = res_lab[0].probs.top1
            certeza = float(res_lab[0].probs.top1conf)
            clase_ganadora = res_lab[0].names[top1_index]'''
    
    code = code.replace(old_fallback, new_fallback)
    
    # Guardar
    with open(BRIDGE_FILE, "w") as f:
        f.write(code)
    
    log("  ✅ Bridge actualizado con EfficientNet-B0")
    return True

def paso_3_entrenar_lora():
    """Lanza el entrenamiento LoRA de Qwen2.5-VL con el dataset mejorado."""
    log("=" * 60)
    log("PASO 3: Entrenando LoRA de Qwen2.5-VL (1000 pasos)...")
    log("=" * 60)
    
    log("  🧹 Liberando memoria: cerrando Pangea en segundo plano...")
    subprocess.run("pkill -f 'pangea_inference_bridge.py'", shell=True, capture_output=True)
    time.sleep(2)
    log("  ✅ Memoria liberada. Entorno limpio para entrenamiento pesado.")
    
    dataset_abs = os.path.abspath(LORA_DATASET)
    
    cmd = [
        PYTHON, "-m", "mlx_vlm.lora",
        "--model-path", "model_local",
        "--dataset", dataset_abs,
        "--batch-size", "1",
        "--steps", "1000",
        "--learning-rate", "1e-5",
        "--output-path", os.path.join(LORA_OUTPUT, "adapters.safetensors"),
        "--apply-chat-template",
        "--image-resize-shape", "1024", "1024"
    ]
    
    log(f"  Comando: {' '.join(cmd)}")
    log(f"  Dataset: {dataset_abs}")
    log(f"  Output: {LORA_OUTPUT}")
    log("  ⚡ Esto puede tomar 2-3 horas en Apple M4...")
    
    os.makedirs(LORA_OUTPUT, exist_ok=True)
    
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    
    if result.returncode == 0:
        log("  ✅ Entrenamiento LoRA completado exitosamente")
        log(f"  📁 Adaptador guardado en: {LORA_OUTPUT}")
        return True
    else:
        log(f"  ❌ Error en entrenamiento LoRA: {result.stderr[-500:]}")
        return False

def paso_4_integrar_lora():
    """Copia el nuevo LoRA como adaptador activo."""
    log("=" * 60)
    log("PASO 4: Integrando nuevo LoRA v2...")
    log("=" * 60)
    
    new_adapters = os.path.join(LORA_OUTPUT, "adapters.safetensors")
    new_config = os.path.join(LORA_OUTPUT, "adapter_config.json")
    
    if not os.path.exists(new_adapters):
        log(f"  ❌ No se encontró {new_adapters}")
        return False
    
    # Backup del LoRA anterior
    old_adapters = os.path.join(BASE_DIR, "adapters.safetensors")
    old_config = os.path.join(BASE_DIR, "adapter_config.json")
    
    if os.path.exists(old_adapters):
        backup = old_adapters + ".v1_backup"
        if not os.path.exists(backup):
            shutil.copy2(old_adapters, backup)
            log(f"  📋 Backup LoRA v1: {backup}")
    
    # Copiar nuevo LoRA
    shutil.copy2(new_adapters, old_adapters)
    if os.path.exists(new_config):
        shutil.copy2(new_config, old_config)
    
    log("  ✅ LoRA v2 instalado como adaptador activo")
    return True

def paso_5_resumen():
    """Genera reporte final."""
    log("=" * 60)
    log("PIPELINE COMPLETADO — RESUMEN")
    log("=" * 60)
    
    # EfficientNet
    if os.path.exists(EFFICIENTNET_INFO):
        with open(EFFICIENTNET_INFO) as f:
            info = json.load(f)
        log(f"  🎯 EfficientNet-B0: Val Acc = {info['best_val_acc']:.1f}% ({len(info['classes'])} clases)")
    
    # LoRA
    new_adapters = os.path.join(LORA_OUTPUT, "adapters.safetensors")
    if os.path.exists(new_adapters):
        size_mb = os.path.getsize(new_adapters) / (1024*1024)
        log(f"  🧠 LoRA v2: {size_mb:.1f} MB (rank=16, alpha=32, 1000 pasos)")
    
    log("")
    log("  Para activar los cambios, reiniciá Pangea:")
    log(f"  cd {BASE_DIR} && ./venv/bin/python3 pangea_inference_bridge.py")
    log("")
    log("  ¡Listo para diagnosticar caña con precisión! 🌾")

if __name__ == "__main__":
    log("\n" + "🚀" * 20)
    log("AGRO-IA — PIPELINE AUTOMÁTICO DE MEJORA DE MODELOS")
    log("🚀" * 20 + "\n")
    
    # Paso 1: Esperar EfficientNet
    if not paso_1_esperar_efficientnet():
        sys.exit(1)
    
    # Paso 2: Integrar EfficientNet en bridge
    if not paso_2_integrar_efficientnet():
        log("⚠️ Fallo en integración EfficientNet, continuando con LoRA...")
    
    # Paso 3: Entrenar LoRA
    if paso_3_entrenar_lora():
        # Paso 4: Integrar LoRA
        paso_4_integrar_lora()
    else:
        log("⚠️ LoRA no completó, pero EfficientNet está integrado.")
    
    # Paso 5: Resumen
    paso_5_resumen()
