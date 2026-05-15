"""
Agro-IA — Generador de Dataset JSONL para Fine-Tuning LoRA de Qwen2.5-VL
=========================================================================
Genera un dataset de alta calidad a partir del Mendeley dataset (12.000 imgs).
Selecciona 40 ejemplos por clase con respuestas clínicas profesionales.
"""

import os
import json
import random

MENDELEY_DIR = os.path.join(os.path.dirname(__file__), "mendeley_dataset", "train")
OUTPUT_TRAIN = os.path.join(os.path.dirname(__file__), "forja_lora", "train.jsonl")
OUTPUT_VALID = os.path.join(os.path.dirname(__file__), "forja_lora", "valid.jsonl")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "forja_lora", "images")

EXAMPLES_PER_CLASS = 40  # 40 train + 10 val = 50 total por clase

# Respuestas clínicas profesionales por clase (variadas para evitar overfitting)
DIAGNOSTICOS = {
    "Healthy": [
        "La hoja presenta coloración verde uniforme sin lesiones visibles. El tejido mesófilo se encuentra en estado fisiológico normal, con nervaduras bien definidas y sin signos de estrés biótico o abiótico. [VEREDICTO: SANO]",
        "No se detectan anomalías en la superficie foliar. La planta muestra vigor vegetativo adecuado con turgencia celular óptima. Sin evidencia de esporas fúngicas ni daño mecánico. [VEREDICTO: SANO]",
        "Lámina foliar en excelente estado sanitario. Color verde intenso homogéneo, ausencia de manchas cloróticas o necróticas. Compatible con un lote en buenas condiciones de manejo agronómico. [VEREDICTO: SANO]",
        "Tejido vegetal sano. No se observan pústulas, estrías ni decoloraciones anómalas. La hoja mantiene integridad estructural completa. [VEREDICTO: SANO]",
        "Estado fitosanitario óptimo. La observación no revela presencia de patógenos fúngicos, bacterianos ni virales. Recomendación: mantener monitoreo preventivo estándar. [VEREDICTO: SANO]",
    ],
    "Rust": [
        "Se identifican **pústulas elongadas de color marrón-anaranjado** distribuidas a lo largo de las nervaduras foliares. Patrón altamente consistente con **Roya Marrón (Puccinia melanocephala)**. La severidad estimada es moderada a alta con cobertura >30% del área foliar visible. [VEREDICTO: RIESGO]",
        "Presencia de **uredosoros** característicos de color óxido sobre el haz de la hoja. La distribución es paralela a las nervaduras, signo patognomónico de **Puccinia melanocephala (Roya de la Caña)**. Se recomienda intervención fungicida inmediata. [VEREDICTO: RIESGO]",
        "Diagnóstico: **Roya Marrón**. Se observan formaciones pustulares elongadas de coloración ferruginosa sobre la superficie adaxial de la hoja. El grado de infección sugiere condiciones de alta humedad relativa y temperatura favorable para el patógeno. [VEREDICTO: RIESGO]",
        "La imagen revela **lesiones pustulares típicas de Roya** con halo clorótico periférico. Las esporas de color marrón-rojizo indican fase uredinial activa de **Puccinia melanocephala**. Severidad: alta. Acción urgente requerida. [VEREDICTO: RIESGO]",
        "Infección fúngica confirmada: **Roya Marrón (Puccinia melanocephala)**. Múltiples pústulas alineadas longitudinalmente. La densidad de esporas sugiere un foco activo de dispersión. Prioridad: tratamiento fungicida + aislamiento del cuadrante afectado. [VEREDICTO: RIESGO]",
    ],
    "Mosaic": [
        "Se detecta un **patrón de mosaico verde claro/verde oscuro** irregular en la lámina foliar, altamente indicativo de **Virus del Mosaico de la Caña (SCMV)**. Las áreas cloróticas alternan con tejido sano en bandas irregulares. [VEREDICTO: RIESGO]",
        "Diagnóstico: **Mosaico (SCMV)**. La hoja presenta **decoloración en patrón de mosaico** con zonas verde-amarillentas intercaladas con verde normal. Este patrón es patognomónico de infección viral. No hay tratamiento curativo; se recomienda eliminar plantas afectadas. [VEREDICTO: RIESGO]",
        "Infección viral confirmada: **Sugarcane Mosaic Virus (SCMV)**. Se observan **estrías cloróticas irregulares** que forman un patrón de mosaico característico. La transmisión ocurre por áfidos vectores. Recomendación: remoción de plantas sintomáticas y control de vectores. [VEREDICTO: RIESGO]",
        "La imagen muestra **moteado foliar** con alternancia de colores verde intenso y verde pálido/amarillento. Distribución compatible con **Mosaico de la Caña de Azúcar**. El virus causa reducción de sacarosa en tallos infectados. [VEREDICTO: RIESGO]",
        "Presencia de **síntomas de mosaico viral**: bandas cloróticas difusas sobre fondo verde. La irregularidad del patrón y la ausencia de estructuras fúngicas confirman etiología viral (**SCMV**). Severidad: moderada. [VEREDICTO: RIESGO]",
    ],
    "Red_Rot": [
        "Se observa **necrosis rojiza en la nervadura central** con avance centrífugo hacia los márgenes. Patrón consistente con **Pudrición Roja (Colletotrichum falcatum)**. La coloración rojo-vino del tejido interno es diagnóstica. [VEREDICTO: RIESGO]",
        "Diagnóstico: **Red Rot (Colletotrichum falcatum)**. Las **lesiones rojizas con margen oscuro** en la zona central de la hoja y la decoloración del tejido vascular son signos clásicos. Esta enfermedad compromete la calidad del jugo. [VEREDICTO: RIESGO]",
        "Infección fúngica por **Colletotrichum falcatum** confirmada. Se identifican **áreas necróticas de color rojo intenso** con bordes bien definidos. La Pudrición Roja afecta directamente el contenido de sacarosa y puede causar muerte de la planta. [VEREDICTO: RIESGO]",
        "La imagen revela **manchas rojizo-pardas extensas** sobre la lámina foliar con patrón de avance desde la nervadura central. Compatible con **Red Rot (Pudrición Roja)**. Se recomienda eliminación inmediata de material infectado. [VEREDICTO: RIESGO]",
        "Presencia de **lesiones necróticas de coloración rojo-granate** características de **Colletotrichum falcatum**. La extensión del daño sugiere infección avanzada. Prioridad crítica: evitar propagación a lotes adyacentes. [VEREDICTO: RIESGO]",
    ],
    "Yellow_Disease": [
        "Se detecta **amarillamiento intenso desde la punta hacia la base** de la hoja con nervadura central que mantiene tonalidad verde. Patrón altamente sugestivo de **Síndrome de la Hoja Amarilla (Yellow Leaf Syndrome - ScYLV)**. [VEREDICTO: RIESGO]",
        "Diagnóstico: **Yellow Leaf Disease (ScYLV)**. La hoja presenta **clorosis progresiva** con decoloración amarilla que avanza desde el ápice. La nervadura central permanece verde inicialmente, signo diferencial de esta enfermedad viral. [VEREDICTO: RIESGO]",
        "Infección viral: **Sugarcane Yellow Leaf Virus (ScYLV)**. Se observa **amarillamiento generalizado** con patrón de avance basipétalo. La enfermedad reduce significativamente el rendimiento en sacarosa. Transmisión por áfido Melanaphis sacchari. [VEREDICTO: RIESGO]",
        "La imagen muestra **decoloración amarilla uniforme** de la lámina foliar. El patrón de clorosis es consistente con **Yellow Leaf Disease**. A diferencia de deficiencia nutricional, el amarillamiento sigue un gradiente desde la punta. [VEREDICTO: RIESGO]",
        "Presencia de **síndrome de hoja amarilla**: clorosis severa con progresión acrópeta a basipétala. Compatible con **ScYLV (Sugarcane Yellow Leaf Virus)**. Se recomienda uso de semilla certificada libre de virus en próximas plantaciones. [VEREDICTO: RIESGO]",
    ],
}

# También duplicamos los diagnósticos con variaciones adicionales
DIAGNOSTICOS_EXTRA = {
    "RedRot": DIAGNOSTICOS["Red_Rot"],
    "Yellow": DIAGNOSTICOS["Yellow_Disease"],
}

def generar_dataset():
    """Genera dataset JSONL de calidad desde las imágenes de Mendeley."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    train_entries = []
    valid_entries = []
    
    for class_name in os.listdir(MENDELEY_DIR):
        class_path = os.path.join(MENDELEY_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Normalizar nombre de clase
        if class_name in DIAGNOSTICOS:
            diag_key = class_name
        elif class_name in DIAGNOSTICOS_EXTRA:
            diag_key = class_name
            DIAGNOSTICOS[class_name] = DIAGNOSTICOS_EXTRA[class_name]
        else:
            print(f"  ⚠️  Clase '{class_name}' sin diagnósticos definidos, saltando...")
            continue
        
        diagnosticos = DIAGNOSTICOS.get(diag_key, DIAGNOSTICOS.get(class_name, []))
        if not diagnosticos:
            continue
            
        # Obtener todas las imágenes de la clase
        all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_images)
        
        # Seleccionar subset
        num_total = min(EXAMPLES_PER_CLASS + 10, len(all_images))  # 40 train + 10 val
        selected = all_images[:num_total]
        
        train_imgs = selected[:EXAMPLES_PER_CLASS]
        valid_imgs = selected[EXAMPLES_PER_CLASS:num_total]
        
        print(f"  📁 {class_name}: {len(train_imgs)} train + {len(valid_imgs)} val")
        
        for img_list, entries_list in [(train_imgs, train_entries), (valid_imgs, valid_entries)]:
            for img_name in img_list:
                img_abs = os.path.abspath(os.path.join(class_path, img_name))
                diag = random.choice(diagnosticos)
                
                entry = {
                    "images": [img_abs],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_abs},
                                {"type": "text", "text": "Analiza esta imagen y diagnostica clínicamente el estado fitosanitario de la planta."}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": diag}
                            ]
                        }
                    ]
                }
                entries_list.append(entry)
    
    # Mezclar para evitar que el modelo aprenda por orden
    random.shuffle(train_entries)
    random.shuffle(valid_entries)
    
    # Guardar
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(OUTPUT_VALID, "w", encoding="utf-8") as f:
        for entry in valid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n  ✅ Dataset generado:")
    print(f"     Train: {len(train_entries)} ejemplos → {OUTPUT_TRAIN}")
    print(f"     Valid: {len(valid_entries)} ejemplos → {OUTPUT_VALID}")

if __name__ == "__main__":
    print("=" * 60)
    print(" GENERADOR DE DATASET EXPERTO — Agro-IA")
    print("=" * 60)
    random.seed(42)
    generar_dataset()
