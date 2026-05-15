# 📖 Libro Maestro de Entrenamiento: Visor Caña (Agro-IA)

Este documento registra el origen, volumen y categorías de las imágenes utilizadas para entrenar el cerebro visual de Visor Caña (DINOv2 ViT-Small).

## 📊 Resumen Estadístico (God Mode)
- **Total de Imágenes:** 47,628
- **Categorías Totales:** 21
- **Arquitectura:** DINOv2 (Linear Probing)
- **Hardware:** Apple Silicon M4 (Aceleración MPS)

---

## 📂 Datasets Integrados

### 1. PMC Sugarcane Leaf Dataset (Global)
- **Origen:** PubMed Central / ResearchGate
- **Volumen:** ~6,748 imágenes
- **Categorías:**
  - Banded Chlorosis, Brown Spot, Grassy Shoot, Healthy, Mosaic, Pokkah Boeng, Red Rot, Rust, Sett Rot, Smut, Yellow Disease, Dried Leaves.

### 2. Kaggle Master Dataset (India/Global)
- **Origen:** Kaggle (Sugarcane Leaf Disease Dataset)
- **Volumen:** ~19,926 imágenes (El más grande hasta la fecha)
- **Categorías:**
  - Healthy, Mosaic, Red Rot, Rust, Yellow Disease.

### 3. SugarcaneLD-BD (Bangladesh)
- **Origen:** Mendeley Data (Bangladesh Research)
- **Volumen:** ~638 imágenes
- **Categorías:**
  - Eye Spot, Healthy, Red Leaf Spot, Red Rot, Ring Spot.

### 4. Sugarcane Disease Dataset (Brasil/Global)
- **Origen:** Mendeley Data (Mirror de Embrapa Digipathos)
- **Volumen:** ~3,358 imágenes de alta resolución (1000x1000)
- **Categorías:**
  - Rust, Mites, Red Spot, Yellow Leaf, Mawa, Healthy, Dried Leaves.

### 5. Roboflow LatAm (Latinoamérica)
- **Origen:** Roboflow Universe (Específico Sudamérica/Centroamérica)
- **Volumen:** ~9,850 imágenes
- **Categorías:**
  - Common Rust, Eyespot, Healthy, Leaf Mite, Leaf Scorch, Mosaic Virus, Nitrogen Deficit, Potassium Deficit, Red Rot.

---

## 🔬 Glosario de Diagnóstico (21 Clases Finales)

1.  **Banded_Chlorosis:** Clorosis en bandas.
2.  **Brown_Spot:** Mancha marrón.
3.  **Dried_Leaves:** Hojas secas (Senescencia).
4.  **Eye_Spot:** Mancha de ojo.
5.  **Grassy_Shoot:** Brote herbáceo (Fitoplasma).
6.  **Healthy:** Hoja sana (Referencia).
7.  **Leaf_Scorch:** Quemadura foliar (Muy importante en climas secos).
8.  **Mawa:** Ataque de áfidos específicos.
9.  **Mites:** Ácaros.
10. **Mosaic:** Mosaico viral.
11. **Nitrogen_Deficit:** Falta de Nitrógeno (Amarillamiento central).
12. **Pokkah_Boeng:** Mal del cogollo (Fusarium).
13. **Potassium_Deficit:** Falta de Potasio (Quemado de bordes).
14. **Red_Leaf_Spot:** Mancha roja de la hoja.
15. **Red_Rot:** Muermo rojo / Podredumbre roja.
16. **Red_Spot:** Mancha roja.
17. **Ring_Spot:** Mancha anular.
18. **Rust:** Roya (Café/Naranja).
19. **Sett_Rot:** Podredumbre del trozo.
20. **Smut:** Carbón de la caña.
21. **Yellow_Disease:** Amarillamiento (SCYLV).

---
*Última actualización: 13 de mayo de 2026*
