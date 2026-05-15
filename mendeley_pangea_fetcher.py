import os
import urllib.request
import zipfile
import shutil
import random

# Zenodo Mirror del Mendeley Dataset (Sugarcane Disease Classification Dataset)
ZENODO_URL = "https://zenodo.org/records/14964141/files/Sugarcane%20Disease%20Classification%20Dataset.zip?download=1"
ZIP_FILE = "mendeley_academic_dataset.zip"
RAW_DIR = "mendeley_raw"
CLASSIFICATION_DIR = "mendeley_dataset"

def fetch_and_organize():
    print("[Pangea Fetcher] Inicializando descarga de repositorio académico (Zenodo/Mendeley)...")
    
    # 1. Download
    if not os.path.exists(ZIP_FILE):
        print(f"Descargando paquete de datos pesado desde {ZENODO_URL}...")
        # NOTA: Si el link caduca o está protegido, requerirá descarga manual en navegador y moverlo aquí.
        try:
            urllib.request.urlretrieve(ZENODO_URL, ZIP_FILE)
            print("Descarga completada con éxito.")
        except Exception as e:
            print(f"Error de red bajando el ZIP: {e}. Descárgalo manualmente de Zenodo (ID 14964141) y nómbralo '{ZIP_FILE}'.")
            return
    else:
        print("El archivo ZIP ya existe. Saltando descarga.")

    # 2. Extract
    print("[Pangea Fetcher] Extrayendo banco de imágenes...")
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
    
    # 3. Organizar para YOLOv8-Classification (Split 80/20)
    print("[Pangea Fetcher] Tabulando biomasa para red neuronal de clasificación...")
    os.makedirs(os.path.join(CLASSIFICATION_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(CLASSIFICATION_DIR, 'val'), exist_ok=True)

    # Buscar la carpeta base dentro del zip (suele estar anidada)
    search_path = RAW_DIR
    for root, dirs, files in os.walk(RAW_DIR):
        # Si la carpeta contiene subcarpetas de enfermedades (ej. mas de 5 subcarpetas), es nuestro root.
        if len(dirs) > 5:
            search_path = root
            break

    classes = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
    
    for cls in classes:
        cls_path = os.path.join(search_path, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        split_point = int(len(images) * 0.8) # 80% para entrenar, 20% para validar
        train_imgs = images[:split_point]
        val_imgs = images[split_point:]

        # Crear subdirectorios de clase
        train_cls_dir = os.path.join(CLASSIFICATION_DIR, 'train', cls)
        val_cls_dir = os.path.join(CLASSIFICATION_DIR, 'val', cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_cls_dir, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_cls_dir, img))

        print(f"Clase '{cls}': {len(train_imgs)} Train | {len(val_imgs)} Val")

    print(f"[Pangea Fetcher] Éxito. El ecosistema '/{CLASSIFICATION_DIR}' está perfectamente ordenado para YOLO-Cls.")

if __name__ == "__main__":
    fetch_and_organize()
