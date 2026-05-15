import os
import zipfile
import shutil
import random

# Definimos clases estandarizadas que mapearán los sinónimos de los 3 datasets en un solo idioma
CLASS_MAP = {
    "red_rot": "Red_Rot", "red rot": "Red_Rot", "redrot": "Red_Rot",
    "rust": "Rust", "brown rust": "Rust", "brown_rust": "Rust",
    "mosaic": "Mosaic", "mosaic disease": "Mosaic",
    "healthy": "Healthy", "sano": "Healthy", "healthy leaves": "Healthy",
    "smut": "Smut", "carbon": "Smut",
    "yellow_leaf": "Yellow_Disease", "yellow": "Yellow_Disease"
}

def normalize_and_merge_datasets():
    print("[Módulo Pangea] Iniciando fusión e hibridación de Datasets (Mendeley + Kaggle)...")
    
    zips_to_process = [
        "mendeley_academic_dataset.zip",   # Dataset 1
        "mendeley_dataset_2.zip",          # Dataset 2
        "kaggle_dataset_3.zip"             # Dataset 3
    ]
    
    master_dir = "mendeley_dataset" # La carpeta maestra final
    os.makedirs(os.path.join(master_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(master_dir, 'val'), exist_ok=True)
    
    temp_dir = "temp_extraction"
    
    for arc in zips_to_process:
        if not os.path.exists(arc):
            print(f"[Aviso] No se encontró {arc}. Saltando este volumen de datos.")
            continue
            
        print(f"-> Extrayendo {arc}...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        with zipfile.ZipFile(arc, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Explorar y clasificar
        for root, dirs, files in os.walk(temp_dir):
            # Si hay muchos archivos JPG, asumimos que es una carpeta de clase final
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) > 0:
                d_name = os.path.basename(root).lower()
                
                # Normalizar nombre
                mapeo_oficial = "Otra_Clase"
                for key in CLASS_MAP:
                    if key in d_name:
                        mapeo_oficial = CLASS_MAP[key]
                        break
                        
                print(f"   Mapeando '{os.path.basename(root)}' -> '{mapeo_oficial}' ({len(images)} fotos)")
                
                # Crear ruta
                train_path = os.path.join(master_dir, 'train', mapeo_oficial)
                val_path = os.path.join(master_dir, 'val', mapeo_oficial)
                os.makedirs(train_path, exist_ok=True)
                os.makedirs(val_path, exist_ok=True)
                
                random.shuffle(images)
                split_point = int(len(images) * 0.8)
                
                for idx, img in enumerate(images):
                    src = os.path.join(root, img)
                    # Añadir prefijo para que no choquen nombres iguales entre datasets
                    dst_name = f"fusion_{arc[:4]}_{idx}_{img}"
                    
                    if idx < split_point:
                        shutil.copy(src, os.path.join(train_path, dst_name))
                    else:
                        shutil.copy(src, os.path.join(val_path, dst_name))

        print(f"✓ {arc} inyectado al enjambre neuronal con éxito.")
        
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    print("\n[Éxito] Fusión terminada. La carpeta 'mendeley_dataset' ahora contiene la mezcla genética de los tres repositorios mundiales lista para entrenar.")

if __name__ == "__main__":
    normalize_and_merge_datasets()
