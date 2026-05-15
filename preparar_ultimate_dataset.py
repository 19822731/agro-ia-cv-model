import os
import shutil
import random

def prepare():
    master_dir = "ultimate_dataset"
    os.makedirs(master_dir, exist_ok=True)
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(master_dir, subset), exist_ok=True)

    # Función auxiliar para copiar recursivamente con prefijo
    def add_to_master(src_root, category_mapping, prefix):
        print(f"[*] Procesando origen: {src_root} (Prefijo: {prefix})")
        for src_cat, master_cat in category_mapping.items():
            src_dir = os.path.join(src_root, src_cat)
            if not os.path.exists(src_dir):
                continue
            
            all_files = []
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        all_files.append(os.path.join(root, f))
            
            if not all_files:
                continue
                
            random.shuffle(all_files)
            split_idx = int(len(all_files) * 0.8)
            train_files = all_files[:split_idx]
            val_files = all_files[split_idx:]
            
            for fpath, subset in [(train_files, "train"), (val_files, "val")]:
                dst_dir = os.path.join(master_dir, subset, master_cat)
                os.makedirs(dst_dir, exist_ok=True)
                for path in fpath:
                    fname = os.path.basename(path)
                    dst_path = os.path.join(dst_dir, f"{prefix}_{fname}")
                    shutil.copy2(path, dst_path)

    # 1. Dataset Actual (Mendeley + PMC)
    add_to_master("master_dataset/train", {c: c for c in os.listdir("master_dataset/train")}, "m1")
    add_to_master("master_dataset/val", {c: c for c in os.listdir("master_dataset/val")}, "m1v")

    # 2. Kaggle (19,926 imágenes)
    kaggle_map = {
        "Healthy": "Healthy",
        "Mosaic": "Mosaic",
        "RedRot": "Red_Rot",
        "Rust": "Rust",
        "Yellow": "Yellow_Disease"
    }
    add_to_master("kaggle_extracted/Sugarcane Leaf Disease Dataset", kaggle_map, "kag")

    # 3. Bangladesh (638 imágenes)
    bd_map = {
        "EyeSpot": "Eye_Spot",
        "Healthy": "Healthy",
        "RedLeafSpot": "Red_Leaf_Spot",
        "RedRot": "Red_Rot",
        "RingSpot": "Ring_Spot"
    }
    add_to_master("bd_raw/SugarcaneLD-BD A Sugarcane Leaf Diseases Dataset f/SugarcaneLD-BD", bd_map, "bd")

    # 4. Mites (3,358 imágenes)
    mites_map = {
        "Diseases/BrownRust": "Rust",
        "Diseases/Mawa": "Mawa",
        "Diseases/Mites": "Mites",
        "Diseases/RedSpot": "Red_Spot",
        "Diseases/YellowLeaf": "Yellow_Disease",
        "Healthy Leaves": "Healthy",
        "Dried Leaves": "Dried_Leaves"
    }
    add_to_master("mites_raw", mites_map, "mit")

    # 5. Roboflow LatAm (9,850 imágenes)
    roboflow_map = {
        "Common Rust": "Rust",
        "Eyespot": "Eye_Spot",
        "Healthy": "Healthy",
        "Leaf Mite": "Mites",
        "Leaf Scorch": "Leaf_Scorch",
        "Mosaic Virus": "Mosaic",
        "Nitrogen Deficit": "Nitrogen_Deficit",
        "Potassium Deficit": "Potassium_Deficit",
        "Red Rot": "Red_Rot"
    }
    # Roboflow tiene carpetas train, valid, test
    for subset in ["train", "valid", "test"]:
        src_path = os.path.join("roboflow_v1", subset)
        add_to_master(src_path, roboflow_map, f"rob_{subset}")

    print("\n[*] ¡Ultimate Dataset preparado con éxito!")
    # Conteo final
    for subset in ["train", "val"]:
        total = 0
        path = os.path.join(master_dir, subset)
        for cat in os.listdir(path):
            cat_path = os.path.join(path, cat)
            if os.path.isdir(cat_path):
                count = len(os.listdir(cat_path))
                total += count
        print(f"Total en {subset}: {total}")
    
    # Listar categorías
    cats = sorted(os.listdir(os.path.join(master_dir, "train")))
    print(f"Categorías finales ({len(cats)}): {cats}")

if __name__ == "__main__":
    prepare()
