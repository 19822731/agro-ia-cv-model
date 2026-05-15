import os
import shutil
import random

def prepare():
    master_dir = "master_dataset"
    os.makedirs(master_dir, exist_ok=True)
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(master_dir, subset), exist_ok=True)

    # 1. Copiar dataset actual (Mendeley + Kaggle previo)
    current_dir = "mendeley_dataset"
    print(f"[*] Copiando dataset actual desde {current_dir}...")
    for subset in ["train", "val"]:
        src_subset = os.path.join(current_dir, subset)
        for category in os.listdir(src_subset):
            src_cat = os.path.join(src_subset, category)
            if os.path.isdir(src_cat):
                dst_cat = os.path.join(master_dir, subset, category)
                os.makedirs(dst_cat, exist_ok=True)
                for f in os.listdir(src_cat):
                    shutil.copy2(os.path.join(src_cat, f), os.path.join(dst_cat, f))

    # 2. Mapeo y Split del nuevo PMC Dataset
    pmc_base = "pmc_raw/Sugarcane Leaf Dataset"
    mapping = {
        "Healthy Leaves": "Healthy",
        "BrownRust": "Rust",
        "Viral Disease": "Mosaic",
        "Yellow Leaf": "Yellow_Disease",
        "Banded Chlorosis": "Banded_Chlorosis",
        "Brown Spot": "Brown_Spot",
        "Grassy shoot": "Grassy_Shoot",
        "Pokkah Boeng": "Pokkah_Boeng",
        "Sett Rot": "Sett_Rot",
        "smut": "Smut",
        "Dried Leaves": "Dried_Leaves"
    }

    print("[*] Integrando Dataset PMC (11 categorías)...")
    for pmc_cat, master_cat in mapping.items():
        src_base = os.path.join(pmc_base, pmc_cat)
        if not os.path.exists(src_base):
            continue
            
        # Buscar archivos recursivamente
        all_files = []
        for root, dirs, files in os.walk(src_base):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, f))
        
        if not all_files:
            continue
            
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        # Copy Train
        dst_train = os.path.join(master_dir, "train", master_cat)
        os.makedirs(dst_train, exist_ok=True)
        for fpath in train_files:
            fname = os.path.basename(fpath)
            shutil.copy2(fpath, os.path.join(dst_train, f"pmc_{fname}"))
            
        # Copy Val
        dst_val = os.path.join(master_dir, "val", master_cat)
        os.makedirs(dst_val, exist_ok=True)
        for fpath in val_files:
            fname = os.path.basename(fpath)
            shutil.copy2(fpath, os.path.join(dst_val, f"pmc_{fname}"))
            
    print("[*] ¡Master Dataset preparado con éxito!")
    
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

if __name__ == "__main__":
    prepare()
