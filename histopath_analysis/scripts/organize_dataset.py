# scripts/organize_dataset.py
from pathlib import Path
import shutil
import random

def organize_dataset(
    source_dir="data/raw/IDC_regular_ps50_idx5",
    dest_dir="data/processed",
    split_ratio={"train": 0.7, "val": 0.15, "test": 0.15}
):
    # Get all patient IDs
    patient_dirs = list(Path(source_dir).glob("[0-9]*"))
    random.shuffle(patient_dirs)
    
    # Split patients
    n_patients = len(patient_dirs)
    n_train = int(n_patients * split_ratio["train"])
    n_val = int(n_patients * split_ratio["val"])
    
    splits = {
        "train": patient_dirs[:n_train],
        "val": patient_dirs[n_train:n_train + n_val],
        "test": patient_dirs[n_train + n_val:]
    }
    
    # Copy files
    for split, patients in splits.items():
        for patient_dir in patients:
            for class_dir in patient_dir.glob("[0-1]"):
                dest_class_dir = Path(dest_dir) / split / class_dir.name
                dest_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img in class_dir.glob("*.png"):
                    shutil.copy2(img, dest_class_dir / f"{patient_dir.name}_{img.name}")

if __name__ == "__main__":
    organize_dataset()