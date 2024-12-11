from pathlib import Path
import shutil
import random
import os
#THIS ORGANIZES THE DATASET IN THE TRAIN FOLDER
def organize_dataset(
    source_dir = r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\raw",
    dest_dir = r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\processed",
    split_ratio={"train": 0.7, "val": 0.15, "test": 0.15}
):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    print(f"Split ratios: {split_ratio}")

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"ERROR: Source directory '{source_path}' does not exist.")
        return

    print("\nScanning for patient directories...")
    patient_dirs = list(source_path.glob("[0-9]*"))
    print(f"Found {len(patient_dirs)} patient directories:")
    for patient_dir in patient_dirs:
        print(f"  - {patient_dir}")

    if len(patient_dirs) == 0:
        print("No patient directories found. Please check your source directory structure.")
        return

    # Randomly shuffle and split patients
    print("\nShuffling and splitting patient directories...")
    random.seed(42)  # For reproducibility
    random.shuffle(patient_dirs)
    n_patients = len(patient_dirs)
    n_train = int(n_patients * split_ratio["train"])
    n_val = int(n_patients * split_ratio["val"])

    splits = {
        "train": patient_dirs[:n_train],
        "val": patient_dirs[n_train:n_train + n_val],
        "test": patient_dirs[n_train + n_val:]
    }

    for split_name, split_dirs in splits.items():
        print(f"Split '{split_name}' has {len(split_dirs)} patient directories.")

    # Copy files maintaining structure
    print("\nCopying files to destination...")
    for split_name, patient_list in splits.items():
        print(f"\nProcessing split: {split_name}")
        for patient_dir in patient_list:
            print(f"  Processing patient directory: {patient_dir}")
            for class_dir in patient_dir.glob("[0-1]"):
                if not class_dir.is_dir():
                    print(f"    Skipping non-directory: {class_dir}")
                    continue

                print(f"    Found class directory: {class_dir}")
                dest_class_dir = Path(dest_dir) / split_name / class_dir.name
                dest_class_dir.mkdir(parents=True, exist_ok=True)

                for img_path in class_dir.glob("*.png"):
                    if not img_path.is_file():
                        print(f"      Skipping non-file: {img_path}")
                        continue

                    print(f"      Copying image: {img_path}")
                    dest_path = dest_class_dir / f"{patient_dir.name}_{img_path.name}"
                    shutil.copy2(img_path, dest_path)

    print("\nDataset organization complete.")
    for split_name, patient_list in splits.items():
        print(f"\n{split_name.capitalize()} split:")
        print(f"  Number of patients: {len(patient_list)}")

if __name__ == "__main__":
    organize_dataset()
