import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parents[2]))

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import mlflow.pytorch
import logging
from torch_geometric.data import Batch
from torch.utils.data import Subset
from src.data.dataset import HistopathologyDataset
from src.models.combined_model import CombinedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """Custom collate function for handling both image patches and graphs"""
    patches = torch.stack([item['patches'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    graphs = Batch.from_data_list([item['graph'] for item in batch])
    
    return {
        'patches': patches,
        'graph': graphs,
        'label': labels
    }

def train():
    # Get absolute paths
    project_root = Path(__file__).parents[2]
    train_path = r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\processed\train"
    val_path = r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\processed\val"
    print(f"Train Path: {train_path}")
    print(f"Validation Path: {val_path}")
    
# Load full datasets
    train_dataset_full = HistopathologyDataset(data_dir=str(train_path), mode='train')
    val_dataset_full = HistopathologyDataset(data_dir=str(val_path), mode='val')

    # Use a subset of the datasets
    train_dataset = Subset(train_dataset_full, range(50))  # First 50 samples
    val_dataset = Subset(val_dataset_full, range(50))  # First 50 samples for validation

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # # Initialize datasets
    # train_dataset = HistopathologyDataset(
    #     data_dir=str(train_path),
    #     mode='train'
    # )
    # val_dataset = HistopathologyDataset(
    #     data_dir=str(val_path),
    #     mode='val'
    # )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    # Initialize model
    model = CombinedModel()
    logger.info("Model initialized")

    try:
        # Train
        print("Starting model training...")
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator='auto',
            devices=1,
            log_every_n_steps=10
        )
        trainer.fit(model, train_loader, val_loader)
    except KeyboardInterrupt:
        print("Training interrupted by the user.")

        # Save the trained model
        print("Saving interrupted model...")
        torch.save(
            model.state_dict(),
            r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\model.pth"
        )
        print("Interrupted model saved successfully.")

        # Save the model with MLflow
        print("Saving interrupted MLflow model...")
        mlflow.pytorch.save_model(
            model,
            r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\models"
        )
        print("Interrupted MLflow model saved successfully.")

        return  # Exit after saving on interruption

    # Save after successful training
    print("Saving MLflow model after training...")
    mlflow.pytorch.save_model(
        model,
        r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\models"
    )
    print("MLflow model saved successfully.")

    print("Saving final state dictionary...")
    torch.save(
        model.state_dict(),
        r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\model.pth"
    )
    print("State dictionary saved successfully.")

if __name__ == "__main__":
    train()