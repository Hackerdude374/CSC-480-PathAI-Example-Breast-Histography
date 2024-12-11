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
    train_dataset = HistopathologyDataset(data_dir=str(train_path), mode='train')
    val_dataset = HistopathologyDataset(data_dir=str(val_path), mode='val')

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    

    # Initialize datasets
    train_dataset = HistopathologyDataset(
        data_dir=str(train_path),
        mode='train'
    )
    val_dataset = HistopathologyDataset(
        data_dir=str(val_path),
        mode='val'
    )

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

    # Create checkpoint directory
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Setup training
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss"
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min"
            )
        ]
    )

    # Train
    logger.info("Starting model training...")
    trainer.fit(model, train_loader, val_loader)

    # Create models directory
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Save model with MLflow
    logger.info("Saving model...")
    mlflow.pytorch.save_model(model, str(models_dir / "latest"))
    logger.info("Training complete!")

    # Save training metadata
    metadata = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'epochs': trainer.current_epoch,
        'train_loss': trainer.callback_metrics.get('train_loss', float('nan')),
        'val_loss': trainer.callback_metrics.get('val_loss', float('nan')),
        'val_acc': trainer.callback_metrics.get('val_acc', float('nan'))
    }
    
    # Save metadata to file
    metadata_file = models_dir / "training_metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Training metadata saved to {metadata_file}")

if __name__ == "__main__":
    train()