import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import mlflow.pytorch
from pathlib import Path
import logging

import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parents[2]))

# Now imports will work
from src.data.dataset import HistopathologyDataset
from src.models.combined_model import CombinedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    logger.info("Starting training...")

    # Initialize datasets
    train_dataset = HistopathologyDataset(
        data_dir=r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\processed\train",
        mode='train'
    )
    val_dataset = HistopathologyDataset(
        data_dir=r"C:\GITHUB PROJECTS DO HERE C\CSC 480 AI PAthAI CODE PROJECT\CSC-480-PathAI-Example-Breast-Histography\histopath_analysis\src\data\processed\val",
        mode='val'
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = CombinedModel()
    logger.info("Model initialized")

    # Setup training
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',  # Uses GPU if available
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss"
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3
            )
        ]
    )

    # Train
    logger.info("Starting model training...")
    trainer.fit(model, train_loader, val_loader)

    # Save model with MLflow
    logger.info("Saving model...")
    mlflow.pytorch.save_model(model, "models/latest")
    logger.info("Training complete!")

if __name__ == "__main__":
    train()