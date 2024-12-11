import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow.pytorch
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.dataset import HistopathologyDataset
from src.models.combined_model import CombinedModel

def train():
    # Setup data
    train_dataset = HistopathologyDataset("data/raw/IDC_regular_ps50_idx5")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = CombinedModel()

    # Training
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_loader)

    # Save model with MLflow
    mlflow.pytorch.save_model(model, "models/latest")

if __name__ == "__main__":
    train()