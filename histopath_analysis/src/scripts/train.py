import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules
from data.dataset import HistopathologyDataModule
from models.combined_model import CombinedModel
from utils.metrics import MetricsTracker


def train(
    train_data_dir: str,
    val_data_dir: str,
    test_data_dir: str,
    model_checkpoint_path: str
):
    # Setup data
    data_module = HistopathologyDataModule(
        data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        test_data_dir=test_data_dir,
        batch_size=32,
        num_workers=4,
        patch_size=50,
        num_patches=100
    )

    # Initialize model
    model = CombinedModel()

    # Training
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(model_checkpoint_path).parent,
        filename=Path(model_checkpoint_path).stem
    )
    trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)

    # Evaluate model on test set
    evaluate(
        model_checkpoint_path=model_checkpoint_path,
        test_data_dir=test_data_dir,
        output_dir=Path(model_checkpoint_path).parent / "evaluation"
    )

if __name__ == "__main__":
    train(
        train_data_dir="histopath_analysis/src/data/processed/train",
        val_data_dir="histopath_analysis/src/data/processed/val",
        test_data_dir="histopath_analysis/src/data/processed/test",
        model_checkpoint_path="histopath_analysis/models/latest.ckpt"
    )