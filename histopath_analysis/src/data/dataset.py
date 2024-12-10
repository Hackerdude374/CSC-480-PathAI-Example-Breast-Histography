import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
from PIL import Image
import h5py
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from .preprocessing import create_tissue_graph, extract_patches

class HistopathologyDataset(Dataset):
    """Dataset class for histopathology images"""
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        patch_size: int = 50,
        num_patches: int = 100,
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load image paths and labels
        self.samples = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """Load dataset paths and labels"""
        samples = []
        for class_dir in self.data_dir.glob('**/[0-1]'):
            label = int(class_dir.name)
            for img_path in class_dir.glob('*.png'):
                samples.append({
                    'path': str(img_path),
                    'label': label,
                    'patient_id': img_path.parent.parent.name
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['path']).convert('RGB')
        
        # Extract patches
        patches = extract_patches(
            image,
            patch_size=self.patch_size,
            num_patches=self.num_patches
        )
        
        # Apply transforms to patches
        transformed_patches = torch.stack([
            self.transform(patch) for patch in patches
        ])
        
        # Create tissue graph
        graph = create_tissue_graph(transformed_patches)
        
        return {
            'patches': transformed_patches,
            'graph': graph,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id']
        }

class HistopathologyDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for histopathology dataset"""
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        patch_size: int = 50,
        num_patches: int = 100
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.num_patches = num_patches

    def setup(self, stage: Optional[str] = None):
        """Setup train, validation, and test datasets"""
        # Split data by patient ID for proper validation
        if stage == 'fit' or stage is None:
            self.train_dataset = HistopathologyDataset(
                f"{self.data_dir}/train",
                mode='train',
                patch_size=self.patch_size,
                num_patches=self.num_patches
            )
            self.val_dataset = HistopathologyDataset(
                f"{self.data_dir}/val",
                mode='val',
                patch_size=self.patch_size,
                num_patches=self.num_patches
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = HistopathologyDataset(
                f"{self.data_dir}/test",
                mode='test',
                patch_size=self.patch_size,
                num_patches=self.num_patches
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )