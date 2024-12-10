import pytest
import torch
from pathlib import Path
import numpy as np
from PIL import Image
from src.data.dataset import HistopathologyDataset, HistopathologyDataModule
from src.data.preprocessing import extract_patches, create_tissue_graph

@pytest.fixture
def sample_image():
    """Create sample image for testing"""
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

@pytest.fixture
def sample_dataset(tmp_path):
    """Create temporary dataset for testing"""
    dataset_path = tmp_path / "dataset"
    for label in [0, 1]:
        path = dataset_path / str(label)
        path.mkdir(parents=True)
        for i in range(5):
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            img_path = path / f"image_{i}.png"
            Image.fromarray(img).save(img_path)
    return dataset_path

class TestPreprocessing:
    def test_extract_patches(self, sample_image):
        patches = extract_patches(
            sample_image,
            patch_size=50,
            num_patches=10
        )
        
        assert len(patches) == 10
        assert isinstance(patches[0], Image.Image)
        assert patches[0].size == (50, 50)
        
    def test_create_tissue_graph(self):
        patches = torch.randn(10, 3, 50, 50)
        graph = create_tissue_graph(patches)
        
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'pos')
        
class TestDataset:
    def test_initialization(self, sample_dataset):
        dataset = HistopathologyDataset(str(sample_dataset))
        assert len(dataset) == 10  # 5 images per class
        
    def test_getitem(self, sample_dataset):
        dataset = HistopathologyDataset(str(sample_dataset))
        item = dataset[0]
        
        assert 'patches' in item
        assert 'graph' in item
        assert 'label' in item
        assert isinstance(item['patches'], torch.Tensor)
        
    @pytest.mark.parametrize(
        "patch_size,num_patches",
        [(32, 50), (64, 25)]
    )
    def test_different_configurations(
        self,
        sample_dataset,
        patch_size,
        num_patches
    ):
        dataset = HistopathologyDataset(
            str(sample_dataset),
            patch_size=patch_size,
            num_patches=num_patches
        )
        item = dataset[0]
        
        assert item['patches'].shape == (
            num_patches,
            3,
            patch_size,
            patch_size
        )

class TestDataModule:
    def test_initialization(self, sample_dataset):
        datamodule = HistopathologyDataModule(str(sample_dataset))
        assert isinstance(datamodule, HistopathologyDataModule)
        
    def test_setup(self, sample_dataset):
        datamodule = HistopathologyDataModule(str(sample_dataset))
        datamodule.setup()
        
        assert hasattr(datamodule, 'train_dataset')
        assert hasattr(datamodule, 'val_dataset')
        assert hasattr(datamodule, 'test_dataset')
        
    def test_dataloaders(self, sample_dataset):
        datamodule = HistopathologyDataModule(str(sample_dataset))
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)