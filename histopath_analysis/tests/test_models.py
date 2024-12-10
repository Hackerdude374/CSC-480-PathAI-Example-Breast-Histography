import pytest
import torch
from src.models.mil import MILModel
from src.models.gnn import GNNModel
from src.models.combined_model import CombinedModel

@pytest.fixture
def sample_batch():
    """Create sample batch for testing"""
    batch_size = 2
    num_patches = 4
    patch_size = 50
    
    return {
        'patches': torch.randn(batch_size, num_patches, 3, patch_size, patch_size),
        'graph': {
            'x': torch.randn(batch_size * 10, 512),
            'edge_index': torch.randint(0, batch_size * 10, (2, 30)),
            'batch': torch.repeat_interleave(
                torch.arange(batch_size),
                repeats=10
            )
        },
        'label': torch.randint(0, 2, (batch_size,))
    }

class TestMILModel:
    def test_initialization(self):
        model = MILModel(num_classes=2)
        assert isinstance(model, MILModel)
        
    def test_forward_pass(self, sample_batch):
        model = MILModel(num_classes=2)
        logits, attention = model(sample_batch['patches'])
        
        assert logits.shape == (2, 2)  # (batch_size, num_classes)
        assert attention.shape == (2, 4, 1)  # (batch_size, num_patches, 1)
        
    def test_training_step(self, sample_batch):
        model = MILModel(num_classes=2)
        loss = model.training_step(sample_batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

class TestGNNModel:
    def test_initialization(self):
        model = GNNModel()
        assert isinstance(model, GNNModel)
        
    def test_forward_pass(self, sample_batch):
        model = GNNModel()
        logits, features = model(sample_batch['graph'])
        
        assert logits.shape == (2, 2)  # (batch_size, num_classes)
        assert features.shape == (2, 256)  # (batch_size, hidden_dim)
        
    def test_training_step(self, sample_batch):
        model = GNNModel()
        loss = model.training_step(sample_batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

class TestCombinedModel:
    def test_initialization(self):
        model = CombinedModel()
        assert isinstance(model, CombinedModel)
        
    def test_forward_pass(self, sample_batch):
        model = CombinedModel()
        logits, outputs = model(sample_batch)
        
        assert logits.shape == (2, 2)  # (batch_size, num_classes)
        assert 'mil_attention' in outputs
        assert 'gnn_features' in outputs
        
    def test_training_step(self, sample_batch):
        model = CombinedModel()
        loss = model.training_step(sample_batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        model = CombinedModel()
        batch = {
            'patches': torch.randn(batch_size, 4, 3, 50, 50),
            'graph': {
                'x': torch.randn(batch_size * 10, 512),
                'edge_index': torch.randint(0, batch_size * 10, (2, 30)),
                'batch': torch.repeat_interleave(
                    torch.arange(batch_size),
                    repeats=10
                )
            },
            'label': torch.randint(0, 2, (batch_size,))
        }
        
        logits, outputs = model(batch)
        assert logits.shape == (batch_size, 2)