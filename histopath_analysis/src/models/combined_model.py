import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple

from .mil import MILModel
from .gnn import GNNModel

class CombinedModel(pl.LightningModule):
    """Combined MIL-GNN model for comprehensive tissue analysis"""
    def __init__(
        self,
        num_classes: int = 2,
        mil_feature_dim: int = 512,
        gnn_hidden_dim: int = 256,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize sub-models
        self.mil_model = MILModel(num_classes=num_classes)
        self.gnn_model = GNNModel(
            in_channels=4,  # Change from 512 to 4
            hidden_channels=256,
            num_classes=num_classes
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_classes, num_classes)
        )

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        # MIL forward pass
        mil_logits, mil_attention = self.mil_model(batch['patches'])
        
        # GNN forward pass
        gnn_logits, gnn_features = self.gnn_model(batch['graph'])
        
        # Combine predictions
        combined_features = torch.cat([mil_logits, gnn_logits], dim=1)
        final_logits = self.fusion(combined_features)
        
        # Store intermediate results
        outputs = {
            'mil_attention': mil_attention,
            'gnn_features': gnn_features,
            'mil_logits': mil_logits,
            'gnn_logits': gnn_logits
        }
        
        return final_logits, outputs

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits, _ = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch['label'])
        
        # Log metrics
        self.log('train_loss', loss)
        acc = (logits.argmax(1) == batch['label']).float().mean()
        self.log('train_acc', acc)
        
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        logits, outputs = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch['label'])
        
        # Log metrics
        self.log('val_loss', loss)
        acc = (logits.argmax(1) == batch['label']).float().mean()
        self.log('val_acc', acc)
        
        # Log sub-model metrics
        mil_acc = (outputs['mil_logits'].argmax(1) == batch['label']).float().mean()
        gnn_acc = (outputs['gnn_logits'].argmax(1) == batch['label']).float().mean()
        self.log('val_mil_acc', mil_acc)
        self.log('val_gnn_acc', gnn_acc)
        
        return {
            'val_loss': loss,
            'val_acc': acc,
            'val_mil_acc': mil_acc,
            'val_gnn_acc': gnn_acc
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }