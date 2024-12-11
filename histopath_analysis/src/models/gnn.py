import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Dict, Tuple

class GNNModel(pl.LightningModule):
    """Graph Neural Network for tissue structure analysis"""
    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        num_classes: int = 2,
        num_layers: int = 3,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GNN layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        logits = self.mlp(x)
        
        return logits, x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        logits, _ = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch.y)
        
        # Log metrics
        self.log('train_loss', loss)
        acc = (logits.argmax(1) == batch.y).float().mean()
        self.log('train_acc', acc)
        
        return loss

    def validation_step(self, batch, batch_idx) -> Dict:
        logits, _ = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch.y)
        
        # Log metrics
        self.log('val_loss', loss)
        acc = (logits.argmax(1) == batch.y).float().mean()
        self.log('val_acc', acc)
        
        return {'val_loss': loss, 'val_acc': acc}

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