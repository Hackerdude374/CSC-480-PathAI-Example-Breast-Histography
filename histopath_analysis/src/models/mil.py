import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from typing import Tuple, Dict

class AttentionPool(nn.Module):
    """Attention pooling layer for MIL"""
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_features = torch.sum(x * attention_weights, dim=1)
        return weighted_features, attention_weights

class MILModel(pl.LightningModule):
    """Multiple Instance Learning model for histopathology analysis"""
    def __init__(self, num_classes: int = 2, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature extractor (ResNet18 backbone)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # Attention pooling
        self.attention_pool = AttentionPool(self.feature_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_instances = x.size(0), x.size(1)
        x = x.view(-1, *x.size()[2:])
        
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_instances, -1)
        
        # Pool features with attention
        pooled_features, attention_weights = self.attention_pool(features)
        
        # Classify
        logits = self.classifier(pooled_features)
        return logits, attention_weights

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch['image'], batch['label']
        logits, _ = self(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log('train_acc', acc)
        
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        images, labels = batch['image'], batch['label']
        logits, attention = self(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Log metrics
        self.log('val_loss', loss)
        acc = (logits.argmax(1) == labels).float().mean()
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