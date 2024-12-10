import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track and compute various evaluation metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.attention_weights = []
        
    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor,
        attention: torch.Tensor = None
    ):
        """Update metrics with batch results"""
        self.predictions.extend(preds.cpu().numpy())
        self.true_labels.extend(labels.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
        if attention is not None:
            self.attention_weights.extend(attention.cpu().numpy())
            
    def compute_metrics(self) -> Dict:
        """Compute all metrics"""
        try:
            metrics = {
                'accuracy': accuracy_score(
                    self.true_labels,
                    self.predictions
                ),
                'precision': precision_score(
                    self.true_labels,
                    self.predictions,
                    average='weighted'
                ),
                'recall': recall_score(
                    self.true_labels,
                    self.predictions,
                    average='weighted'
                ),
                'f1': f1_score(
                    self.true_labels,
                    self.predictions,
                    average='weighted'
                ),
                'roc_auc': roc_auc_score(
                    self.true_labels,
                    self.probabilities[:, 1]  # Probability of positive class
                ),
                'confusion_matrix': confusion_matrix(
                    self.true_labels,
                    self.predictions
                ).tolist()
            }
            
            # Add attention statistics if available
            if self.attention_weights:
                metrics.update(self._compute_attention_metrics())
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {}
            
    def _compute_attention_metrics(self) -> Dict:
        """Compute attention-specific metrics"""
        attention_array = np.array(self.attention_weights)
        return {
            'attention_mean': float(np.mean(attention_array)),
            'attention_std': float(np.std(attention_array)),
            'attention_entropy': float(self._compute_attention_entropy(attention_array))
        }
        
    @staticmethod
    def _compute_attention_entropy(attention: np.ndarray) -> float:
        """Compute entropy of attention weights"""
        epsilon = 1e-10
        attention = np.clip(attention, epsilon, 1.0)
        return float(-np.sum(attention * np.log(attention)) / len(attention))

def calculate_confidence_metrics(
    probabilities: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """Calculate confidence-based metrics"""
    confidence_scores = np.max(probabilities, axis=1)
    high_confidence = confidence_scores >= threshold
    
    return {
        'mean_confidence': float(np.mean(confidence_scores)),
        'high_confidence_ratio': float(np.mean(high_confidence)),
        'confidence_std': float(np.std(confidence_scores))
    }

def compute_class_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Compute per-class metrics"""
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        class_metrics[class_name] = {
            'precision': precision_score(
                true_labels == i,
                predictions == i,
                average='binary'
            ),
            'recall': recall_score(
                true_labels == i,
                predictions == i,
                average='binary'
            ),
            'f1': f1_score(
                true_labels == i,
                predictions == i,
                average='binary'
            ),
            'support': int(np.sum(class_mask))
        }
        
    return class_metrics

def calibration_metrics(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute calibration metrics including ECE (Expected Calibration Error)
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == true_labels
    
    # Create confidence bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in current bin
        in_bin = np.logical_and(
            confidences > bin_lower,
            confidences <= bin_upper
        )
        
        if any(in_bin):
            bin_conf = np.mean(confidences[in_bin])
            bin_acc = np.mean(accuracies[in_bin])
            bin_size = np.sum(in_bin)
            
            ece += bin_size * np.abs(bin_conf - bin_acc)
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
            
    ece = ece / len(confidences)
    
    return ece, np.array(bin_confidences), np.array(bin_accuracies)