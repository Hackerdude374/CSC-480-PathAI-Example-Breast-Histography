import pytorch_lightning as pl
import mlflow
import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
import json
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from src.models.combined_model import CombinedModel
from src.data.dataset import HistopathologyDataModule
from src.utils.metrics import MetricsTracker, calculate_confidence_metrics
from src.utils.visualization import (
    create_attention_heatmap,
    visualize_tissue_graph,
    plot_prediction_confidence
)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="training_config")
def evaluate(cfg: DictConfig):
    """
    Evaluate trained model on test dataset
    """
    # Load model from MLflow
    run_id = cfg.evaluation.run_id
    logger.info(f"Loading model from run ID: {run_id}")
    try:
        model = mlflow.pytorch.load_model(
            f"runs:/{run_id}/model",
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load model")

    # Initialize data module
    datamodule = HistopathologyDataModule(
        data_dir=cfg.data.test_path,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.training.num_workers,
        patch_size=cfg.data.patch_size,
        num_patches=cfg.data.num_patches
    )
    datamodule.setup('test')

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Create output directory
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate model
    results = []
    with torch.no_grad():
        for batch in tqdm(datamodule.test_dataloader(), desc="Evaluating"):
            # Move batch to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get predictions
            logger.debug("Forwarding batch through model")
            logits, outputs = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Update metrics
            logger.debug("Updating metrics")
            metrics_tracker.update(
                predictions.cpu(),
                batch['label'].cpu(),
                probabilities.cpu(),
                outputs['mil_attention'].cpu()
            )
            
            # Store results for each sample
            for i in range(len(predictions)):
                result = {
                    'patient_id': batch['patient_id'][i],
                    'true_label': batch['label'][i].item(),
                    'predicted_label': predictions[i].item(),
                    'confidence': probabilities[i].max().item(),
                    'probabilities': probabilities[i].tolist(),
                    'attention_weights': outputs['mil_attention'][i].tolist()
                }
                results.append(result)
                
                # Save visualizations if configured
                if cfg.evaluation.save_visualizations:
                    logger.debug("Saving visualizations")
                    patient_dir = output_dir / 'visualizations' / batch['patient_id'][i]
                    patient_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Attention heatmap
                    attention_map = create_attention_heatmap(
                        batch['patches'][i],
                        outputs['mil_attention'][i]
                    )
                    plt.imsave(
                        patient_dir / 'attention_heatmap.png',
                        attention_map
                    )
                    
                    # Tissue graph
                    graph_fig = visualize_tissue_graph(
                        batch['graph'],
                        outputs['gnn_features'][i]
                    )
                    graph_fig.write_html(patient_dir / 'tissue_graph.html')
    
    # Compute and save metrics
    logger.info("Computing and saving metrics")
    metrics = metrics_tracker.compute_metrics()
    metrics.update(calculate_confidence_metrics(
        np.array([r['probabilities'] for r in results])
    ))
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create and save confusion matrix
    logger.info("Generating and saving confusion matrix")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(
        [r['true_label'] for r in results],
        [r['predicted_label'] for r in results]
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    
    # Generate classification report
    logger.info("Generating and saving classification report")
    report = classification_report(
        [r['true_label'] for r in results],
        [r['predicted_label'] for r in results],
        target_names=['Benign', 'Malignant'],
        output_dict=True
    )
    pd.DataFrame(report).to_csv(output_dir / 'classification_report.csv')
    
    # Create confidence distribution plot
    logger.info("Generating and saving confidence distribution plot")
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=pd.DataFrame(results),
        x='confidence',
        hue='true_label',
        bins=30,
        multiple="layer"
    )
    plt.title('Confidence Distribution by Class')
    plt.savefig(output_dir / 'confidence_distribution.png')
    plt.close()
    
    # Save detailed results
    logger.info("Saving detailed results")
    pd.DataFrame(results).to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # Log results to MLflow
    logger.info("Logging results to MLflow")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir))
        
        # Log confusion matrix as a figure
        mlflow.log_figure(
            plt.figure(figsize=(10, 8)),
            'confusion_matrix.png'
        )
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
    print(f"\nDetailed results saved to: {output_dir}")

def analyze_failures(results_df: pd.DataFrame, output_dir: Path):
    """Analyze misclassified cases"""
    misclassified = results_df[
        results_df['true_label'] != results_df['predicted_label']
    ]
    
    # Analyze confidence distribution for misclassified cases
    plt.figure(figsize=(10, 6))
    sns.histplot(data=misclassified, x='confidence', bins=20)
    plt.title('Confidence Distribution for Misclassified Cases')
    plt.savefig(output_dir / 'misclassified_confidence.png')
    plt.close()
    
    # Analyze patterns in misclassification
    error_analysis = {
        'total_misclassified': len(misclassified),
        'false_positives': len(misclassified[
            misclassified['predicted_label'] == 1
        ]),
        'false_negatives': len(misclassified[
            misclassified['predicted_label'] == 0
        ]),
        'mean_confidence': misclassified['confidence'].mean(),
        'high_confidence_errors': len(misclassified[
            misclassified['confidence'] > 0.9
        ])
    }
    
    # Save error analysis
    with open(output_dir / 'error_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=4)

if __name__ == "__main__":
    evaluate()