from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, List
import mlflow
from pathlib import Path

from ..models.combined_model import CombinedModel
from ..data.preprocessing import extract_patches, create_tissue_graph
from ..utils.visualization import create_attention_heatmap

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Histopathology Analysis API",
    description="API for analyzing histopathology images using MIL-GNN model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelService:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = None
        self.load_model()

    def load_model(self):
        try:
            # Load model from MLflow
            print("Loading model from MLflow...")
            logged_model = mlflow.pytorch.load_model(
                f"runs:/latest/model",
                map_location=self.device
            )
            self.model = logged_model.eval()
            logging.info(f"Model loaded successfully on {self.device}")
            print("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            self.model = None
            print(f"Error loading model: {str(e)}")
            raise RuntimeError("Failed to load model")

    async def process_image(self, image: Image.Image) -> Dict:
        """Process a single image"""
        try:
            # Extract patches
            print("Extracting patches from image...")
            patches = extract_patches(image, num_patches=100)
            
            # Transform patches
            print("Transforming patches...")
            patch_tensors = torch.stack([
                self.transform(patch) for patch in patches
            ]).unsqueeze(0)
            
            # Create tissue graph
            print("Creating tissue graph...")
            graph = create_tissue_graph(patch_tensors[0])
            
            # Prepare input batch
            batch = {
                'patches': patch_tensors.to(self.device),
                'graph': graph.to(self.device)
            }
            
            # Get predictions
            print("Getting predictions from model...")
            with torch.no_grad():
                logits, outputs = self.model(batch)
                probabilities = torch.softmax(logits, dim=1)
                
            # Create attention heatmap
            print("Creating attention heatmap...")
            heatmap = create_attention_heatmap(
                image,
                outputs['mil_attention'][0],
                patch_size=50
            )
            
            return {
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'attention_weights': outputs['mil_attention'][0].cpu().numpy().tolist(),
                'heatmap': heatmap,
                'gnn_features': outputs['gnn_features'][0].cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            print(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )

model_service = ModelService()

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Analyze histopathology image
    
    Parameters:
    - file: Image file (PNG, JPG, or TIFF format)
    
    Returns:
    - Dictionary containing predictions and visualizations
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload PNG, JPG, or TIFF images."
            )
        
        # Read image
        print("Reading image file...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Process image
        print("Processing image...")
        results = await model_service.process_image(image)
        
        return {
            'status': 'success',
            'predictions': {
                'class_probabilities': results['probabilities'],
                'predicted_class': int(np.argmax(results['probabilities'])),
                'confidence': float(np.max(results['probabilities']))
            },
            'visualizations': {
                'attention_heatmap': results['heatmap'],
                'attention_weights': results['attention_weights']
            },
            'metadata': {
                'image_size': image.size,
                'model_version': model_service.model.hparams.get('version', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)) -> Dict:
    """
    Analyze multiple histopathology images
    
    Parameters:
    - files: List of image files
    
    Returns:
    - Dictionary containing predictions for all images
    """
    results = []
    for file in files:
        try:
            print(f"Processing file: {file.filename}")
            result = await predict(file)
            results.append({
                'filename': file.filename,
                'predictions': result
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {'results': results}

@app.get("/model-info")
async def get_model_info() -> Dict:
    """Get information about the loaded model"""
    return {
        'model_type': type(model_service.model).__name__,
        'device': str(model_service.device),
        'model_parameters': model_service.model.hparams
    }

@app.get("/health")
async def health_check() -> Dict:
    """Check if the service is healthy"""
    return {
        'status': 'healthy',
        'model_loaded': model_service.model is not None
    }