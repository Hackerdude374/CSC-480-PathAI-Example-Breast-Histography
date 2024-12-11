from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        self.transform = self._setup_transform()
        
    def _setup_transform(self):
        return transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    async def process_image(self, image: Image.Image) -> Dict:
        """Process a single image"""
        try:
            # Since model is not loaded, return dummy predictions
            return {
                'probabilities': [0.6, 0.4],  # Dummy probabilities
                'attention_weights': [[0.5] * 100],  # Dummy attention weights
                'heatmap': np.zeros((224, 224, 3), dtype=np.uint8),  # Dummy heatmap
                'gnn_features': [[0.0] * 256]  # Dummy GNN features
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )

model_service = ModelService()

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    try:
        # Log received file info
        print(f"Received file: {file.filename}")
        print(f"Content type: {file.content_type}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload PNG, JPG, or TIFF images."
            )
        
        # Read file contents
        contents = await file.read()
        print(f"Read file size: {len(contents)} bytes")
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )
        
        # Create BytesIO object and open image
        image_bytes = io.BytesIO(contents)
        try:
            image = Image.open(image_bytes)
            image.load()  # Force load the image data
            print(f"Image opened successfully: size={image.size}, mode={image.mode}")
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error opening image: {str(e)}"
            )

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted image to RGB mode")

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
                'attention_heatmap': results['heatmap'].tolist(),
                'attention_weights': results['attention_weights']
            },
            'metadata': {
                'image_size': image.size,
                'image_mode': image.mode
            }
        }
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict:
    """Check if the service is healthy"""
    return {
        'status': 'healthy',
        'model_loaded': model_service.model is not None
    }

@app.get("/model-info")
async def get_model_info() -> Dict:
    """Get information about the loaded model"""
    return {
        'model_type': "Dummy Model (No model loaded)",
        'device': str(model_service.device),
        'status': 'Test mode - No model loaded'
    }
    
    


@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)) -> Dict:
    results = []
    for file in files:
        try:
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