import pytest
from fastapi.testclient import TestClient
from src.api.endpoints import app
import io
from PIL import Image
import numpy as np
import json

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def sample_image():
    """Create sample image for testing"""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

class TestAPI:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
    def test_model_info(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        assert "model_type" in response.json()
        
    def test_predict_endpoint(self, client, sample_image):
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "visualizations" in data
        
        # Check prediction structure
        pred = data["predictions"]
        assert "class_probabilities" in pred
        assert "predicted_class" in pred
        assert "confidence" in pred
        
        # Validate probability values
        probs = pred["class_probabilities"]
        assert len(probs) == 2  # Binary classification
        assert all(0 <= p <= 1 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6  # Sum to 1
        
    def test_batch_predict(self, client, sample_image):
        files = [
            ("files", ("test1.png", sample_image, "image/png")),
            ("files", ("test2.png", sample_image, "image/png"))
        ]
        response = client.post("/batch-predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        
    @pytest.mark.parametrize("file_ext", ["jpg", "tiff", "bmp"])
    def test_invalid_file_format(self, client, file_ext):
        invalid_image = io.BytesIO(b"invalid image content")
        files = {"file": (f"test.{file_ext}", invalid_image, "image/jpeg")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]
        
    def test_error_handling(self, client):
        # Test with invalid file
        files = {"file": ("test.png", io.BytesIO(b""), "image/png")}
        response = client.post("/predict", files=files)
        assert response.status_code == 500
        
    def test_concurrent_requests(self, client, sample_image):
        import concurrent.futures
        
        def make_request():
            files = {"file": ("test.png", sample_image, "image/png")}
            return client.post("/predict", files=files)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in futures]
            
        assert all(r.status_code == 200 for r in responses)