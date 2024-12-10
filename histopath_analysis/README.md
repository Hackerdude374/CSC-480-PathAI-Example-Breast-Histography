uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000

streamlit run frontend/app.py

python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --port 5000