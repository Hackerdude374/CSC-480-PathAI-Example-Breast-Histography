uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000

streamlit run frontend/app.py

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

put the kaggle archive.zip under data/raw
unzip it 

then make 3 new folders make processed under data 

then in processed make test, train, and val


then run organize_dataset.py and train.py under scripts folder